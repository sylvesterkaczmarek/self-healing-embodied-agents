import json
import os

import numpy as np
import pytest
import torch

from self_healing_embodied_agents.env import TabletopManipulationEnv
from self_healing_embodied_agents.training import (
    TrainingMetrics,
    load_bundle,
    save_bundle,
)
from self_healing_embodied_agents.types import Action, ActionKind
from self_healing_embodied_agents.world_model import (
    ACTION_DIM,
    STATE_DIM,
    ModelBundle,
    TransitionMLP,
)


def bundle_and_metrics(hidden=64, dtype=torch.float32):
    bundle = ModelBundle(TransitionMLP(hidden).to(dtype=dtype), 0.1)
    metrics = TrainingMetrics(7, 10, 5, 0.01, 0.1)
    return bundle, metrics


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_threshold_rejected(threshold) -> None:
    with pytest.raises(ValueError, match="residual_threshold"):
        ModelBundle(TransitionMLP(), threshold)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_predict_preserves_precision_and_mixed_training_modes(dtype) -> None:
    bundle, _ = bundle_and_metrics(dtype=dtype)
    bundle.model.train()
    bundle.model.net[0].eval()
    before = [module.training for module in bundle.model.modules()]
    state = TabletopManipulationEnv().reset()
    prediction = bundle.predict(state, Action(ActionKind.MOVE_TO_OBJECT))
    assert prediction.dtype == (np.float64 if dtype == torch.float64 else np.float32)
    assert prediction.shape == (STATE_DIM,)
    assert np.isfinite(prediction).all()
    assert [module.training for module in bundle.model.modules()] == before
    assert next(bundle.model.parameters()).dtype == dtype


def test_prediction_overflow_is_rejected_and_mode_restored() -> None:
    bundle, _ = bundle_and_metrics()
    bundle.model.train()
    with torch.no_grad():
        for parameter in bundle.model.parameters():
            parameter.fill_(1e30)
    state = TabletopManipulationEnv().reset()
    with pytest.raises(FloatingPointError, match="prediction"):
        bundle.predict(state, Action(ActionKind.GRASP))
    assert bundle.model.training


def test_nonfinite_state_is_rejected() -> None:
    bundle, _ = bundle_and_metrics()
    state = TabletopManipulationEnv().reset()
    state.ee_xy[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bundle.predict(state, Action(ActionKind.GRASP))


def test_nonfinite_model_is_rejected_at_construction_and_prediction() -> None:
    bundle, _ = bundle_and_metrics()
    with torch.no_grad():
        next(bundle.model.parameters()).fill_(float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        ModelBundle(bundle.model, 0.1)
    with pytest.raises(ValueError, match="non-finite"):
        bundle.predict(TabletopManipulationEnv().reset(), Action(ActionKind.GRASP))


@pytest.mark.parametrize("hidden,dtype", [(64, torch.float32), (17, torch.float64)])
def test_checkpoint_round_trip_preserves_predictions(hidden, dtype, tmp_path) -> None:
    bundle, metrics = bundle_and_metrics(hidden, dtype)
    path, metrics_path = tmp_path / "model.pt", tmp_path / "metrics.json"
    save_bundle(bundle, metrics, path, metrics_path)
    rng = torch.random.get_rng_state().clone()
    restored = load_bundle(path)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert restored.model.hidden == hidden
    assert next(restored.model.parameters()).dtype == dtype
    assert not restored.model.training
    state, action = TabletopManipulationEnv(seed=5).reset(), Action(ActionKind.MOVE_TO_OBJECT)
    np.testing.assert_array_equal(bundle.predict(state, action), restored.predict(state, action))
    assert json.loads(metrics_path.read_text())["residual_threshold"] == 0.1


def test_legacy_checkpoint_remains_loadable(tmp_path) -> None:
    bundle, _ = bundle_and_metrics()
    path = tmp_path / "legacy.pt"
    torch.save({
        "state_dict": bundle.model.state_dict(), "residual_threshold": 0.1,
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
    }, path)
    restored = load_bundle(path)
    for name, parameter in bundle.model.state_dict().items():
        assert torch.equal(parameter, restored.model.state_dict()[name])


@pytest.mark.parametrize(
    "key,value",
    [
        ("state_dim", 999), ("action_dim", 999), ("state_dim", 9.0),
        ("residual_threshold", float("nan")), ("residual_threshold", 0),
        ("schema_version", 2), ("schema_version", True),
        ("action_order", ["wrong"]), ("dtype", "float64"), ("hidden", 0),
        ("hidden", 10**10),
        ("state_dict", {}), ("state_dict", {"wrong": "tensor"}),
    ],
)
def test_malformed_checkpoint_is_rejected(key, value, tmp_path) -> None:
    bundle, metrics = bundle_and_metrics()
    path = tmp_path / "model.pt"
    save_bundle(bundle, metrics, path, tmp_path / "metrics.json")
    payload = torch.load(path, weights_only=True)
    payload[key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_bundle(path)


def test_checkpoint_nonfinite_parameters_rejected(tmp_path) -> None:
    bundle, metrics = bundle_and_metrics()
    path = tmp_path / "model.pt"
    save_bundle(bundle, metrics, path, tmp_path / "metrics.json")
    payload = torch.load(path, weights_only=True)
    next(iter(payload["state_dict"].values())).fill_(float("nan"))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite"):
        load_bundle(path)


def test_invalid_metrics_do_not_overwrite_existing_artifacts(tmp_path) -> None:
    bundle, metrics = bundle_and_metrics()
    path, metrics_path = tmp_path / "model.pt", tmp_path / "metrics.json"
    path.write_bytes(b"existing checkpoint")
    metrics_path.write_text("existing metrics")
    metrics.validation_mse = float("nan")
    with pytest.raises(ValueError):
        save_bundle(bundle, metrics, path, metrics_path)
    assert path.read_bytes() == b"existing checkpoint"
    assert metrics_path.read_text() == "existing metrics"


def test_mismatched_thresholds_and_output_paths_rejected(tmp_path) -> None:
    bundle, metrics = bundle_and_metrics()
    metrics.residual_threshold = 0.2
    with pytest.raises(ValueError, match="thresholds must match"):
        save_bundle(bundle, metrics, tmp_path / "model.pt", tmp_path / "metrics.json")
    metrics.residual_threshold = 0.1
    with pytest.raises(ValueError, match="paths must be different"):
        save_bundle(bundle, metrics, tmp_path / "same", tmp_path / "same")


def test_directory_metrics_destination_preserves_existing_checkpoint(tmp_path) -> None:
    bundle, metrics = bundle_and_metrics()
    checkpoint, metrics_path = tmp_path / "model.pt", tmp_path / "metrics.json"
    checkpoint.write_bytes(b"original checkpoint")
    metrics_path.mkdir()
    with pytest.raises(IsADirectoryError, match="directory"):
        save_bundle(bundle, metrics, checkpoint, metrics_path)
    assert checkpoint.read_bytes() == b"original checkpoint"
    assert metrics_path.is_dir()
    assert set(tmp_path.iterdir()) == {checkpoint, metrics_path}


@pytest.mark.parametrize("existing_checkpoint", [False, True])
def test_second_publish_failure_restores_both_originals(
    existing_checkpoint, tmp_path, monkeypatch
) -> None:
    bundle, metrics = bundle_and_metrics()
    checkpoint, metrics_path = tmp_path / "model.pt", tmp_path / "metrics.json"
    if existing_checkpoint:
        checkpoint.write_bytes(b"original checkpoint")
    metrics_path.write_bytes(b"original metrics")
    replace = os.replace
    replacements = []

    def fail_second_replace(source, destination):
        replacements.append(destination)
        if len(replacements) == 2:
            raise OSError("forced metrics publication error")
        return replace(source, destination)

    monkeypatch.setattr("self_healing_embodied_agents.training.os.replace", fail_second_replace)
    with pytest.raises(OSError, match="forced metrics publication error"):
        save_bundle(bundle, metrics, checkpoint, metrics_path)
    assert replacements[:2] == [checkpoint, metrics_path]
    if existing_checkpoint:
        assert checkpoint.read_bytes() == b"original checkpoint"
    else:
        assert not checkpoint.exists()
    assert metrics_path.read_bytes() == b"original metrics"
    assert set(tmp_path.iterdir()) == (
        {checkpoint, metrics_path} if existing_checkpoint else {metrics_path}
    )
