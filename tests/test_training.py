import numpy as np
import pytest
import torch

from self_healing_embodied_agents.training import (
    _collect_nominal_transitions,
    train_transition_model,
)


def test_training_calibrates_on_whole_unseen_episodes() -> None:
    bundle, metrics = train_transition_model(seed=1, episodes=30, epochs=10)
    assert metrics.validation_mse >= 0.0
    assert metrics.residual_threshold > 0.0
    assert metrics.train_episodes == 24
    assert metrics.validation_episodes == 6
    assert set(metrics.train_episode_ids).isdisjoint(metrics.validation_episode_ids)
    assert sorted(metrics.train_episode_ids + metrics.validation_episode_ids) == list(range(30))
    states, actions, targets, ids = _collect_nominal_transitions(1, 30, return_episode_ids=True)
    train = np.isin(ids, metrics.train_episode_ids)
    val = np.isin(ids, metrics.validation_episode_ids)
    assert int(train.sum()) == metrics.train_samples
    assert int(val.sum()) == metrics.validation_samples
    # All repeated target coordinates of a trajectory remain in one split.
    assert {tuple(x[4:6]) for x in states[train]}.isdisjoint(
        {tuple(x[4:6]) for x in states[val]}
    )
    with torch.no_grad():
        predictions = bundle.model(torch.from_numpy(states[val]), torch.from_numpy(actions[val]))
        error = (predictions - torch.from_numpy(targets[val])) ** 2
    expected_threshold = max(np.quantile(error.mean(dim=1).sqrt().numpy(), 0.995) * 1.35, 0.035)
    assert metrics.validation_mse == pytest.approx(error.mean().item())
    assert metrics.residual_threshold == pytest.approx(expected_threshold)
    assert metrics.epochs == 10
    assert metrics.learning_rate == 2e-3
    assert metrics.validation_role.startswith("nominal residual calibration")


def test_smallest_valid_split_has_training_and_calibration_data() -> None:
    _, metrics = train_transition_model(seed=0, episodes=2, epochs=1)
    assert metrics.train_episodes == metrics.validation_episodes == 1
    assert metrics.train_samples > 0
    assert metrics.validation_samples > 0


@pytest.mark.parametrize(
    "settings",
    [
        {"episodes": 0}, {"episodes": 1}, {"episodes": -2}, {"episodes": True},
        {"episodes": 2.0}, {"epochs": 0}, {"epochs": -1}, {"epochs": False},
        {"epochs": 1.5}, {"seed": -1}, {"seed": 2**32}, {"seed": True},
        {"seed": 1.0}, {"learning_rate": 0}, {"learning_rate": -0.1},
        {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
        {"learning_rate": True}, {"learning_rate": "0.01"},
    ],
)
def test_invalid_training_settings_fail_before_collection(settings, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("invalid settings must be rejected before collecting data")

    monkeypatch.setattr("self_healing_embodied_agents.training._collect_nominal_transitions", forbidden)
    with pytest.raises(ValueError):
        train_transition_model(**settings)


def test_nonfinite_optimizer_update_stops_training(monkeypatch) -> None:
    def corrupt_parameters(optim, *args, **kwargs):
        with torch.no_grad():
            optim.param_groups[0]["params"][0].fill_(float("nan"))

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_parameters)
    with pytest.raises(FloatingPointError, match="updated parameter"):
        train_transition_model(seed=2, episodes=3, epochs=1)


def test_training_repeats_exactly_for_same_seed() -> None:
    first, first_metrics = train_transition_model(seed=3, episodes=5, epochs=2)
    second, second_metrics = train_transition_model(seed=3, episodes=5, epochs=2)
    assert first_metrics == second_metrics
    for name, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, second.model.state_dict()[name])


def test_collector_default_return_contract_is_unchanged() -> None:
    arrays = _collect_nominal_transitions(7, 2)
    assert len(arrays) == 3
    assert len(arrays[0]) == len(arrays[1]) == len(arrays[2])
