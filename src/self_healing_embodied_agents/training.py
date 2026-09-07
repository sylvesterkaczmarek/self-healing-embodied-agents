from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import TabletopManipulationEnv
from .types import Action, ActionKind
from .world_model import (
    ACTION_DIM,
    ACTION_ORDER,
    STATE_DIM,
    ModelBundle,
    TransitionMLP,
    encode_action,
)


@dataclass
class TrainingMetrics:
    seed: int
    train_samples: int
    validation_samples: int
    validation_mse: float
    residual_threshold: float
    epochs: int = 120
    learning_rate: float = 2e-3
    train_episodes: int = 0
    validation_episodes: int = 0
    train_episode_ids: list[int] = field(default_factory=list)
    validation_episode_ids: list[int] = field(default_factory=list)
    split_unit: str = "episode"
    validation_role: str = "nominal residual calibration; not an independent detector test"
    environment_seed_rule: str = "SeedSequence([seed, episode_id, 0]) generates one uint64"
    calibration_quantile: float = 0.995
    calibration_multiplier: float = 1.35
    calibration_floor: float = 0.035


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def set_determinism(seed: int) -> None:
    _integer("seed", seed, 0)
    if seed >= 2**32:
        raise ValueError("seed must be less than 2**32")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _collect_nominal_transitions(
    seed: int, episodes: int, *, return_episode_ids: bool = False
) -> tuple[np.ndarray, ...]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    episode_ids: list[int] = []
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        environment_seed = int(
            np.random.SeedSequence([seed, ep, 0]).generate_state(1, dtype=np.uint64)[0]
        )
        env = TabletopManipulationEnv(seed=environment_seed, perturbation="none")
        state = env.reset()
        plan = env.nominal_plan(state)
        if rng.random() < 0.25:
            plan = [Action(ActionKind.REOBSERVE), *plan]
        for action in plan:
            result = env.step(action)
            states.append(state.vector())
            actions.append(encode_action(action))
            next_states.append(result.state.vector())
            episode_ids.append(ep)
            state = result.state
            if state.success:
                break

    arrays = (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
    )
    if return_episode_ids:
        return (*arrays, np.asarray(episode_ids, dtype=np.int64))
    return arrays


def _require_finite(value: torch.Tensor, description: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite {description}; training stopped")


def train_transition_model(
    *,
    seed: int = 7,
    episodes: int = 500,
    epochs: int = 120,
    learning_rate: float = 2e-3,
) -> tuple[ModelBundle, TrainingMetrics]:
    _integer("episodes", episodes, 2)
    _integer("epochs", epochs, 1)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    set_determinism(seed)
    states, actions, targets, episode_ids = _collect_nominal_transitions(
        seed, episodes, return_episode_ids=True
    )
    # Every transition of an episode stays in one partition. Repeated positions
    # from the same trajectory must not appear in training and calibration.
    order = np.random.default_rng(seed).permutation(episodes)
    split = max(1, int(episodes * 0.8))
    train_episodes, val_episodes = order[:split], order[split:]
    train_idx = np.flatnonzero(np.isin(episode_ids, train_episodes))
    val_idx = np.flatnonzero(np.isin(episode_ids, val_episodes))

    model = TransitionMLP(hidden=64)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    s_train = torch.from_numpy(states[train_idx])
    a_train = torch.from_numpy(actions[train_idx])
    y_train = torch.from_numpy(targets[train_idx])
    for tensor in (s_train, a_train, y_train):
        _require_finite(tensor, "training data")

    model.train()
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        pred = model(s_train, a_train)
        _require_finite(pred, "prediction")
        loss = loss_fn(pred, y_train)
        _require_finite(loss, "loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                _require_finite(parameter.grad, "gradient")
        optim.step()
        for parameter in model.parameters():
            _require_finite(parameter, "updated parameter")

    model.eval()
    with torch.no_grad():
        s_val = torch.from_numpy(states[val_idx])
        a_val = torch.from_numpy(actions[val_idx])
        y_val = torch.from_numpy(targets[val_idx])
        pred = model(s_val, a_val)
        _require_finite(pred, "calibration prediction")
        squared_error = (pred - y_val) ** 2
        _require_finite(squared_error, "calibration error")
        per_sample = torch.sqrt(torch.mean(squared_error, dim=1)).cpu().numpy()
        val_mse = float(torch.mean(squared_error).item())

    # A nominal-data heuristic, not a finite-sample false-alarm guarantee.
    threshold = float(max(np.quantile(per_sample, 0.995) * 1.35, 0.035))
    metrics = TrainingMetrics(
        seed=seed,
        train_samples=len(train_idx),
        validation_samples=len(val_idx),
        validation_mse=val_mse,
        residual_threshold=threshold,
        epochs=epochs,
        learning_rate=float(learning_rate),
        train_episodes=len(train_episodes),
        validation_episodes=len(val_episodes),
        train_episode_ids=sorted(train_episodes.tolist()),
        validation_episode_ids=sorted(val_episodes.tolist()),
    )
    return ModelBundle(model=model, residual_threshold=threshold), metrics


def save_bundle(bundle: ModelBundle, metrics: TrainingMetrics, path: Path, metrics_path: Path) -> None:
    bundle.validate()
    if metrics.residual_threshold != bundle.residual_threshold:
        raise ValueError("metrics and bundle residual thresholds must match")
    if metrics.validation_mse < 0:
        raise ValueError("validation_mse must be non-negative")
    metrics_text = json.dumps(asdict(metrics), indent=2, allow_nan=False) + "\n"
    path, metrics_path = Path(path), Path(metrics_path)
    if path.resolve() == metrics_path.resolve():
        raise ValueError("checkpoint and metrics paths must be different")
    for destination in (path, metrics_path):
        if destination.is_dir():
            raise IsADirectoryError(f"output path is a directory: {destination}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"output path must be a regular file: {destination}")
    parameters = list(bundle.model.parameters())
    if len({parameter.dtype for parameter in parameters}) != 1:
        raise ValueError("model parameters must share a dtype")
    payload = {
        "schema_version": 1,
        "state_dict": {key: value.detach().cpu() for key, value in bundle.model.state_dict().items()},
        "residual_threshold": float(bundle.residual_threshold),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_order": [kind.value for kind in ACTION_ORDER],
        "hidden": bundle.model.hidden,
        "dtype": str(parameters[0].dtype).removeprefix("torch."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    retained_backups: set[Path] = set()
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as checkpoint:
            temporary_paths.append(Path(checkpoint.name))
            torch.save(payload, checkpoint)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=metrics_path.parent, suffix=".json", delete=False
        ) as output:
            temporary_paths.append(Path(output.name))
            output.write(metrics_text)
        backups: dict[Path, Path | None] = {}
        for destination in (path, metrics_path):
            if destination.exists() or destination.is_symlink():
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=".backup", delete=False
                ) as backup:
                    backup_path = Path(backup.name)
                    temporary_paths.append(backup_path)
                # Preserve a destination symlink itself when restoring a failed save.
                if destination.is_symlink():
                    backup_path.unlink()
                shutil.copy2(destination, backup_path, follow_symlinks=False)
                backups[destination] = backup_path
            else:
                backups[destination] = None
        published: list[Path] = []
        try:
            for staged, destination in zip(temporary_paths[:2], (path, metrics_path), strict=True):
                os.replace(staged, destination)
                published.append(destination)
        except OSError as publish_error:
            rollback_errors: list[str] = []
            for destination in reversed(published):
                backup_path = backups[destination]
                try:
                    if backup_path is None:
                        destination.unlink(missing_ok=True)
                    else:
                        os.replace(backup_path, destination)
                except OSError as rollback_error:
                    if backup_path is not None:
                        retained_backups.add(backup_path)
                    rollback_errors.append(
                        f"{destination}: {rollback_error}; original backup: {backup_path}"
                    )
            if rollback_errors:
                raise RuntimeError(
                    "save failed and originals could not all be restored: "
                    + "; ".join(rollback_errors)
                ) from publish_error
            raise
    finally:
        for temporary in temporary_paths:
            if temporary not in retained_backups:
                temporary.unlink(missing_ok=True)


def load_bundle(path: Path) -> ModelBundle:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")  # noqa: TRY004
    version = payload.get("schema_version", 0)
    if type(version) is not int or version not in (0, 1):
        raise ValueError("unsupported checkpoint schema_version")
    for name, expected in (("state_dim", STATE_DIM), ("action_dim", ACTION_DIM)):
        if type(payload.get(name)) is not int or payload[name] != expected:
            raise ValueError(f"checkpoint {name} must be {expected}")
    if version == 1 and payload.get("action_order") != [kind.value for kind in ACTION_ORDER]:
        raise ValueError("checkpoint action_order does not match the action encoding")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint state_dict must be a non-empty dictionary")
    if any(
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        for value in state_dict.values()
    ):
        raise ValueError("checkpoint parameters must be finite floating-point tensors")
    dtypes = {value.dtype for value in state_dict.values()}
    if len(dtypes) != 1:
        raise ValueError("checkpoint parameters must share a dtype")
    dtype = next(iter(dtypes))
    if version == 1 and payload.get("dtype") != str(dtype).removeprefix("torch."):
        raise ValueError("checkpoint dtype does not match its parameters")
    hidden = payload.get("hidden", 64)
    if version == 1 and "hidden" not in payload:
        raise ValueError("checkpoint must record hidden width")
    _integer("checkpoint hidden", hidden, 1)
    first_weight = state_dict.get("net.0.weight")
    if first_weight is None or first_weight.shape != (hidden, STATE_DIM + ACTION_DIM):
        raise ValueError("checkpoint hidden width does not match its input layer")
    # Loading an inference model should not advance a caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        model = TransitionMLP(hidden=hidden).to(dtype=dtype)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"checkpoint parameters do not match the model: {error}") from error
    model.eval()
    return ModelBundle(model=model, residual_threshold=payload.get("residual_threshold"))
