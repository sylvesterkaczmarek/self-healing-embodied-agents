from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import TabletopManipulationEnv
from .types import Action, ActionKind
from .world_model import ACTION_DIM, STATE_DIM, ModelBundle, TransitionMLP, encode_action


@dataclass
class TrainingMetrics:
    seed: int
    train_samples: int
    validation_samples: int
    validation_mse: float
    residual_threshold: float


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _collect_nominal_transitions(seed: int, episodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        env = TabletopManipulationEnv(seed=seed * 1000 + ep, perturbation="none")
        state = env.reset()
        plan = env.nominal_plan(state)
        if rng.random() < 0.25:
            plan = [Action(ActionKind.REOBSERVE), *plan]
        for action in plan:
            result = env.step(action)
            states.append(state.vector())
            actions.append(encode_action(action))
            next_states.append(result.state.vector())
            state = result.state
            if state.success:
                break

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
    )


def train_transition_model(
    *,
    seed: int = 7,
    episodes: int = 500,
    epochs: int = 120,
    learning_rate: float = 2e-3,
) -> tuple[ModelBundle, TrainingMetrics]:
    set_determinism(seed)
    states, actions, targets = _collect_nominal_transitions(seed, episodes)
    n = len(states)
    split = max(1, int(n * 0.8))
    order = np.random.default_rng(seed).permutation(n)
    train_idx, val_idx = order[:split], order[split:]

    model = TransitionMLP(hidden=64)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    s_train = torch.from_numpy(states[train_idx])
    a_train = torch.from_numpy(actions[train_idx])
    y_train = torch.from_numpy(targets[train_idx])

    model.train()
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        pred = model(s_train, a_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        s_val = torch.from_numpy(states[val_idx])
        a_val = torch.from_numpy(actions[val_idx])
        y_val = torch.from_numpy(targets[val_idx])
        pred = model(s_val, a_val)
        per_sample = torch.sqrt(torch.mean((pred - y_val) ** 2, dim=1)).cpu().numpy()
        val_mse = float(torch.mean((pred - y_val) ** 2).item())

    # Conservative threshold calibrated from nominal validation residuals.
    threshold = float(max(np.quantile(per_sample, 0.995) * 1.35, 0.035))
    metrics = TrainingMetrics(
        seed=seed,
        train_samples=int(len(train_idx)),
        validation_samples=int(len(val_idx)),
        validation_mse=val_mse,
        residual_threshold=threshold,
    )
    return ModelBundle(model=model, residual_threshold=threshold), metrics


def save_bundle(bundle: ModelBundle, metrics: TrainingMetrics, path: Path, metrics_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": bundle.model.state_dict(),
            "residual_threshold": bundle.residual_threshold,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
        },
        path,
    )
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")


def load_bundle(path: Path) -> ModelBundle:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TransitionMLP(hidden=64)
    model.load_state_dict(payload["state_dict"])
    return ModelBundle(model=model, residual_threshold=float(payload["residual_threshold"]))
