from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch
from torch import nn

from .env import EnvConfig, nominal_transition
from .types import Action, ActionKind, WorldState

ACTION_ORDER = list(ActionKind)
ACTION_TO_INDEX = {kind: idx for idx, kind in enumerate(ACTION_ORDER)}
STATE_DIM = 9
ACTION_DIM = len(ACTION_ORDER)


def encode_action(action: Action) -> np.ndarray:
    x = np.zeros(ACTION_DIM, dtype=np.float32)
    x[ACTION_TO_INDEX[action.kind]] = 1.0
    return x


class TransitionMLP(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, STATE_DIM),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


@dataclass
class ModelBundle:
    model: TransitionMLP
    residual_threshold: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            isinstance(self.residual_threshold, bool)
            or not isinstance(self.residual_threshold, Real)
            or not np.isfinite(self.residual_threshold)
            or self.residual_threshold <= 0
        ):
            raise ValueError("residual_threshold must be finite and positive")
        if not isinstance(self.model, TransitionMLP):
            raise TypeError("model must be a TransitionMLP")
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("model contains non-finite parameters")

    def predict(self, state: WorldState, action: Action) -> np.ndarray:
        self.validate()
        vector = state.vector()
        if vector.shape != (STATE_DIM,) or not np.isfinite(vector).all():
            raise ValueError("state must contain nine finite values")
        parameter = next(self.model.parameters())
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.no_grad():
                s = torch.as_tensor(vector, device=parameter.device, dtype=parameter.dtype).unsqueeze(0)
                a = torch.as_tensor(
                    encode_action(action), device=parameter.device, dtype=parameter.dtype
                ).unsqueeze(0)
                prediction = self.model(s, a).squeeze(0)
                if prediction.shape != (STATE_DIM,) or not torch.isfinite(prediction).all():
                    raise FloatingPointError("transition model produced an invalid prediction")
                # NumPy has no bfloat16 type; retain double precision when requested.
                output_dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
                return prediction.to(device="cpu", dtype=output_dtype).numpy()
        finally:
            for module, training in modes:
                module.training = training


class SymbolicCounterfactualModel:
    """Nominal simulator dynamics, without injecting future perturbations."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config if config is not None else EnvConfig()

    def transition(self, state: WorldState, action: Action) -> WorldState:
        return nominal_transition(state, action, self.config).state

    def rollout(self, state: WorldState, actions: list[Action]) -> WorldState:
        s = state.copy()
        for action in actions:
            if s.success or s.step_index >= self.config.max_steps:
                break
            s = self.transition(s, action)
        return s
