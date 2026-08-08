from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

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
        super().__init__()
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

    def predict(self, state: WorldState, action: Action) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            s = torch.from_numpy(state.vector()).float().unsqueeze(0)
            a = torch.from_numpy(encode_action(action)).float().unsqueeze(0)
            pred = self.model(s, a).squeeze(0).cpu().numpy()
        return pred.astype(np.float32)


class SymbolicCounterfactualModel:
    """Deterministic nominal skill model used for candidate recovery rollouts."""

    def transition(self, state: WorldState, action: Action) -> WorldState:
        s = state.copy()
        if action.kind == ActionKind.REOBSERVE:
            s.object_visible = True
        elif action.kind == ActionKind.CLEAR_PATH:
            s.path_blocked = False
        elif action.kind == ActionKind.MOVE_TO_OBJECT and s.object_visible:
            s.ee_xy = s.object_xy.copy()
        elif action.kind == ActionKind.GRASP:
            if s.object_visible and np.linalg.norm(s.ee_xy - s.object_xy) <= 0.08:
                s.holding = True
                s.object_xy = s.ee_xy.copy()
        elif action.kind == ActionKind.MOVE_TO_TARGET and not s.path_blocked:
            s.ee_xy = s.target_xy.copy()
            if s.holding:
                s.object_xy = s.ee_xy.copy()
        elif action.kind == ActionKind.PLACE and s.holding:
            s.holding = False
            s.object_xy = s.ee_xy.copy()
            s.success = bool(np.linalg.norm(s.object_xy - s.target_xy) <= 0.10)
        s.step_index += 1
        return s

    def rollout(self, state: WorldState, actions: list[Action]) -> WorldState:
        s = state.copy()
        for action in actions:
            s = self.transition(s, action)
        return s
