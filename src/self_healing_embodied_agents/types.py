from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ActionKind(str, Enum):
    MOVE_TO_OBJECT = "move_to_object"
    GRASP = "grasp"
    MOVE_TO_TARGET = "move_to_target"
    PLACE = "place"
    REOBSERVE = "reobserve"
    CLEAR_PATH = "clear_path"


@dataclass(frozen=True)
class Action:
    kind: ActionKind

    def __str__(self) -> str:
        return self.kind.value


@dataclass
class WorldState:
    ee_xy: np.ndarray
    object_xy: np.ndarray
    target_xy: np.ndarray
    holding: bool = False
    object_visible: bool = True
    path_blocked: bool = False
    success: bool = False
    step_index: int = 0

    def copy(self) -> "WorldState":
        return WorldState(
            ee_xy=self.ee_xy.copy(),
            object_xy=self.object_xy.copy(),
            target_xy=self.target_xy.copy(),
            holding=bool(self.holding),
            object_visible=bool(self.object_visible),
            path_blocked=bool(self.path_blocked),
            success=bool(self.success),
            step_index=int(self.step_index),
        )

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.ee_xy[0],
                self.ee_xy[1],
                self.object_xy[0],
                self.object_xy[1],
                self.target_xy[0],
                self.target_xy[1],
                float(self.holding),
                float(self.object_visible),
                float(self.path_blocked),
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_vector(cls, x: np.ndarray, *, step_index: int = 0) -> "WorldState":
        x = np.asarray(x, dtype=np.float32)
        return cls(
            ee_xy=x[0:2].copy(),
            object_xy=x[2:4].copy(),
            target_xy=x[4:6].copy(),
            holding=bool(x[6] >= 0.5),
            object_visible=bool(x[7] >= 0.5),
            path_blocked=bool(x[8] >= 0.5),
            step_index=step_index,
        )


@dataclass
class StepResult:
    state: WorldState
    action_succeeded: bool
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    perturbation: str
    agent: str
    seed: int
    interventions: int = 0
    true_failures: int = 0
    detections: int = 0
    true_positive_detections: int = 0
    false_positive_detections: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    event_log: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        precision = (
            self.true_positive_detections / self.detections if self.detections else float("nan")
        )
        recall = (
            self.true_positive_detections / self.true_failures if self.true_failures else float("nan")
        )
        return {
            "agent": self.agent,
            "perturbation": self.perturbation,
            "seed": self.seed,
            "success": int(self.success),
            "steps": self.steps,
            "interventions": self.interventions,
            "true_failures": self.true_failures,
            "detections": self.detections,
            "true_positive_detections": self.true_positive_detections,
            "false_positive_detections": self.false_positive_detections,
            "detection_precision": precision,
            "detection_recall": recall,
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
        }
