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

    def __post_init__(self) -> None:
        try:
            kind = ActionKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown action kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)

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

    def __post_init__(self) -> None:
        for name in ("ee_xy", "object_xy", "target_xy"):
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    value = np.asarray(getattr(self, name), dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must contain two finite coordinates") from exc
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain two finite coordinates")
            setattr(self, name, value.copy())
        for name in ("holding", "object_visible", "path_blocked", "success"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, bool(value))
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a nonnegative integer")

    def copy(self) -> "WorldState":
        return WorldState(
            ee_xy=self.ee_xy,
            object_xy=self.object_xy,
            target_xy=self.target_xy,
            holding=self.holding,
            object_visible=self.object_visible,
            path_blocked=self.path_blocked,
            success=self.success,
            step_index=self.step_index,
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
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.asarray(x, dtype=np.float32)
        if x.shape != (9,) or not np.all(np.isfinite(x)):
            raise ValueError("state vector must have shape (9,) and finite values")
        if np.any((x[6:] < 0) | (x[6:] > 1)):
            raise ValueError("state vector flags must be in [0, 1]")
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
    root_seed: int | None = None
    repetition: int | None = None

    def as_dict(self) -> dict[str, Any]:
        precision = (
            self.true_positive_detections / self.detections if self.detections else None
        )
        recall = (
            self.true_positive_detections / self.true_failures if self.true_failures else None
        )
        return {
            "agent": self.agent,
            "perturbation": self.perturbation,
            "seed": self.seed,
            "root_seed": self.root_seed,
            "repetition": self.repetition,
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
