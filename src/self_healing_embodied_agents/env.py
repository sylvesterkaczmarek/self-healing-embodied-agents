from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .types import Action, ActionKind, StepResult, WorldState


PERTURBATIONS = (
    "none",
    "grasp_slip",
    "object_displacement",
    "transient_occlusion",
    "blocked_path",
    "stale_observation",
    "compound_slip_block",
)


@dataclass
class EnvConfig:
    grasp_radius: float = 0.06
    placement_radius: float = 0.08
    workspace_low: float = 0.05
    workspace_high: float = 0.95
    max_steps: int = 24


class TabletopManipulationEnv:
    """Small deterministic manipulation testbed with injected execution faults.

    The environment is intentionally lightweight. It is a controlled benchmark for
    recovery logic, not a rigid-body physics simulator.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        perturbation: str = "none",
        config: EnvConfig | None = None,
    ) -> None:
        if perturbation not in PERTURBATIONS:
            raise ValueError(f"unknown perturbation: {perturbation}")
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.perturbation = perturbation
        self.config = config or EnvConfig()
        self.state: WorldState | None = None
        self._observation_history: list[WorldState] = []
        self._injected: set[str] = set()
        self._events: list[dict] = []
        self._pending_occlusion_reads = 0
        self._pending_stale_reads = 0

    def reset(self) -> WorldState:
        self._injected.clear()
        self._events.clear()
        self._observation_history.clear()
        self._pending_occlusion_reads = 0
        self._pending_stale_reads = 0

        object_xy = self.rng.uniform([0.18, 0.18], [0.42, 0.82]).astype(np.float32)
        target_xy = self.rng.uniform([0.63, 0.18], [0.88, 0.82]).astype(np.float32)
        ee_xy = np.asarray([0.1, 0.5], dtype=np.float32)
        self.state = WorldState(ee_xy=ee_xy, object_xy=object_xy, target_xy=target_xy)
        self._observation_history.append(self.state.copy())
        return self.observe()

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def _emit(self, name: str, *, failure: bool = False, **payload: object) -> dict:
        event = {
            "step": int(self.state.step_index if self.state else 0),
            "event": name,
            "failure": bool(failure),
            **payload,
        }
        self._events.append(event)
        return event

    def _clip(self, xy: np.ndarray) -> np.ndarray:
        return np.clip(xy, self.config.workspace_low, self.config.workspace_high).astype(np.float32)

    def _inject_before(self, action: Action) -> list[dict]:
        assert self.state is not None
        events: list[dict] = []

        if self.perturbation == "object_displacement" and "object_displacement" not in self._injected:
            if action.kind == ActionKind.GRASP:
                delta = self.rng.normal(0.0, 0.11, size=2).astype(np.float32)
                self.state.object_xy = self._clip(self.state.object_xy + delta)
                self._injected.add("object_displacement")
                events.append(self._emit("object_displacement", failure=True, dx=float(delta[0]), dy=float(delta[1])))

        if self.perturbation == "transient_occlusion" and "transient_occlusion" not in self._injected:
            if action.kind == ActionKind.MOVE_TO_OBJECT:
                self.state.object_visible = False
                self._pending_occlusion_reads = 2
                self._injected.add("transient_occlusion")
                events.append(self._emit("transient_occlusion", failure=True))

        if self.perturbation == "stale_observation" and "stale_observation" not in self._injected:
            if action.kind == ActionKind.GRASP:
                delta = self.rng.normal(0.0, 0.09, size=2).astype(np.float32)
                self.state.object_xy = self._clip(self.state.object_xy + delta)
                self._pending_stale_reads = 1
                self._injected.add("stale_observation")
                events.append(self._emit("stale_observation", failure=True, dx=float(delta[0]), dy=float(delta[1])))

        if self.perturbation in {"blocked_path", "compound_slip_block"} and "blocked_path" not in self._injected:
            if action.kind == ActionKind.MOVE_TO_TARGET and self.state.holding:
                self.state.path_blocked = True
                self._injected.add("blocked_path")
                events.append(self._emit("blocked_path", failure=True))

        return events

    def _inject_after(self, action: Action) -> list[dict]:
        assert self.state is not None
        events: list[dict] = []

        if self.perturbation in {"grasp_slip", "compound_slip_block"} and "grasp_slip" not in self._injected:
            if action.kind == ActionKind.MOVE_TO_TARGET and self.state.holding:
                self.state.holding = False
                slip = self.rng.normal(0.0, 0.08, size=2).astype(np.float32)
                self.state.object_xy = self._clip(self.state.ee_xy + slip)
                self._injected.add("grasp_slip")
                events.append(self._emit("grasp_slip", failure=True, dx=float(slip[0]), dy=float(slip[1])))

        return events

    def observe(self, *, fresh: bool = False) -> WorldState:
        assert self.state is not None

        if fresh:
            self._pending_occlusion_reads = 0
            self._pending_stale_reads = 0
            self.state.object_visible = True

        if self._pending_stale_reads > 0 and len(self._observation_history) >= 2:
            self._pending_stale_reads -= 1
            stale = self._observation_history[-2].copy()
            stale.step_index = self.state.step_index
            return stale

        obs = self.state.copy()
        if self._pending_occlusion_reads > 0:
            self._pending_occlusion_reads -= 1
            obs.object_visible = False

        self._observation_history.append(obs.copy())
        return obs

    def step(self, action: Action) -> StepResult:
        assert self.state is not None
        if self.state.success:
            return StepResult(self.observe(), True, [])

        self.state.step_index += 1
        events = self._inject_before(action)
        ok = True

        if action.kind == ActionKind.REOBSERVE:
            obs = self.observe(fresh=True)
            events.append(self._emit("reobserve"))
            return StepResult(obs, True, events)

        if action.kind == ActionKind.CLEAR_PATH:
            self.state.path_blocked = False
            events.append(self._emit("path_cleared"))

        elif action.kind == ActionKind.MOVE_TO_OBJECT:
            if not self.state.object_visible:
                ok = False
            else:
                self.state.ee_xy = self.state.object_xy.copy()

        elif action.kind == ActionKind.GRASP:
            distance = float(np.linalg.norm(self.state.ee_xy - self.state.object_xy))
            if not self.state.object_visible or distance > self.config.grasp_radius:
                ok = False
            else:
                self.state.holding = True
                self.state.object_xy = self.state.ee_xy.copy()

        elif action.kind == ActionKind.MOVE_TO_TARGET:
            if self.state.path_blocked:
                ok = False
            else:
                self.state.ee_xy = self.state.target_xy.copy()
                if self.state.holding:
                    self.state.object_xy = self.state.ee_xy.copy()

        elif action.kind == ActionKind.PLACE:
            if not self.state.holding:
                ok = False
            else:
                self.state.holding = False
                self.state.object_xy = self.state.ee_xy.copy()
                self.state.success = bool(
                    np.linalg.norm(self.state.object_xy - self.state.target_xy)
                    <= self.config.placement_radius
                )
                ok = self.state.success

        else:
            raise ValueError(action.kind)

        events.extend(self._inject_after(action))
        self._observation_history.append(self.state.copy())
        return StepResult(self.observe(), ok, events)

    def nominal_plan(self, state: WorldState | None = None) -> list[Action]:
        s = state or self.observe()
        if s.success:
            return []
        if not s.object_visible:
            return [Action(ActionKind.REOBSERVE)]
        if s.path_blocked:
            return [Action(ActionKind.CLEAR_PATH)]
        if s.holding:
            return [Action(ActionKind.MOVE_TO_TARGET), Action(ActionKind.PLACE)]
        return [
            Action(ActionKind.MOVE_TO_OBJECT),
            Action(ActionKind.GRASP),
            Action(ActionKind.MOVE_TO_TARGET),
            Action(ActionKind.PLACE),
        ]

    def rollout(self, actions: Iterable[Action]) -> WorldState:
        for action in actions:
            if self.state is None or self.state.success:
                break
            self.step(action)
        assert self.state is not None
        return self.state.copy()
