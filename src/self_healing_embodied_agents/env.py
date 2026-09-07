from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Real
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

    def __post_init__(self) -> None:
        for name in ("grasp_radius", "placement_radius", "workspace_low", "workspace_high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.grasp_radius < 0 or self.placement_radius < 0:
            raise ValueError("grasp_radius and placement_radius must be nonnegative")
        if self.workspace_low >= self.workspace_high:
            raise ValueError("workspace_low must be less than workspace_high")
        bounds = np.asarray([self.workspace_low, self.workspace_high], dtype=np.float64)
        if np.any(np.abs(bounds) > np.finfo(np.float32).max):
            raise ValueError("workspace bounds must fit finite float32 coordinates")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")


def nominal_transition(
    state: WorldState, action: Action, config: EnvConfig | None = None
) -> StepResult:
    """Apply one fault-free skill using the same contract as the environment.

    Completed and timed-out states are absorbing. The input state is never
    modified; an active action advances its step counter exactly once.
    """
    if not isinstance(action, Action):
        raise TypeError("action must be an Action")
    config = EnvConfig() if config is None else config
    if not isinstance(config, EnvConfig):
        raise TypeError("config must be an EnvConfig")
    config.__post_init__()
    s = state.copy()
    if s.success or s.step_index >= config.max_steps:
        return StepResult(s, s.success)
    s.step_index += 1
    ok = True
    if action.kind == ActionKind.REOBSERVE:
        s.object_visible = True
    elif action.kind == ActionKind.CLEAR_PATH:
        s.path_blocked = False
    elif action.kind == ActionKind.MOVE_TO_OBJECT:
        if not s.object_visible:
            ok = False
        else:
            s.ee_xy = s.object_xy.copy()
    elif action.kind == ActionKind.GRASP:
        distance = float(np.linalg.norm(s.ee_xy.astype(np.float64) - s.object_xy))
        if not s.object_visible or distance > config.grasp_radius:
            ok = False
        else:
            s.holding = True
            s.object_xy = s.ee_xy.copy()
    elif action.kind == ActionKind.MOVE_TO_TARGET:
        if s.path_blocked:
            ok = False
        else:
            s.ee_xy = s.target_xy.copy()
            if s.holding:
                s.object_xy = s.ee_xy.copy()
    elif action.kind == ActionKind.PLACE:
        if not s.holding:
            ok = False
        else:
            s.holding = False
            s.object_xy = s.ee_xy.copy()
            s.success = bool(
                np.linalg.norm(s.object_xy.astype(np.float64) - s.target_xy)
                <= config.placement_radius
            )
            ok = s.success
    return StepResult(s, ok)


def nominal_actions(state: WorldState, config: EnvConfig | None = None) -> list[Action]:
    """Plan from the observed state, omitting already completed motions.

    Exact coordinate equality identifies a completed motion. Grasp and placement
    tolerances remain action preconditions, not shortcuts for this planner.
    """
    config = EnvConfig() if config is None else config
    if not isinstance(config, EnvConfig):
        raise TypeError("config must be an EnvConfig")
    config.__post_init__()
    if state.success or state.step_index >= config.max_steps:
        return []
    if not state.object_visible:
        return [Action(ActionKind.REOBSERVE)]
    if state.path_blocked:
        return [Action(ActionKind.CLEAR_PATH)]
    plan: list[Action] = []
    transport_start = state.ee_xy
    if not state.holding:
        if not np.array_equal(state.ee_xy, state.object_xy):
            plan.append(Action(ActionKind.MOVE_TO_OBJECT))
        plan.append(Action(ActionKind.GRASP))
        transport_start = state.object_xy
    if not np.array_equal(transport_start, state.target_xy):
        plan.append(Action(ActionKind.MOVE_TO_TARGET))
    plan.append(Action(ActionKind.PLACE))
    return plan


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
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if config is not None and not isinstance(config, EnvConfig):
            raise TypeError("config must be an EnvConfig")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.perturbation = perturbation
        self.config = config or EnvConfig()
        self.state: WorldState | None = None
        self._observation_history: deque[WorldState] = deque(maxlen=1)
        self._injected: set[str] = set()
        self._events: list[dict] = []
        self._pending_occlusion_reads = 0
        self._pending_stale_reads = 0

    def reset(self) -> WorldState:
        self.config.__post_init__()
        self._injected.clear()
        self._events.clear()
        self._observation_history.clear()
        self._pending_occlusion_reads = 0
        self._pending_stale_reads = 0

        object_xy = self.rng.uniform([0.18, 0.18], [0.42, 0.82]).astype(np.float32)
        target_xy = self.rng.uniform([0.63, 0.18], [0.88, 0.82]).astype(np.float32)
        ee_xy = np.asarray([0.1, 0.5], dtype=np.float32)
        if self.config.workspace_low != 0.05 or self.config.workspace_high != 0.95:
            # Preserve the default geometry while mapping custom workspaces to
            # the same relative layout.
            span = self.config.workspace_high - self.config.workspace_low
            def remap(xy: np.ndarray) -> np.ndarray:
                relative = (xy.astype(np.float64) - 0.05) / 0.9
                return self._clip(self.config.workspace_low + relative * span)

            object_xy, target_xy, ee_xy = map(remap, (object_xy, target_xy, ee_xy))
        self.state = WorldState(ee_xy=ee_xy, object_xy=object_xy, target_xy=target_xy)
        return self.observe()

    @property
    def done(self) -> bool:
        return self.state is not None and (
            self.state.success or self.state.step_index >= self.config.max_steps
        )

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def _emit(
        self, name: str, *, failure: bool = False, step: int | None = None, **payload: object
    ) -> dict:
        event = {
            "step": int(step if step is not None else self.state.step_index if self.state else 0),
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
                events.append(self._emit(
                    "object_displacement", failure=True, step=self.state.step_index + 1,
                    dx=float(delta[0]), dy=float(delta[1]),
                ))

        if self.perturbation == "transient_occlusion" and "transient_occlusion" not in self._injected:
            if action.kind == ActionKind.MOVE_TO_OBJECT:
                self.state.object_visible = False
                self._pending_occlusion_reads = 2
                self._injected.add("transient_occlusion")
                events.append(self._emit(
                    "transient_occlusion", failure=True, step=self.state.step_index + 1,
                ))

        if self.perturbation == "stale_observation" and "stale_observation" not in self._injected:
            if action.kind == ActionKind.GRASP:
                delta = self.rng.normal(0.0, 0.09, size=2).astype(np.float32)
                self.state.object_xy = self._clip(self.state.object_xy + delta)
                self._pending_stale_reads = 1
                self._injected.add("stale_observation")
                events.append(self._emit(
                    "stale_observation", failure=True, step=self.state.step_index + 1,
                    dx=float(delta[0]), dy=float(delta[1]),
                ))

        if self.perturbation in {"blocked_path", "compound_slip_block"} and "blocked_path" not in self._injected:
            if action.kind == ActionKind.MOVE_TO_TARGET and self.state.holding:
                self.state.path_blocked = True
                self._injected.add("blocked_path")
                events.append(self._emit(
                    "blocked_path", failure=True, step=self.state.step_index + 1,
                ))

        return events

    def _inject_after(self, action: Action, *, action_succeeded: bool) -> list[dict]:
        assert self.state is not None
        events: list[dict] = []

        if self.perturbation in {"grasp_slip", "compound_slip_block"} and "grasp_slip" not in self._injected:
            if action.kind == ActionKind.MOVE_TO_TARGET and self.state.holding and action_succeeded:
                self.state.holding = False
                slip = self.rng.normal(0.0, 0.08, size=2).astype(np.float32)
                self.state.object_xy = self._clip(self.state.ee_xy + slip)
                self._injected.add("grasp_slip")
                events.append(self._emit(
                    "grasp_slip", failure=True, dx=float(slip[0]), dy=float(slip[1]),
                ))

        return events

    def observe(self, *, fresh: bool = False) -> WorldState:
        if self.state is None:
            raise RuntimeError("reset must be called before observing")

        if fresh:
            self._pending_occlusion_reads = 0
            self._pending_stale_reads = 0
            self.state.object_visible = True

        if self._pending_stale_reads > 0 and self._observation_history:
            self._pending_stale_reads -= 1
            stale = self._observation_history[-1].copy()
            stale.step_index = self.state.step_index
            return stale

        obs = self.state.copy()
        if self._pending_occlusion_reads > 0:
            self._pending_occlusion_reads -= 1
            obs.object_visible = False
            if self._pending_occlusion_reads == 0:
                self.state.object_visible = True

        self._observation_history.append(obs.copy())
        return obs

    def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise RuntimeError("reset must be called before stepping")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        if self.done:
            return StepResult(self.state.copy(), self.state.success, [])

        events = self._inject_before(action)
        result = nominal_transition(self.state, action, self.config)
        self.state = result.state
        if action.kind == ActionKind.REOBSERVE:
            obs = self.observe(fresh=True)
            events.append(self._emit("reobserve"))
            return StepResult(obs, True, events)
        if action.kind == ActionKind.CLEAR_PATH:
            events.append(self._emit("path_cleared"))
        events.extend(self._inject_after(action, action_succeeded=result.action_succeeded))
        return StepResult(self.observe(), result.action_succeeded, events)

    def nominal_plan(self, state: WorldState | None = None) -> list[Action]:
        return nominal_actions(state if state is not None else self.observe(), self.config)

    def rollout(self, actions: Iterable[Action]) -> WorldState:
        if self.state is None:
            raise RuntimeError("reset must be called before a rollout")
        for action in actions:
            if self.done:
                break
            self.step(action)
        return self.state.copy()
