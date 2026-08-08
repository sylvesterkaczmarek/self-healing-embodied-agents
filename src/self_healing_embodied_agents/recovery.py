from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Action, ActionKind, WorldState
from .world_model import SymbolicCounterfactualModel


@dataclass
class RecoveryMemory:
    successes: dict[tuple[str, str], int] = field(default_factory=dict)
    attempts: dict[tuple[str, str], int] = field(default_factory=dict)

    def score(self, failure: str, name: str) -> float:
        key = (failure, name)
        s = self.successes.get(key, 0)
        n = self.attempts.get(key, 0)
        return (s + 1.0) / (n + 2.0)

    def update(self, failure: str, name: str, success: bool) -> None:
        key = (failure, name)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if success:
            self.successes[key] = self.successes.get(key, 0) + 1


@dataclass(frozen=True)
class RecoveryCandidate:
    name: str
    actions: tuple[Action, ...]


def diagnose(previous: WorldState, observed: WorldState, action: Action, action_succeeded: bool) -> str:
    if not observed.object_visible:
        return "perception_loss"
    if observed.path_blocked:
        return "path_obstruction"
    if action.kind == ActionKind.MOVE_TO_TARGET and previous.holding and not observed.holding:
        return "grasp_loss"
    if action.kind in {ActionKind.MOVE_TO_OBJECT, ActionKind.GRASP}:
        if np.linalg.norm(observed.object_xy - previous.object_xy) > 0.07:
            return "object_state_shift"
    if not action_succeeded:
        return "execution_failure"
    return "state_divergence"


def candidate_recoveries(failure: str, state: WorldState) -> list[RecoveryCandidate]:
    retry_from_state = RecoveryCandidate(
        "replan_from_observation",
        tuple(_nominal_actions(state)),
    )
    candidates = [retry_from_state]

    if failure == "perception_loss":
        candidates.append(
            RecoveryCandidate(
                "refresh_then_replan",
                (Action(ActionKind.REOBSERVE), *tuple(_nominal_actions(_visible_copy(state)))),
            )
        )
    elif failure == "path_obstruction":
        candidates.append(
            RecoveryCandidate(
                "clear_then_resume",
                (Action(ActionKind.CLEAR_PATH), Action(ActionKind.MOVE_TO_TARGET), Action(ActionKind.PLACE)),
            )
        )
    elif failure in {"grasp_loss", "object_state_shift", "execution_failure", "state_divergence"}:
        candidates.append(
            RecoveryCandidate(
                "reobserve_reacquire_resume",
                (
                    Action(ActionKind.REOBSERVE),
                    Action(ActionKind.MOVE_TO_OBJECT),
                    Action(ActionKind.GRASP),
                    Action(ActionKind.MOVE_TO_TARGET),
                    Action(ActionKind.PLACE),
                ),
            )
        )

    return _dedupe(candidates)


def _visible_copy(state: WorldState) -> WorldState:
    s = state.copy()
    s.object_visible = True
    return s


def _nominal_actions(state: WorldState) -> list[Action]:
    if state.success:
        return []
    if not state.object_visible:
        return [Action(ActionKind.REOBSERVE)]
    if state.path_blocked:
        return [Action(ActionKind.CLEAR_PATH)]
    if state.holding:
        return [Action(ActionKind.MOVE_TO_TARGET), Action(ActionKind.PLACE)]
    return [
        Action(ActionKind.MOVE_TO_OBJECT),
        Action(ActionKind.GRASP),
        Action(ActionKind.MOVE_TO_TARGET),
        Action(ActionKind.PLACE),
    ]


def _dedupe(candidates: list[RecoveryCandidate]) -> list[RecoveryCandidate]:
    seen: set[tuple[str, ...]] = set()
    out: list[RecoveryCandidate] = []
    for candidate in candidates:
        signature = tuple(a.kind.value for a in candidate.actions)
        if signature not in seen:
            seen.add(signature)
            out.append(candidate)
    return out


def choose_recovery(
    failure: str,
    state: WorldState,
    candidates: list[RecoveryCandidate],
    *,
    memory: RecoveryMemory | None = None,
) -> RecoveryCandidate:
    model = SymbolicCounterfactualModel()
    best: tuple[float, RecoveryCandidate] | None = None

    for candidate in candidates:
        predicted = model.rollout(state, list(candidate.actions))
        goal_distance = float(np.linalg.norm(predicted.object_xy - predicted.target_xy))
        terminal_bonus = 3.0 if predicted.success else 0.0
        efficiency_penalty = 0.07 * len(candidate.actions)
        memory_bonus = 0.0 if memory is None else 0.8 * (memory.score(failure, candidate.name) - 0.5)
        score = terminal_bonus - goal_distance - efficiency_penalty + memory_bonus
        if best is None or score > best[0]:
            best = (score, candidate)

    assert best is not None
    return best[1]
