from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import EnvConfig, nominal_actions
from .types import Action, ActionKind, WorldState
from .world_model import SymbolicCounterfactualModel


@dataclass
class RecoveryMemory:
    successes: dict[tuple[str, str], int] = field(default_factory=dict)
    attempts: dict[tuple[str, str], int] = field(default_factory=dict)

    def _counts(self, failure: str, name: str) -> tuple[int, int]:
        if not isinstance(failure, str) or not failure.strip():
            raise ValueError("failure must be a nonempty string")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("recovery name must be a nonempty string")
        key = (failure, name)
        s = self.successes.get(key, 0)
        n = self.attempts.get(key, 0)
        if type(s) is not int or type(n) is not int or not 0 <= s <= n:
            raise ValueError("memory counts must be integers with 0 <= successes <= attempts")
        return s, n

    def score(self, failure: str, name: str) -> float:
        s, n = self._counts(failure, name)
        return (s + 1) / (n + 2)

    def update(self, failure: str, name: str, success: bool) -> None:
        if type(success) is not bool:
            raise ValueError("success must be a boolean")
        s, n = self._counts(failure, name)
        key = (failure, name)
        self.attempts[key] = n + 1
        if success:
            self.successes[key] = s + 1


@dataclass
class TaskOutcomeMemory:
    """Task completion credit discounted by actions since candidate selection.

    Every selected candidate receives terminal credit, including candidates
    interrupted by another recovery. This is outcome credit, not a causal
    attribution of the eventual task result to any single candidate.
    """

    reward_sums: dict[tuple[str, str], float] = field(default_factory=dict)
    attempts: dict[tuple[str, str], int] = field(default_factory=dict)

    def _counts(self, failure: str, name: str) -> tuple[float, int]:
        if not isinstance(failure, str) or not failure.strip():
            raise ValueError("failure must be a nonempty string")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("recovery name must be a nonempty string")
        key = (failure, name)
        reward, count = self.reward_sums.get(key, 0.0), self.attempts.get(key, 0)
        if type(count) is not int or not np.isfinite(reward) or not 0 <= reward <= count:
            raise ValueError("memory requires 0 <= reward sum <= integer attempts")
        return reward, count

    def score(self, failure: str, name: str) -> float:
        reward, count = self._counts(failure, name)
        return (reward + 1) / (count + 2)

    def update(self, failure: str, name: str, success: bool, remaining_actions: int) -> None:
        if type(success) is not bool:
            raise ValueError("success must be a boolean")
        if type(remaining_actions) is not int or remaining_actions < 0:
            raise ValueError("remaining_actions must be a nonnegative integer")
        reward, count = self._counts(failure, name)
        key = (failure, name)
        self.reward_sums[key] = reward + float(success) / (1 + remaining_actions)
        self.attempts[key] = count + 1


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


def candidate_recoveries(
    failure: str, state: WorldState, *, config: EnvConfig | None = None
) -> list[RecoveryCandidate]:
    retry_from_state = RecoveryCandidate(
        "replan_from_observation",
        tuple(nominal_actions(state, config)),
    )
    candidates = [retry_from_state]

    if failure == "perception_loss":
        candidates.append(
            RecoveryCandidate(
                "refresh_then_replan",
                (Action(ActionKind.REOBSERVE), *tuple(nominal_actions(_visible_copy(state), config))),
            )
        )
    elif failure == "path_obstruction":
        cleared = state.copy()
        cleared.path_blocked = False
        candidates.append(
            RecoveryCandidate(
                "clear_then_resume",
                (Action(ActionKind.CLEAR_PATH), *tuple(nominal_actions(cleared, config))),
            )
        )
    elif failure in {"grasp_loss", "object_state_shift", "execution_failure", "state_divergence"}:
        candidates.append(
            RecoveryCandidate(
                "reobserve_reacquire_resume",
                (
                    Action(ActionKind.REOBSERVE),
                    *tuple(nominal_actions(_visible_copy(state), config)),
                ),
            )
        )

    return _dedupe(candidates)


def _visible_copy(state: WorldState) -> WorldState:
    s = state.copy()
    s.object_visible = True
    return s


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
    memory: RecoveryMemory | TaskOutcomeMemory | None = None,
    config: EnvConfig | None = None,
    remaining_steps: int | None = None,
) -> RecoveryCandidate:
    """Rank the executable prefix of each plan within the available action budget.

    A partial plan can still be useful: the agent replans from its observation
    after that plan completes. Ties retain the supplied candidate order.
    """
    if not candidates:
        raise ValueError("at least one recovery candidate is required")
    model = SymbolicCounterfactualModel(config)
    available = max(0, model.config.max_steps - state.step_index)
    if remaining_steps is not None:
        if type(remaining_steps) is not int or remaining_steps < 0:
            raise ValueError("remaining_steps must be a nonnegative integer")
        available = min(available, remaining_steps)
    best: tuple[float, RecoveryCandidate] | None = None

    for candidate in candidates:
        if not isinstance(candidate, RecoveryCandidate):
            raise ValueError("candidates must contain RecoveryCandidate values")
        if not isinstance(candidate.name, str) or not candidate.name.strip():
            raise ValueError("candidate names must be nonempty strings")
        if not candidate.actions and not state.success:
            raise ValueError("a recovery candidate must contain an action for an unfinished task")
        if any(not isinstance(action, Action) for action in candidate.actions):
            raise ValueError("candidate actions must contain Action values")
        predicted = model.rollout(state, list(candidate.actions[:available]))
        goal_distance = float(np.linalg.norm(predicted.object_xy - predicted.target_xy))
        terminal_bonus = 3.0 if predicted.success else 0.0
        executed_steps = predicted.step_index - state.step_index
        efficiency_penalty = 0.07 * executed_steps
        memory_bonus = 0.0 if memory is None else 0.8 * (memory.score(failure, candidate.name) - 0.5)
        score = terminal_bonus - goal_distance - efficiency_penalty + memory_bonus
        if not np.isfinite(score):
            raise FloatingPointError("recovery candidate produced a non-finite score")
        if best is None or score > best[0]:
            best = (score, candidate)

    assert best is not None
    return best[1]
