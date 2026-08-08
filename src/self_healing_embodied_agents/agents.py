from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .env import TabletopManipulationEnv
from .recovery import RecoveryMemory, candidate_recoveries, choose_recovery, diagnose
from .types import Action, ActionKind, EpisodeResult, WorldState
from .world_model import ModelBundle


def residual(predicted: np.ndarray, observed: WorldState) -> float:
    return float(np.sqrt(np.mean((predicted - observed.vector()) ** 2)))


@dataclass
class AgentConfig:
    max_steps: int = 24
    detection_window: int = 1


class BaseAgent:
    name = "base"

    def __init__(self, *, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def run_episode(self, env: TabletopManipulationEnv) -> EpisodeResult:
        raise NotImplementedError


class OpenLoopAgent(BaseAgent):
    name = "open_loop"

    def run_episode(self, env: TabletopManipulationEnv) -> EpisodeResult:
        state = env.reset()
        plan = deque(env.nominal_plan(state))
        steps = 0
        while plan and steps < self.config.max_steps and not state.success:
            action = plan.popleft()
            result = env.step(action)
            state = result.state
            steps += 1
        failures = len({int(e["step"]) for e in env.events if e.get("failure")})
        return EpisodeResult(
            success=state.success,
            steps=steps,
            perturbation=env.perturbation,
            agent=self.name,
            seed=env.seed,
            true_failures=failures,
            event_log=env.events,
        )


class ReactiveReplanAgent(BaseAgent):
    name = "reactive_replan"

    def run_episode(self, env: TabletopManipulationEnv) -> EpisodeResult:
        state = env.reset()
        plan = deque(env.nominal_plan(state))
        steps = 0
        interventions = 0
        recovery_attempts = 0

        while steps < self.config.max_steps and not state.success:
            if not plan:
                plan = deque(env.nominal_plan(state))
                if not plan:
                    break
            action = plan.popleft()
            result = env.step(action)
            steps += 1
            state = result.state
            if not result.action_succeeded:
                interventions += 1
                recovery_attempts += 1
                plan = deque(env.nominal_plan(state))

        failures = len({int(e["step"]) for e in env.events if e.get("failure")})
        return EpisodeResult(
            success=state.success,
            steps=steps,
            perturbation=env.perturbation,
            agent=self.name,
            seed=env.seed,
            interventions=interventions,
            true_failures=failures,
            recovery_attempts=recovery_attempts,
            recovery_successes=int(state.success and recovery_attempts > 0),
            event_log=env.events,
        )


class SelfHealingAgent(BaseAgent):
    name = "self_healing"

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        memory: RecoveryMemory | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self.bundle = bundle
        self.memory = memory

    def run_episode(self, env: TabletopManipulationEnv) -> EpisodeResult:
        state = env.reset()
        plan = deque(env.nominal_plan(state))
        steps = 0
        interventions = 0
        detections = 0
        tp = 0
        fp = 0
        recovery_attempts = 0
        recovery_successes = 0
        recovery_context: tuple[str, str] | None = None
        matched_failure_steps: set[int] = set()

        while steps < self.config.max_steps and not state.success:
            if not plan:
                plan = deque(env.nominal_plan(state))
                if not plan:
                    break

            action = plan.popleft()
            previous = state.copy()
            predicted = self.bundle.predict(previous, action)
            result = env.step(action)
            steps += 1
            state = result.state
            score = residual(predicted, state)

            recovery_action = action.kind in {ActionKind.REOBSERVE, ActionKind.CLEAR_PATH}
            diverged = (not result.action_succeeded) or (
                not recovery_action and score > self.bundle.residual_threshold
            )
            if diverged:
                detections += 1
                interventions += 1
                eligible_failure_steps = sorted(
                    {
                        int(e["step"])
                        for e in env.events
                        if e.get("failure")
                        and 0 <= state.step_index - int(e["step"]) <= self.config.detection_window
                        and int(e["step"]) not in matched_failure_steps
                    },
                    reverse=True,
                )
                if eligible_failure_steps:
                    matched_failure_steps.add(eligible_failure_steps[0])
                    tp += 1
                else:
                    fp += 1

                failure = diagnose(previous, state, action, result.action_succeeded)
                candidates = candidate_recoveries(failure, state)
                selected = choose_recovery(failure, state, candidates, memory=self.memory)
                plan = deque(selected.actions)
                recovery_attempts += 1
                recovery_context = (failure, selected.name)

            if state.success and recovery_context is not None:
                recovery_successes += 1
                if self.memory is not None:
                    self.memory.update(*recovery_context, success=True)
                recovery_context = None

        if recovery_context is not None and self.memory is not None:
            self.memory.update(*recovery_context, success=False)

        failures = len({int(e["step"]) for e in env.events if e.get("failure")})
        return EpisodeResult(
            success=state.success,
            steps=steps,
            perturbation=env.perturbation,
            agent=self.name if self.memory is None else "self_healing_memory",
            seed=env.seed,
            interventions=interventions,
            true_failures=failures,
            detections=detections,
            true_positive_detections=tp,
            false_positive_detections=fp,
            recovery_attempts=recovery_attempts,
            recovery_successes=recovery_successes,
            event_log=env.events,
        )
