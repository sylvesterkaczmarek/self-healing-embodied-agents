from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .env import TabletopManipulationEnv
from .recovery import RecoveryMemory, candidate_recoveries, choose_recovery, diagnose
from .types import Action, ActionKind, EpisodeResult, WorldState
from .world_model import ModelBundle


def residual(predicted: np.ndarray, observed: WorldState) -> float:
    predicted = _prediction(predicted)
    error = predicted - _prediction(observed.vector())
    scale = float(np.max(np.abs(error)))
    return float(scale * np.sqrt(np.mean((error / scale) ** 2))) if scale else 0.0


def _prediction(predicted: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted)
    if predicted.shape != (9,) or predicted.dtype.kind not in "fiu":
        raise ValueError("predicted state must be a real vector of length nine")
    if not np.isfinite(predicted).all():
        raise FloatingPointError("nonfinite model predictions cannot guide physical execution")
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = predicted.astype(np.float64)
    if not np.isfinite(predicted).all():
        raise FloatingPointError("predicted state exceeds supported numerical precision")
    return predicted


def _record_transition(trace: list[dict], action: Action, result) -> None:
    trace.extend(result.events)
    trace.append({
        "event": "transition", "step": result.state.step_index,
        "action": action.kind.value, "action_succeeded": result.action_succeeded,
        "observed_state": result.state.vector().tolist(), "task_success": result.state.success,
    })


@dataclass
class AgentConfig:
    max_steps: int = 24
    detection_window: int = 1

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(self.detection_window) is not int or self.detection_window < 0:
            raise ValueError("detection_window must be a nonnegative integer")


class BaseAgent:
    name = "base"

    def __init__(self, *, config: AgentConfig | None = None) -> None:
        self.config = AgentConfig() if config is None else config
        if not isinstance(self.config, AgentConfig):
            raise TypeError("config must be an AgentConfig")

    def run_episode(self, env: TabletopManipulationEnv) -> EpisodeResult:
        raise NotImplementedError


class OpenLoopAgent(BaseAgent):
    name = "open_loop"

    def run_episode(self, env: TabletopManipulationEnv) -> EpisodeResult:
        state = env.reset()
        plan = deque(env.nominal_plan(state))
        steps = 0
        trace: list[dict] = []
        limit = min(self.config.max_steps, env.config.max_steps)
        while plan and steps < limit and not env.done:
            action = plan.popleft()
            result = env.step(action)
            _record_transition(trace, action, result)
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
            event_log=trace,
        )


class ReactiveReplanAgent(BaseAgent):
    name = "reactive_replan"

    def run_episode(self, env: TabletopManipulationEnv) -> EpisodeResult:
        state = env.reset()
        plan = deque(env.nominal_plan(state))
        steps = 0
        interventions = 0
        recovery_attempts = 0
        recovery_successes = 0
        recovering = False
        trace: list[dict] = []
        limit = min(self.config.max_steps, env.config.max_steps)

        while steps < limit and not env.done:
            if not plan:
                plan = deque(env.nominal_plan(state))
                if not plan:
                    break
            action = plan.popleft()
            result = env.step(action)
            _record_transition(trace, action, result)
            steps += 1
            state = result.state
            if not result.action_succeeded:
                interventions += 1
                if recovering:
                    trace.append({"event": "recovery_outcome", "step": state.step_index,
                                  "success": False, "reason": "interrupted"})
                    recovering = False
                if steps < limit and not env.done:
                    recovery_attempts += 1
                    plan = deque(env.nominal_plan(state))
                    recovering = True
                    trace.append({"event": "recovery_selected", "step": state.step_index,
                                  "candidate": "reactive_replan", "actions": [a.kind.value for a in plan]})
            elif recovering and (not plan or state.success):
                recovery_successes += 1
                recovering = False
                trace.append({"event": "recovery_outcome", "step": state.step_index,
                              "success": True, "reason": "completed"})

        if recovering:
            trace.append({"event": "recovery_outcome", "step": state.step_index,
                          "success": False, "reason": "budget_exhausted"})

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
            recovery_successes=recovery_successes,
            event_log=trace,
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
        if not np.isfinite(bundle.residual_threshold) or bundle.residual_threshold < 0:
            raise ValueError("residual threshold must be finite and nonnegative")

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
        trace: list[dict] = []
        limit = min(self.config.max_steps, env.config.max_steps)

        def finish_recovery(success: bool, reason: str) -> None:
            nonlocal recovery_context, recovery_successes
            if recovery_context is None:
                return
            failure, candidate = recovery_context
            if self.memory is not None:
                self.memory.update(failure, candidate, success=success)
            recovery_successes += int(success)
            trace.append({"event": "recovery_outcome", "step": state.step_index,
                          "failure_class": failure, "candidate": candidate,
                          "success": success, "reason": reason})
            recovery_context = None

        try:
            while steps < limit and not env.done:
                if not plan:
                    plan = deque(env.nominal_plan(state))
                    if not plan:
                        break

                action = plan.popleft()
                previous = state.copy()
                predicted = _prediction(self.bundle.predict(previous, action))
                result = env.step(action)
                _record_transition(trace, action, result)
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
                    )
                    if eligible_failure_steps:
                        matched_failure_steps.add(eligible_failure_steps[0])
                        tp += 1
                    else:
                        fp += 1

                    trace.append({"event": "detection", "step": state.step_index,
                                  "residual": score, "threshold": float(self.bundle.residual_threshold),
                                  "execution_failure": not result.action_succeeded,
                                  "matched_failure_step": eligible_failure_steps[0] if eligible_failure_steps else None})
                    finish_recovery(state.success, "completed" if state.success else "interrupted")
                    if state.success or steps >= limit or env.done:
                        continue
                    failure = diagnose(previous, state, action, result.action_succeeded)
                    candidates = candidate_recoveries(failure, state, config=env.config)
                    selected = choose_recovery(
                        failure, state, candidates, memory=self.memory,
                        config=env.config, remaining_steps=limit - steps,
                    )
                    plan = deque(selected.actions)
                    recovery_attempts += 1
                    recovery_context = (failure, selected.name)
                    trace.append({"event": "recovery_selected", "step": state.step_index,
                                  "failure_class": failure, "candidate": selected.name,
                                  "actions": [a.kind.value for a in selected.actions],
                                  "remaining_steps": limit - steps})

                elif recovery_context is not None and (not plan or state.success):
                    finish_recovery(True, "completed")
        except Exception:
            finish_recovery(False, "execution_error")
            raise

        finish_recovery(False, "budget_exhausted")

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
            event_log=trace,
        )
