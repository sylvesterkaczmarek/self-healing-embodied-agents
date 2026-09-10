from copy import deepcopy

import numpy as np
import pytest

from self_healing_embodied_agents.agents import (
    AgentConfig, AlwaysReplanAgent, ReactiveReplanAgent, SelfHealingAgent,
    postcondition_failed,
)
from self_healing_embodied_agents.env import EnvConfig, PERTURBATIONS, TabletopManipulationEnv
from self_healing_embodied_agents.recovery import RecoveryCandidate, RecoveryMemory, TaskOutcomeMemory
from self_healing_embodied_agents.types import Action, ActionKind
from self_healing_embodied_agents.world_model import SymbolicCounterfactualModel


class NominalBundle:
    residual_threshold = 0.01

    def predict(self, state, action):
        return SymbolicCounterfactualModel().transition(state, action).vector()


class DivergingBundle:
    residual_threshold = 0.0

    def predict(self, state, action):
        return np.zeros(9, dtype=np.float32)


def actions(result):
    return [event["action"] for event in result.event_log if event["event"] == "transition"]


@pytest.mark.parametrize("perturbation", PERTURBATIONS)
def test_failure_nominal_preserves_reactive_execution(perturbation):
    reactive = ReactiveReplanAgent().run_episode(TabletopManipulationEnv(seed=17, perturbation=perturbation))
    failure = SelfHealingAgent(detector="failure", recovery="nominal").run_episode(
        TabletopManipulationEnv(seed=17, perturbation=perturbation)
    )
    assert actions(failure) == actions(reactive)
    assert failure.success == reactive.success
    assert failure.recovery_attempts == reactive.recovery_attempts
    assert failure.recovery_successes == reactive.recovery_successes
    assert failure.agent == "failure_nominal"
    assert all(event["trigger"] == "execution_failure" for event in failure.event_log if event["event"] == "detection")


@pytest.mark.parametrize("detector", ["failure", "postcondition", "analytical"])
def test_nonlearned_controls_never_call_bundle(detector):
    class UnavailableBundle:
        def predict(self, state, action):
            raise AssertionError("nonlearned control called a learned predictor")

    result = SelfHealingAgent(UnavailableBundle(), detector=detector, recovery="nominal").run_episode(
        TabletopManipulationEnv(seed=17)
    )
    assert result.success
    assert result.steps == 4
    assert result.detections == 0


@pytest.mark.parametrize("detector,trigger", [
    ("learned", "residual_only"), ("postcondition", "postcondition_only"), ("analytical", "analytical_only"),
])
def test_monitors_catch_slip_before_explicit_place_failure(detector, trigger):
    result = SelfHealingAgent(NominalBundle(), detector=detector, recovery="nominal").run_episode(
        TabletopManipulationEnv(seed=17, perturbation="grasp_slip")
    )
    alarms = [event for event in result.event_log if event["event"] == "detection"]
    assert result.success
    assert alarms[0]["step"] == 3
    assert alarms[0]["trigger"] == trigger
    assert not alarms[0]["execution_failure"]
    assert alarms[0]["matched_failure_step"] == 3
    failure = SelfHealingAgent(detector="failure", recovery="nominal").run_episode(
        TabletopManipulationEnv(seed=17, perturbation="grasp_slip")
    )
    assert result.steps < failure.steps


@pytest.mark.parametrize("perturbation", PERTURBATIONS)
def test_always_replan_records_actions_and_planning_without_alarms(perturbation):
    result = AlwaysReplanAgent().run_episode(TabletopManipulationEnv(seed=17, perturbation=perturbation))
    assert result.success
    assert len(actions(result)) == result.steps
    assert sum(event["event"] == "planning" for event in result.event_log) == result.steps
    assert result.detections == result.interventions == result.recovery_attempts == 0


@pytest.mark.parametrize("agent", [
    AlwaysReplanAgent(), SelfHealingAgent(detector="postcondition", recovery="nominal"),
])
def test_comparison_controls_respect_actual_budget(agent):
    result = agent.run_episode(TabletopManipulationEnv(seed=17, config=EnvConfig(max_steps=2)))
    assert result.steps == 2
    assert not result.success


def test_postcondition_tolerates_rounding_but_catches_observed_endpoint_shift():
    env = TabletopManipulationEnv(seed=17)
    previous = env.reset()
    action = Action(ActionKind.MOVE_TO_OBJECT)
    observed = env.step(action).state
    observed.ee_xy[0] = np.nextafter(observed.ee_xy[0], np.float32(np.inf))
    assert not postcondition_failed(previous, observed, action)
    observed.object_xy[0] += 0.01
    assert postcondition_failed(previous, observed, action)


@pytest.mark.parametrize("kind,flag", [(ActionKind.REOBSERVE, "object_visible"), (ActionKind.CLEAR_PATH, "path_blocked")])
def test_postcondition_checks_observable_repair_result(kind, flag):
    observed = TabletopManipulationEnv(seed=17).reset()
    previous = observed.copy()
    setattr(observed, flag, kind == ActionKind.CLEAR_PATH)
    assert postcondition_failed(previous, observed, Action(kind))
    setattr(observed, flag, kind == ActionKind.REOBSERVE)
    assert not postcondition_failed(previous, observed, Action(kind))


@pytest.mark.parametrize("detector", ["failure", "learned", "postcondition", "analytical"])
def test_evaluator_labels_cannot_change_control_actions(detector):
    class RelabelledEnv(TabletopManipulationEnv):
        @property
        def events(self):
            return [{"event": "unrelated_label", "failure": True, "step": step}
                    for step in range(1, self.state.step_index + 1)]

    agent = SelfHealingAgent(NominalBundle(), detector=detector, recovery="nominal")
    normal = agent.run_episode(TabletopManipulationEnv(seed=17, perturbation="compound_slip_block"))
    relabelled = agent.run_episode(RelabelledEnv(seed=17, perturbation="compound_slip_block"))
    assert actions(normal) == actions(relabelled)
    assert normal.success == relabelled.success
    assert normal.true_failures != relabelled.true_failures


def test_task_credit_waits_until_terminal_outcome_and_includes_interruptions():
    memory = TaskOutcomeMemory()

    class InspectingEnv(TabletopManipulationEnv):
        def step(self, action):
            assert not memory.attempts, "terminal credit was applied during the episode"
            return super().step(action)

    result = SelfHealingAgent(DivergingBundle(), memory=memory, config=AgentConfig(max_steps=4)).run_episode(
        InspectingEnv(seed=1)
    )
    assert result.success
    assert result.agent == "self_healing_task_memory"
    assert result.recovery_attempts == sum(memory.attempts.values()) == 3
    assert result.recovery_successes == 1
    updates = [event for event in result.event_log if event["event"] == "memory_update"]
    assert [event["reward"] for event in updates] == pytest.approx([1 / 4, 1 / 3, 1 / 2])
    assert all(event["step"] == 4 for event in updates)
    assert sum(memory.reward_sums.values()) == pytest.approx(sum(event["reward"] for event in updates))


def test_completed_candidate_gets_no_task_credit_when_task_budget_expires(monkeypatch):
    monkeypatch.setattr(
        "self_healing_embodied_agents.agents.choose_recovery",
        lambda *args, **kwargs: RecoveryCandidate("observe_only", (Action(ActionKind.REOBSERVE),)),
    )
    memory = TaskOutcomeMemory()
    result = SelfHealingAgent(DivergingBundle(), memory=memory, config=AgentConfig(max_steps=2)).run_episode(
        TabletopManipulationEnv(seed=1)
    )
    assert not result.success
    assert result.recovery_successes == 1
    assert sum(memory.attempts.values()) == 1
    assert sum(memory.reward_sums.values()) == 0


def test_budget_exhausted_candidates_receive_zero_task_credit():
    memory = TaskOutcomeMemory()
    result = SelfHealingAgent(DivergingBundle(), memory=memory, config=AgentConfig(max_steps=3)).run_episode(
        TabletopManipulationEnv(seed=1)
    )
    assert not result.success
    assert sum(memory.attempts.values()) == result.recovery_attempts == 2
    assert sum(memory.reward_sums.values()) == 0


@pytest.mark.parametrize("memory", [RecoveryMemory(), TaskOutcomeMemory()])
def test_frozen_memory_evaluations_leave_history_unchanged(memory):
    training = SelfHealingAgent(DivergingBundle(), memory=memory)
    training.run_episode(TabletopManipulationEnv(seed=1))
    snapshot = deepcopy(memory)
    held_out = SelfHealingAgent(DivergingBundle(), memory=memory, update_memory=False)
    first = held_out.run_episode(TabletopManipulationEnv(seed=17, perturbation="grasp_slip"))
    second = held_out.run_episode(TabletopManipulationEnv(seed=17, perturbation="grasp_slip"))
    assert memory == snapshot
    assert actions(first) == actions(second)
    assert first.recovery_attempts > 0


def test_exception_records_zero_terminal_credit_for_selected_candidate():
    class FailingBundle(DivergingBundle):
        def predict(self, state, action):
            if state.step_index:
                raise RuntimeError("predictor unavailable")
            return super().predict(state, action)

    memory = TaskOutcomeMemory()
    with pytest.raises(RuntimeError, match="predictor unavailable"):
        SelfHealingAgent(FailingBundle(), memory=memory).run_episode(TabletopManipulationEnv(seed=1))
    assert sum(memory.attempts.values()) == 1
    assert sum(memory.reward_sums.values()) == 0


@pytest.mark.parametrize("kwargs", [
    {}, {"detector": "missing"}, {"detector": "failure", "recovery": "missing"},
    {"detector": "failure", "update_memory": "yes"},
    {"detector": "failure", "recovery": "nominal", "memory": RecoveryMemory()},
])
def test_invalid_comparison_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SelfHealingAgent(**kwargs)


@pytest.mark.parametrize("remaining", [-1, 1.5, True])
def test_task_memory_rejects_invalid_action_counts_without_partial_update(remaining):
    memory = TaskOutcomeMemory()
    with pytest.raises(ValueError):
        memory.update("grasp_loss", "retry", True, remaining)
    assert not memory.attempts
    assert not memory.reward_sums
