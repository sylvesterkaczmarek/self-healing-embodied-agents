import numpy as np
import pytest

from self_healing_embodied_agents.agents import AgentConfig, OpenLoopAgent, ReactiveReplanAgent, SelfHealingAgent
from self_healing_embodied_agents.env import EnvConfig, PERTURBATIONS, TabletopManipulationEnv
from self_healing_embodied_agents.recovery import RecoveryMemory
from self_healing_embodied_agents.training import train_transition_model
from self_healing_embodied_agents.types import ActionKind
from self_healing_embodied_agents.world_model import SymbolicCounterfactualModel


@pytest.fixture(scope="session")
def bundle():
    bundle, _ = train_transition_model(seed=7, episodes=80, epochs=40)
    return bundle


def test_self_healing_recovers_from_slip(bundle) -> None:
    seed = 17
    baseline = OpenLoopAgent().run_episode(TabletopManipulationEnv(seed=seed, perturbation="grasp_slip"))
    healed = SelfHealingAgent(bundle).run_episode(TabletopManipulationEnv(seed=seed, perturbation="grasp_slip"))
    assert not baseline.success
    assert healed.success
    assert healed.recovery_attempts >= 1


class NominalBundle:
    """Exact nominal predictions isolate agent control-flow tests from training."""

    residual_threshold = 0.01

    def predict(self, state, action):
        return SymbolicCounterfactualModel().transition(state, action).vector()


class DivergingBundle:
    residual_threshold = 0.0

    def predict(self, state, action):
        return np.zeros(9, dtype=np.float32)


@pytest.mark.parametrize("perturbation", ["none", "grasp_slip", "compound_slip_block"])
def test_false_alarms_on_completed_moves_do_not_restart_them_forever(perturbation):
    class MovementAlertsBundle(NominalBundle):
        def predict(self, state, action):
            prediction = super().predict(state, action)
            if action.kind in {ActionKind.MOVE_TO_OBJECT, ActionKind.MOVE_TO_TARGET}:
                prediction[0] += 1.0
            return prediction

    result = SelfHealingAgent(MovementAlertsBundle(), config=AgentConfig(max_steps=12)).run_episode(
        TabletopManipulationEnv(seed=17, perturbation=perturbation)
    )
    assert result.success
    assert result.steps < 12
    assert result.detections > 0
    assert any(e["event"] == "transition" and e["action"] == "grasp" for e in result.event_log)
    assert len([e for e in result.event_log if e["event"] == "recovery_outcome"]) == result.recovery_attempts


def test_every_interrupted_recovery_has_a_memory_outcome():
    memory = RecoveryMemory()
    result = SelfHealingAgent(DivergingBundle(), memory=memory, config=AgentConfig(max_steps=4)).run_episode(
        TabletopManipulationEnv(seed=1)
    )
    assert result.success
    assert result.recovery_attempts == 3
    assert result.recovery_successes == 1
    assert sum(memory.attempts.values()) == 3
    assert sum(memory.successes.values()) == 1
    outcomes = [e for e in result.event_log if e["event"] == "recovery_outcome"]
    assert len(outcomes) == result.recovery_attempts
    assert [e["success"] for e in outcomes] == [False, False, True]
    assert [e["reason"] for e in outcomes] == ["interrupted", "interrupted", "completed"]


@pytest.mark.parametrize("perturbation", PERTURBATIONS)
def test_nominal_oracle_control_flow_recovers_and_accounts_for_each_candidate(perturbation):
    memory = RecoveryMemory()
    result = SelfHealingAgent(NominalBundle(), memory=memory).run_episode(
        TabletopManipulationEnv(seed=17, perturbation=perturbation)
    )
    assert result.success
    assert sum(memory.attempts.values()) == result.recovery_attempts
    assert sum(memory.successes.values()) == result.recovery_successes
    selected = [e for e in result.event_log if e["event"] == "recovery_selected"]
    outcomes = [e for e in result.event_log if e["event"] == "recovery_outcome"]
    assert len(selected) == len(outcomes) == result.recovery_attempts
    assert result.true_positive_detections <= result.true_failures
    assert result.detections == result.true_positive_detections + result.false_positive_detections


@pytest.mark.parametrize("agent", [OpenLoopAgent(), ReactiveReplanAgent(), SelfHealingAgent(NominalBundle())])
def test_agent_respects_environment_budget(agent):
    env = TabletopManipulationEnv(seed=1, config=EnvConfig(max_steps=2))
    result = agent.run_episode(env)
    assert result.steps == env.state.step_index == 2
    assert not result.success


def test_no_unexecutable_recovery_is_registered_at_budget_exhaustion():
    memory = RecoveryMemory()
    result = SelfHealingAgent(DivergingBundle(), memory=memory, config=AgentConfig(max_steps=1)).run_episode(
        TabletopManipulationEnv(seed=1)
    )
    assert result.detections == 1
    assert result.recovery_attempts == 0
    assert not memory.attempts


@pytest.mark.parametrize("bad_prediction", [np.full(9, np.nan), np.full(9, np.inf), np.ones((1, 9))])
def test_invalid_prediction_stops_before_physical_action(bad_prediction):
    class InvalidBundle(NominalBundle):
        def predict(self, state, action):
            return bad_prediction

    env = TabletopManipulationEnv(seed=1)
    with pytest.raises((ValueError, FloatingPointError)):
        SelfHealingAgent(InvalidBundle()).run_episode(env)
    assert env.state.step_index == 0


def test_prediction_exception_records_pending_recovery_failure():
    class FailingBundle(DivergingBundle):
        def predict(self, state, action):
            if state.step_index:
                raise RuntimeError("model unavailable")
            return super().predict(state, action)

    memory = RecoveryMemory()
    env = TabletopManipulationEnv(seed=1)
    with pytest.raises(RuntimeError, match="model unavailable"):
        SelfHealingAgent(FailingBundle(), memory=memory).run_episode(env)
    assert env.state.step_index == 1
    assert sum(memory.attempts.values()) == 1
    assert sum(memory.successes.values()) == 0


def test_wider_float_prediction_cannot_overflow_before_physical_execution():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        return  # This runtime has no wider floating-point NumPy type.

    class WideBundle(NominalBundle):
        def predict(self, state, action):
            return np.full(9, np.longdouble("1e400"))

    env = TabletopManipulationEnv(seed=1)
    with pytest.raises(FloatingPointError, match="numerical precision"):
        SelfHealingAgent(WideBundle()).run_episode(env)
    assert env.state.step_index == 0


def test_overlapping_detection_windows_match_oldest_eligible_fault_first():
    class EvaluationEventsEnv(TabletopManipulationEnv):
        def step(self, action):
            result = super().step(action)
            if self.state.step_index in (1, 2):
                result.events.append(self._emit("evaluation_fault", failure=True))
            return result

    class LateAlertsBundle(NominalBundle):
        def predict(self, state, action):
            predicted = super().predict(state, action)
            if state.step_index + 1 in (2, 3):
                predicted[0] += 1
            return predicted

    result = SelfHealingAgent(LateAlertsBundle()).run_episode(EvaluationEventsEnv(seed=1))
    detections = [e for e in result.event_log if e["event"] == "detection"]
    assert [e["matched_failure_step"] for e in detections] == [1, 2]
    assert result.true_positive_detections == result.true_failures == 2


@pytest.mark.parametrize("settings", [{"max_steps": 0}, {"max_steps": True}, {"detection_window": -1}])
def test_invalid_agent_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        AgentConfig(**settings)
