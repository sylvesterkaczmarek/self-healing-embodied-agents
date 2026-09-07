import numpy as np
import pytest

from self_healing_embodied_agents.env import EnvConfig, TabletopManipulationEnv, nominal_actions
from self_healing_embodied_agents.recovery import (
    RecoveryCandidate,
    RecoveryMemory,
    candidate_recoveries,
    choose_recovery,
)
from self_healing_embodied_agents.types import Action, ActionKind, WorldState
from self_healing_embodied_agents.world_model import SymbolicCounterfactualModel


def state_at(*, object_x=0.27, ee_x=0.2, target_x=0.8, **flags):
    return WorldState(
        ee_xy=np.array([ee_x, 0.5]),
        object_xy=np.array([object_x, 0.5]),
        target_xy=np.array([target_x, 0.5]),
        **flags,
    )


def actions(*kinds):
    return tuple(Action(kind) for kind in kinds)


@pytest.mark.parametrize(
    "radius,distance,expected",
    [(0.06, 0.07, False), (0.02, 0.04, False), (0.10, 0.09, True)],
)
def test_grasp_rollout_respects_simulator_radius(radius, distance, expected):
    config = EnvConfig(grasp_radius=radius)
    state = state_at(object_x=0.2 + distance)
    env = TabletopManipulationEnv(config=config)
    env.reset()
    env.state = state.copy()
    action = Action(ActionKind.GRASP)
    predicted = SymbolicCounterfactualModel(config).transition(state, action)
    observed = env.step(action).state
    assert predicted.holding is expected
    assert observed.holding is expected
    np.testing.assert_array_equal(predicted.vector(), observed.vector())
    assert not state.holding
    assert state.step_index == 0


@pytest.mark.parametrize(
    "radius,distance,expected",
    [(0.08, 0.09, False), (0.04, 0.06, False), (0.12, 0.09, True)],
)
def test_placement_rollout_respects_simulator_radius(radius, distance, expected):
    config = EnvConfig(placement_radius=radius)
    state = state_at(ee_x=0.8 - distance, object_x=0.8 - distance, holding=True)
    env = TabletopManipulationEnv(config=config)
    env.reset()
    env.state = state.copy()
    action = Action(ActionKind.PLACE)
    predicted = SymbolicCounterfactualModel(config).transition(state, action)
    observed = env.step(action).state
    assert predicted.success is expected
    assert observed.success is expected
    assert predicted.holding is False
    np.testing.assert_array_equal(predicted.vector(), observed.vector())


@pytest.mark.parametrize("kind", list(ActionKind))
@pytest.mark.parametrize("terminal", ["success", "budget"])
def test_terminal_rollouts_do_not_execute_more_actions(kind, terminal):
    config = EnvConfig(max_steps=3)
    state = state_at(
        ee_x=0.8, object_x=0.8, success=terminal == "success", step_index=3
    )
    env = TabletopManipulationEnv(config=config)
    env.reset()
    env.state = state.copy()
    predicted = SymbolicCounterfactualModel(config).transition(state, Action(kind))
    observed = env.step(Action(kind)).state
    np.testing.assert_array_equal(predicted.vector(), state.vector())
    np.testing.assert_array_equal(observed.vector(), state.vector())
    assert predicted.step_index == observed.step_index == 3
    assert predicted.success == observed.success == state.success
    assert predicted is not state


def test_rollout_stops_at_action_budget():
    state = state_at(ee_x=0.2, object_x=0.2, holding=True, step_index=1)
    predicted = SymbolicCounterfactualModel(EnvConfig(max_steps=2)).rollout(
        state, list(actions(ActionKind.MOVE_TO_TARGET, ActionKind.PLACE))
    )
    assert predicted.step_index == 2
    assert predicted.holding
    assert not predicted.success
    np.testing.assert_array_equal(predicted.object_xy, predicted.target_xy)


@pytest.mark.parametrize("radius,distance", [(0.06, 0.07), (0.02, 0.04)])
def test_recovery_avoids_shorter_plan_that_cannot_grasp(radius, distance):
    config = EnvConfig(grasp_radius=radius)
    state = state_at(object_x=0.2 + distance)
    candidates = [
        RecoveryCandidate(
            "skip_approach", actions(ActionKind.GRASP, ActionKind.MOVE_TO_TARGET, ActionKind.PLACE)
        ),
        RecoveryCandidate(
            "approach",
            actions(
                ActionKind.MOVE_TO_OBJECT, ActionKind.GRASP,
                ActionKind.MOVE_TO_TARGET, ActionKind.PLACE,
            ),
        ),
    ]
    selected = choose_recovery("execution_failure", state, candidates, config=config)
    assert selected.name == "approach"
    env = TabletopManipulationEnv(config=config)
    env.reset()
    env.state = state.copy()
    assert env.rollout(selected.actions).success


def test_candidate_scoring_does_not_reward_success_beyond_remaining_budget():
    state = state_at(ee_x=0.8, object_x=0.8, holding=True)
    delayed = RecoveryCandidate(
        "delayed_place", actions(ActionKind.CLEAR_PATH, ActionKind.CLEAR_PATH, ActionKind.PLACE)
    )
    short = RecoveryCandidate("refresh", actions(ActionKind.REOBSERVE))
    candidates = [delayed, short]
    assert choose_recovery("state_divergence", state, candidates).name == "delayed_place"
    assert choose_recovery("state_divergence", state, candidates, remaining_steps=2).name == "refresh"


def test_actions_after_success_do_not_increase_predicted_action_cost():
    state = state_at(ee_x=0.8, object_x=0.8, holding=True)
    candidates = [
        RecoveryCandidate("place_then_unused", actions(ActionKind.PLACE, ActionKind.GRASP)),
        RecoveryCandidate("place", actions(ActionKind.PLACE)),
    ]
    assert choose_recovery("state_divergence", state, candidates) is candidates[0]


@pytest.mark.parametrize("holding", [False, True])
def test_obstruction_candidate_resumes_from_actual_grasp_state(holding):
    state = state_at(ee_x=0.2, object_x=0.2 if holding else 0.27, holding=holding, path_blocked=True)
    candidate = next(c for c in candidate_recoveries("path_obstruction", state) if c.name == "clear_then_resume")
    assert (ActionKind.GRASP in [a.kind for a in candidate.actions]) is not holding
    env = TabletopManipulationEnv()
    env.reset()
    env.state = state.copy()
    assert env.rollout(candidate.actions).success


@pytest.mark.parametrize("remaining", [-1, 1.5, True])
def test_invalid_remaining_budget_is_rejected(remaining):
    with pytest.raises(ValueError, match="remaining_steps"):
        choose_recovery(
            "execution_failure", state_at(),
            [RecoveryCandidate("refresh", actions(ActionKind.REOBSERVE))],
            remaining_steps=remaining,
        )


def test_empty_candidates_and_empty_active_plans_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        choose_recovery("execution_failure", state_at(), [])
    with pytest.raises(ValueError, match="contain an action"):
        choose_recovery("execution_failure", state_at(), [RecoveryCandidate("empty", ())])


def test_memory_score_counts_successes_and_failures():
    memory = RecoveryMemory()
    assert memory.score("failure", "recovery") == 0.5
    memory.update("failure", "recovery", True)
    memory.update("failure", "recovery", False)
    memory.update("failure", "recovery", False)
    assert memory.attempts[("failure", "recovery")] == 3
    assert memory.successes[("failure", "recovery")] == 1
    assert memory.score("failure", "recovery") == pytest.approx(0.4)


@pytest.mark.parametrize("successes,attempts", [(1, 0), (-1, 2), (0, -1), (True, 2), (0, 1.5)])
def test_invalid_memory_counts_are_rejected_before_mutation(successes, attempts):
    key = ("failure", "recovery")
    memory = RecoveryMemory(successes={key: successes}, attempts={key: attempts})
    with pytest.raises(ValueError, match="memory counts"):
        memory.score(*key)
    with pytest.raises(ValueError, match="memory counts"):
        memory.update(*key, True)
    assert memory.successes[key] == successes
    assert memory.attempts[key] == attempts


@pytest.mark.parametrize(
    "state,expected",
    [
        (state_at(object_x=0.2), [ActionKind.GRASP, ActionKind.MOVE_TO_TARGET, ActionKind.PLACE]),
        (state_at(ee_x=0.8, object_x=0.8, holding=True), [ActionKind.PLACE]),
        (state_at(object_x=0.8), [ActionKind.MOVE_TO_OBJECT, ActionKind.GRASP, ActionKind.PLACE]),
        (
            state_at(ee_x=0.8, object_x=0.2),
            [ActionKind.MOVE_TO_OBJECT, ActionKind.GRASP, ActionKind.MOVE_TO_TARGET, ActionKind.PLACE],
        ),
    ],
)
def test_nominal_plan_skips_only_motions_already_completed(state, expected):
    env = TabletopManipulationEnv()
    env.reset()
    env.state = state.copy()
    plan = nominal_actions(state)
    assert [a.kind for a in plan] == expected
    assert env.nominal_plan(state) == plan
    assert env.rollout(plan).success


def test_grasp_tolerance_does_not_skip_uncompleted_approach():
    state = state_at(object_x=0.21)
    assert nominal_actions(state, EnvConfig(grasp_radius=0.1))[0].kind == ActionKind.MOVE_TO_OBJECT


def test_replanning_after_every_successful_action_still_makes_progress():
    env = TabletopManipulationEnv()
    state = env.reset()
    executed = []
    for _ in range(4):
        action = env.nominal_plan(state)[0]
        executed.append(action.kind)
        result = env.step(action)
        assert result.action_succeeded
        state = result.state
    assert state.success
    assert executed == [
        ActionKind.MOVE_TO_OBJECT, ActionKind.GRASP, ActionKind.MOVE_TO_TARGET, ActionKind.PLACE
    ]


def test_refresh_candidate_retains_grasp_and_omits_completed_transport():
    state = state_at(ee_x=0.8, object_x=0.8, holding=True)
    refreshed = next(
        c for c in candidate_recoveries("state_divergence", state)
        if c.name == "reobserve_reacquire_resume"
    )
    assert [a.kind for a in refreshed.actions] == [ActionKind.REOBSERVE, ActionKind.PLACE]
    assert SymbolicCounterfactualModel().rollout(state, list(refreshed.actions)).success


def test_refresh_candidate_reveals_then_grasps_at_observed_position():
    state = state_at(object_x=0.2, object_visible=False)
    refreshed = next(
        c for c in candidate_recoveries("perception_loss", state)
        if c.name == "refresh_then_replan"
    )
    assert [a.kind for a in refreshed.actions] == [
        ActionKind.REOBSERVE, ActionKind.GRASP, ActionKind.MOVE_TO_TARGET, ActionKind.PLACE
    ]


def test_shared_planning_respects_custom_environment_budget():
    state = state_at(object_x=0.2, step_index=25)
    assert nominal_actions(state) == []
    config = EnvConfig(max_steps=40)
    expected = nominal_actions(state, config)
    candidate = candidate_recoveries("state_divergence", state, config=config)[0]
    assert list(candidate.actions) == expected
    assert expected[0].kind == ActionKind.GRASP
    state.step_index = 40
    assert nominal_actions(state, config) == []
