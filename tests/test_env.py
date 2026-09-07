import numpy as np
import pytest

from self_healing_embodied_agents.env import EnvConfig, TabletopManipulationEnv, nominal_transition
from self_healing_embodied_agents.types import Action, ActionKind, WorldState



def test_nominal_plan_completes() -> None:
    env = TabletopManipulationEnv(seed=3, perturbation="none")
    state = env.reset()
    for action in env.nominal_plan(state):
        state = env.step(action).state
    assert state.success


def test_seed_is_deterministic() -> None:
    a = TabletopManipulationEnv(seed=9).reset()
    b = TabletopManipulationEnv(seed=9).reset()
    assert (a.vector() == b.vector()).all()


def test_slip_is_injected() -> None:
    env = TabletopManipulationEnv(seed=2, perturbation="grasp_slip")
    state = env.reset()
    for action in env.nominal_plan(state):
        env.step(action)
    assert any(e["event"] == "grasp_slip" for e in env.events)


def test_occlusion_expires_after_two_reads_without_permanent_blindness() -> None:
    env = TabletopManipulationEnv(seed=3, perturbation="transient_occlusion")
    env.reset()
    result = env.step(Action(ActionKind.MOVE_TO_OBJECT))
    assert not result.action_succeeded
    assert not result.state.object_visible
    assert not env.observe().object_visible
    assert env.observe().object_visible
    assert env.step(Action(ActionKind.MOVE_TO_OBJECT)).action_succeeded
    assert [(e["event"], e["step"]) for e in env.events] == [("transient_occlusion", 1)]


def test_reobserve_clears_pending_occlusion_immediately() -> None:
    env = TabletopManipulationEnv(perturbation="transient_occlusion")
    env.reset()
    env.step(Action(ActionKind.MOVE_TO_OBJECT))
    result = env.step(Action(ActionKind.REOBSERVE))
    assert result.state.object_visible
    assert env.observe().object_visible
    assert result.state.step_index == 2


def test_stale_read_uses_last_observation_then_returns_current_state() -> None:
    env = TabletopManipulationEnv(seed=3, perturbation="stale_observation")
    env.reset()
    previous = env.step(Action(ActionKind.MOVE_TO_OBJECT)).state
    result = env.step(Action(ActionKind.GRASP))
    np.testing.assert_array_equal(result.state.vector(), previous.vector())
    assert result.state.step_index == 2
    current = env.observe()
    np.testing.assert_array_equal(current.vector(), env.state.vector())
    assert not np.array_equal(current.vector(), previous.vector())
    for _ in range(100):
        env.observe()
    assert len(env._observation_history) == 1


def test_environment_step_budget_is_absorbing() -> None:
    env = TabletopManipulationEnv(config=EnvConfig(max_steps=1))
    env.reset()
    first = env.step(Action(ActionKind.MOVE_TO_OBJECT))
    assert env.done
    assert not first.state.success
    assert env.nominal_plan() == []
    for kind in ActionKind:
        result = env.step(Action(kind))
        assert not result.action_succeeded
        assert result.state.step_index == 1
        np.testing.assert_array_equal(result.state.vector(), first.state.vector())
        assert result.events == []


def test_successful_state_is_absorbing() -> None:
    env = TabletopManipulationEnv()
    env.reset()
    state = env.rollout(env.nominal_plan())
    assert state.success and env.done
    for kind in ActionKind:
        result = env.step(Action(kind))
        assert result.action_succeeded
        assert result.state.step_index == state.step_index
        np.testing.assert_array_equal(result.state.vector(), state.vector())


def test_custom_workspace_contains_reset_geometry() -> None:
    env = TabletopManipulationEnv(seed=19, config=EnvConfig(workspace_low=0.5, workspace_high=0.6))
    state = env.reset()
    coordinates = state.vector()[:6]
    assert np.all(coordinates >= 0.5)
    assert np.all(coordinates <= np.float32(0.6))
    assert state.object_xy[0] < state.target_xy[0]
    assert env.rollout(env.nominal_plan()).success


def test_default_reset_keeps_original_seeded_geometry() -> None:
    rng = np.random.default_rng(19)
    expected_object = rng.uniform([0.18, 0.18], [0.42, 0.82]).astype(np.float32)
    expected_target = rng.uniform([0.63, 0.18], [0.88, 0.82]).astype(np.float32)
    state = TabletopManipulationEnv(seed=19).reset()
    np.testing.assert_array_equal(state.object_xy, expected_object)
    np.testing.assert_array_equal(state.target_xy, expected_target)
    np.testing.assert_array_equal(state.ee_xy, np.asarray([0.1, 0.5], dtype=np.float32))


def test_compound_slip_waits_until_blocked_transport_can_execute() -> None:
    env = TabletopManipulationEnv(seed=2, perturbation="compound_slip_block")
    env.reset()
    env.step(Action(ActionKind.MOVE_TO_OBJECT))
    env.step(Action(ActionKind.GRASP))
    blocked = env.step(Action(ActionKind.MOVE_TO_TARGET))
    assert not blocked.action_succeeded
    assert blocked.state.holding
    assert [(e["event"], e["step"]) for e in blocked.events] == [("blocked_path", 3)]
    env.step(Action(ActionKind.CLEAR_PATH))
    moved = env.step(Action(ActionKind.MOVE_TO_TARGET))
    assert moved.action_succeeded
    assert not moved.state.holding
    assert [(e["event"], e["step"]) for e in moved.events] == [("grasp_slip", 5)]


def test_nominal_transition_is_pure_and_uses_configured_radii() -> None:
    state = WorldState(ee_xy=[0.1, 0.1], object_xy=[0.17, 0.1], target_xy=[0.5, 0.5])
    failed = nominal_transition(state, Action(ActionKind.GRASP))
    passed = nominal_transition(state, Action(ActionKind.GRASP), EnvConfig(grasp_radius=0.08))
    assert not failed.state.holding and not failed.action_succeeded
    assert passed.state.holding and passed.action_succeeded
    assert state.step_index == 0 and not state.holding
    assert failed.state.step_index == passed.state.step_index == 1


@pytest.mark.parametrize("kwargs", [
    {"grasp_radius": -1}, {"placement_radius": -1}, {"grasp_radius": float("nan")},
    {"placement_radius": float("inf")}, {"workspace_low": float("nan")},
    {"workspace_high": float("inf")}, {"workspace_low": 1, "workspace_high": 0},
    {"workspace_low": 1, "workspace_high": 1}, {"workspace_high": 1e100},
    {"grasp_radius": True}, {"max_steps": 0}, {"max_steps": 1.5}, {"max_steps": True},
])
def test_environment_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


@pytest.mark.parametrize("seed", [-1, 0.5, True, "3"])
def test_environment_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError):
        TabletopManipulationEnv(seed=seed)


@pytest.mark.parametrize("coordinates", [[0.1], [[0.1, 0.2]], [float("nan"), 0.1], [0.1, float("inf")], [1e100, 0.1]])
def test_world_state_rejects_invalid_coordinates(coordinates: list) -> None:
    with pytest.raises(ValueError, match="coordinates"):
        WorldState(ee_xy=coordinates, object_xy=[0.1, 0.2], target_xy=[0.8, 0.2])


@pytest.mark.parametrize("vector", [np.zeros(8), np.zeros(10), np.zeros((1, 9)), np.full(9, np.nan), np.array([0] * 6 + [2, 1, 0])])
def test_state_vector_rejects_malformed_or_nonfinite_values(vector: np.ndarray) -> None:
    with pytest.raises(ValueError):
        WorldState.from_vector(vector)


def test_world_state_copies_caller_owned_coordinates() -> None:
    xy = np.asarray([0.1, 0.2], dtype=np.float32)
    state = WorldState(ee_xy=xy, object_xy=xy, target_xy=xy)
    xy[:] = 0
    np.testing.assert_array_equal(state.ee_xy, np.asarray([0.1, 0.2], dtype=np.float32))
    state.ee_xy[:] = 0
    assert state.object_xy[0] != 0


def test_action_string_is_normalised_and_invalid_kind_rejected() -> None:
    action = Action("grasp")
    assert action.kind is ActionKind.GRASP
    assert str(action) == "grasp"
    with pytest.raises(ValueError, match="unknown action"):
        Action("unknown")


def test_uninitialised_environment_has_clear_errors() -> None:
    env = TabletopManipulationEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.observe()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action(ActionKind.GRASP))
