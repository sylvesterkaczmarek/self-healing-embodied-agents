from self_healing_embodied_agents.env import TabletopManipulationEnv


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
