from pathlib import Path

import pytest

from self_healing_embodied_agents.agents import OpenLoopAgent, SelfHealingAgent
from self_healing_embodied_agents.env import TabletopManipulationEnv
from self_healing_embodied_agents.training import load_bundle


@pytest.fixture(scope="session")
def bundle():
    path = Path("artifacts/world_model.pt")
    if not path.exists():
        pytest.skip("trained smoke artifact not present")
    return load_bundle(path)


def test_self_healing_recovers_from_slip(bundle) -> None:
    seed = 17
    baseline = OpenLoopAgent().run_episode(TabletopManipulationEnv(seed=seed, perturbation="grasp_slip"))
    healed = SelfHealingAgent(bundle).run_episode(TabletopManipulationEnv(seed=seed, perturbation="grasp_slip"))
    assert not baseline.success
    assert healed.success
    assert healed.recovery_attempts >= 1
