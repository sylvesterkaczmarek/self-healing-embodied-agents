from __future__ import annotations

from pathlib import Path

from self_healing_embodied_agents.agents import SelfHealingAgent
from self_healing_embodied_agents.env import TabletopManipulationEnv
from self_healing_embodied_agents.training import load_bundle


def main() -> None:
    bundle = load_bundle(Path("artifacts/world_model.pt"))
    env = TabletopManipulationEnv(seed=13, perturbation="grasp_slip")
    result = SelfHealingAgent(bundle).run_episode(env)
    print(f"success={result.success} steps={result.steps} interventions={result.interventions}")
    for event in result.event_log:
        print(event)


if __name__ == "__main__":
    main()
