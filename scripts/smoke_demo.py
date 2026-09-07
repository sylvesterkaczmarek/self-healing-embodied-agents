from __future__ import annotations

import argparse
from pathlib import Path

from self_healing_embodied_agents.agents import SelfHealingAgent
from self_healing_embodied_agents.env import PERTURBATIONS, TabletopManipulationEnv
from self_healing_embodied_agents.training import load_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show a self-healing episode and its recorded events.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/world_model.pt"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--perturbation", choices=PERTURBATIONS, default="grasp_slip")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be in 0..2**64-1")
    bundle = load_bundle(args.model)
    env = TabletopManipulationEnv(seed=args.seed, perturbation=args.perturbation)
    result = SelfHealingAgent(bundle).run_episode(env)
    print(f"success={result.success} steps={result.steps} interventions={result.interventions}")
    for event in result.event_log:
        print(event)


if __name__ == "__main__":
    main()
