from __future__ import annotations

import argparse
from pathlib import Path

from self_healing_embodied_agents.benchmark import aggregate, run_benchmark, save_episode_csv, save_summary
from self_healing_embodied_agents.plotting import plot_budget_success, plot_success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/world_model.pt"))
    parser.add_argument("--seeds", type=int, default=10, help="number of benchmark seeds")
    parser.add_argument("--episodes-per-condition", type=int, default=3)
    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model,
        seeds=list(range(args.seeds)),
        episodes_per_condition=args.episodes_per_condition,
    )
    summary = aggregate(results)
    save_episode_csv(results, Path("results/episodes.csv"))
    save_summary(summary, Path("results/summary.json"), Path("results/summary.csv"))
    plot_success(summary, Path("results/success_by_perturbation.svg"))
    plot_budget_success(summary, Path("results/success_within_7_actions.svg"))

    print(f"episodes: {len(results)}")
    for row in summary:
        if row["perturbation"] in {"none", "grasp_slip", "compound_slip_block"}:
            print(f"{row['agent']:>20} | {row['perturbation']:<20} | success={row['success_rate']:.3f}")


if __name__ == "__main__":
    main()
