from __future__ import annotations

import json
from pathlib import Path

from self_healing_embodied_agents.benchmark import aggregate, run_benchmark, save_episode_csv, save_summary
from self_healing_embodied_agents.plotting import plot_budget_success, plot_success
from self_healing_embodied_agents.training import save_bundle, train_transition_model


def main() -> None:
    config = json.loads(Path("configs/reproduce.json").read_text(encoding="utf-8"))
    wm = config["world_model"]
    bench = config["benchmark"]

    bundle, metrics = train_transition_model(
        seed=int(wm["seed"]),
        episodes=int(wm["episodes"]),
        epochs=int(wm["epochs"]),
    )
    save_bundle(bundle, metrics, Path("artifacts/world_model.pt"), Path("results/world_model_metrics.json"))

    results = run_benchmark(
        model_path=Path("artifacts/world_model.pt"),
        seeds=list(range(int(bench["seeds"]))),
        episodes_per_condition=int(bench["episodes_per_condition"]),
    )
    summary = aggregate(results)
    save_episode_csv(results, Path("results/episodes.csv"))
    save_summary(summary, Path("results/summary.json"), Path("results/summary.csv"))
    plot_success(summary, Path("results/success_by_perturbation.svg"))
    plot_budget_success(summary, Path("results/success_within_7_actions.svg"))
    print(f"reproduced {len(results)} episodes")


if __name__ == "__main__":
    main()
