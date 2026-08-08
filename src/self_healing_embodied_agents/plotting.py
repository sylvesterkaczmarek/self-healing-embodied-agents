from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_success(summary: list[dict], output: Path) -> None:
    agents = ["open_loop", "reactive_replan", "self_healing"]
    perturbations = [
        "none",
        "grasp_slip",
        "object_displacement",
        "transient_occlusion",
        "blocked_path",
        "stale_observation",
        "compound_slip_block",
    ]
    x = list(range(len(perturbations)))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, agent in enumerate(agents):
        lookup = {(r["agent"], r["perturbation"]): r for r in summary}
        y = [lookup[(agent, p)]["success_rate"] for p in perturbations]
        offsets = [v + (idx - (len(agents) - 1) / 2) * width for v in x]
        ax.bar(offsets, y, width=width, label=agent.replace("_", " "))

    ax.set_ylabel("Task success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, [p.replace("_", "\n") for p in perturbations])
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_budget_success(summary: list[dict], output: Path) -> None:
    agents = ["open_loop", "reactive_replan", "self_healing"]
    perturbations = [
        "none",
        "grasp_slip",
        "object_displacement",
        "transient_occlusion",
        "blocked_path",
        "stale_observation",
        "compound_slip_block",
    ]
    x = list(range(len(perturbations)))
    width = 0.24
    lookup = {(r["agent"], r["perturbation"]): r for r in summary}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, agent in enumerate(agents):
        y = [lookup[(agent, p)]["success_within_7_actions"] for p in perturbations]
        offsets = [v + (idx - 1) * width for v in x]
        ax.bar(offsets, y, width=width, label=agent.replace("_", " "))

    ax.set_ylabel("Success within 7 actions")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, [p.replace("_", "\n") for p in perturbations])
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
