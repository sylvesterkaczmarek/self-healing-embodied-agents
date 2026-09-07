from __future__ import annotations

from html import escape
from math import isfinite

from .env import PERTURBATIONS
from pathlib import Path


def _write_grouped_bar_svg(
    summary: list[dict],
    output: Path,
    *,
    metric: str,
    ylabel: str,
) -> None:
    if not summary:
        raise ValueError("summary must contain at least one row")
    lookup = {(row["agent"], row["perturbation"]): row for row in summary}
    if len(lookup) != len(summary):
        raise ValueError("duplicate agent/perturbation summary row")
    available_agents = {key[0] for key in lookup}
    preferred_agents = ["open_loop", "reactive_replan", "self_healing", "self_healing_memory"]
    agents = [agent for agent in preferred_agents if agent in available_agents]
    agents += sorted(available_agents - set(agents))
    available_perturbations = {key[1] for key in lookup}
    perturbations = [name for name in PERTURBATIONS if name in available_perturbations]
    perturbations += sorted(available_perturbations - set(perturbations))
    for agent in agents:
        for perturbation in perturbations:
            if (agent, perturbation) not in lookup:
                raise ValueError("summary must contain every agent/perturbation combination")
            value = lookup[(agent, perturbation)][metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{metric} must be a finite rate in [0, 1]")
    width, height = 1100, 560
    left, right, top, bottom = 90, 35, 35, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(perturbations)
    bar_w = group_w * 0.80 / len(agents)
    fills = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(ylabel)} by perturbation">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" font-size="14" fill="#222">',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end">{value:.1f}</text>')

    for p_idx, perturbation in enumerate(perturbations):
        center = left + (p_idx + 0.5) * group_w
        for a_idx, agent in enumerate(agents):
            value = float(lookup[(agent, perturbation)][metric])
            x = center + (a_idx - (len(agents) - 1) / 2) * bar_w - bar_w * 0.42
            h = value * plot_h
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.84:.1f}" height="{h:.1f}" fill="{fills[a_idx % len(fills)]}"/>')
        label = escape(perturbation.replace("_", " "))
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+28}" text-anchor="middle" transform="rotate(-24 {center:.1f} {height-bottom+28})">{label}</text>')

    parts.append(f'<text x="24" y="{top + plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h/2:.1f})">{escape(ylabel)}</text>')
    legend_step = min(220, plot_w / len(agents))
    legend_x = left
    for idx, agent in enumerate(agents):
        x = legend_x + idx * legend_step
        parts.append(f'<rect x="{x}" y="{height-35}" width="16" height="16" fill="{fills[idx % len(fills)]}"/>')
        parts.append(f'<text x="{x+23}" y="{height-22}">{escape(agent.replace("_", " "))}</text>')
    parts.append('</g></svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_success(summary: list[dict], output: Path) -> None:
    _write_grouped_bar_svg(summary, output, metric="success_rate", ylabel="Task success rate")


def plot_budget_success(summary: list[dict], output: Path) -> None:
    budgets = {row.get("action_budget", 7) for row in summary}
    if len(budgets) != 1:
        raise ValueError("summary must use one action budget")
    budget = budgets.pop()
    if type(budget) is not int or budget < 1:
        raise ValueError("action_budget must be a positive integer")
    # Legacy summaries remain readable, but a mixed schema is ambiguous.
    generic = ["success_within_action_budget" in row for row in summary]
    if any(generic) and not all(generic):
        raise ValueError("summary mixes action-budget metric schemas")
    if not all(generic) and budget != 7:
        raise ValueError("legacy summary metric is defined only for seven actions")
    metric = "success_within_action_budget" if all(generic) else "success_within_7_actions"
    _write_grouped_bar_svg(
        summary,
        output,
        metric=metric,
        ylabel=f"Success within {budget} actions",
    )
