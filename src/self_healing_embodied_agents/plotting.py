from __future__ import annotations

from html import escape
from pathlib import Path


def _write_grouped_bar_svg(
    summary: list[dict],
    output: Path,
    *,
    metric: str,
    ylabel: str,
) -> None:
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
    lookup = {(r["agent"], r["perturbation"]): r for r in summary}
    width, height = 1100, 560
    left, right, top, bottom = 90, 35, 35, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(perturbations)
    bar_w = group_w * 0.20
    fills = ["#4c78a8", "#f58518", "#54a24b"]

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
            x = center + (a_idx - 1) * bar_w - bar_w * 0.42
            h = value * plot_h
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.84:.1f}" height="{h:.1f}" fill="{fills[a_idx]}"/>')
        label = escape(perturbation.replace("_", " "))
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+28}" text-anchor="middle" transform="rotate(-24 {center:.1f} {height-bottom+28})">{label}</text>')

    parts.append(f'<text x="24" y="{top + plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h/2:.1f})">{escape(ylabel)}</text>')
    legend_x = width - right - 450
    for idx, agent in enumerate(agents):
        x = legend_x + idx * 150
        parts.append(f'<rect x="{x}" y="{height-35}" width="16" height="16" fill="{fills[idx]}"/>')
        parts.append(f'<text x="{x+23}" y="{height-22}">{escape(agent.replace("_", " "))}</text>')
    parts.append('</g></svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_success(summary: list[dict], output: Path) -> None:
    _write_grouped_bar_svg(summary, output, metric="success_rate", ylabel="Task success rate")


def plot_budget_success(summary: list[dict], output: Path) -> None:
    _write_grouped_bar_svg(
        summary,
        output,
        metric="success_within_7_actions",
        ylabel="Success within 7 actions",
    )
