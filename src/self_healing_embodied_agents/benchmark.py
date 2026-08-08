from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .agents import OpenLoopAgent, ReactiveReplanAgent, SelfHealingAgent
from .env import PERTURBATIONS, TabletopManipulationEnv
from .recovery import RecoveryMemory
from .training import load_bundle
from .types import EpisodeResult


def run_benchmark(
    *,
    model_path: Path,
    seeds: list[int],
    episodes_per_condition: int = 3,
) -> list[EpisodeResult]:
    bundle = load_bundle(model_path)
    agents = [OpenLoopAgent(), ReactiveReplanAgent(), SelfHealingAgent(bundle)]
    memory_agent = SelfHealingAgent(bundle, memory=RecoveryMemory())
    results: list[EpisodeResult] = []

    for perturbation in PERTURBATIONS:
        for seed in seeds:
            for repetition in range(episodes_per_condition):
                episode_seed = seed * 100 + repetition
                for agent in agents:
                    env = TabletopManipulationEnv(seed=episode_seed, perturbation=perturbation)
                    results.append(agent.run_episode(env))
                env = TabletopManipulationEnv(seed=episode_seed, perturbation=perturbation)
                results.append(memory_agent.run_episode(env))
    return results


def save_episode_csv(results: list[EpisodeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_dict() for r in results]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(results: list[EpisodeResult]) -> list[dict]:
    groups: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    for r in results:
        groups[(r.agent, r.perturbation)].append(r)

    summary: list[dict] = []
    for (agent, perturbation), rows in sorted(groups.items()):
        successes = np.asarray([r.success for r in rows], dtype=np.float64)
        steps = np.asarray([r.steps for r in rows], dtype=np.float64)
        recoveries = np.asarray([r.recovery_attempts for r in rows], dtype=np.float64)
        detection_precision = []
        detection_recall = []
        for r in rows:
            if r.detections:
                detection_precision.append(r.true_positive_detections / r.detections)
            if r.true_failures:
                detection_recall.append(r.true_positive_detections / r.true_failures)

        summary.append(
            {
                "agent": agent,
                "perturbation": perturbation,
                "episodes": len(rows),
                "success_rate": float(successes.mean()),
                "success_std": float(successes.std(ddof=0)),
                "success_within_7_actions": float(np.mean([r.success and r.steps <= 7 for r in rows])),
                "mean_steps": float(steps.mean()),
                "mean_recovery_attempts": float(recoveries.mean()),
                "detection_precision": float(np.mean(detection_precision)) if detection_precision else None,
                "detection_recall": float(np.mean(detection_recall)) if detection_recall else None,
            }
        )
    return summary


def save_summary(summary: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
