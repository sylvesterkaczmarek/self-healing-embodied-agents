from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .agents import OpenLoopAgent, ReactiveReplanAgent, SelfHealingAgent
from .env import PERTURBATIONS, TabletopManipulationEnv
from .recovery import RecoveryMemory
from .training import load_bundle
from .types import EpisodeResult


def validate_seeds(seeds: list[int]) -> None:
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list of distinct integers")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("root seeds must be integers in [0, 2**32)")
    if len(set(seeds)) != len(seeds):
        raise ValueError("root seeds must be distinct")


def episode_seed(root_seed: int, repetition: int) -> int:
    """Derive paired evaluation seeds without arithmetic-offset overlap."""
    validate_seeds([root_seed])
    if type(repetition) is not int or repetition < 0:
        raise ValueError("repetition must be a nonnegative integer")
    return int(
        np.random.SeedSequence([root_seed, repetition, 0x4556414C])
        .generate_state(1, dtype=np.uint64)[0]
    )


def run_benchmark(
    *,
    model_path: Path,
    seeds: list[int],
    episodes_per_condition: int = 3,
) -> list[EpisodeResult]:
    validate_seeds(seeds)
    if type(episodes_per_condition) is not int or episodes_per_condition < 1:
        raise ValueError("episodes_per_condition must be a positive integer")
    bundle = load_bundle(model_path)
    results: list[EpisodeResult] = []

    for root_seed in seeds:
        # A root seed is an independent run. Memory persists only through this
        # run's fixed perturbation/repetition order, never between root seeds.
        agents = [
            OpenLoopAgent(),
            ReactiveReplanAgent(),
            SelfHealingAgent(bundle),
            SelfHealingAgent(bundle, memory=RecoveryMemory()),
        ]
        for perturbation in PERTURBATIONS:
            for repetition in range(episodes_per_condition):
                seed = episode_seed(root_seed, repetition)
                for agent in agents:
                    env = TabletopManipulationEnv(seed=seed, perturbation=perturbation)
                    result = agent.run_episode(env)
                    result.root_seed = root_seed
                    result.repetition = repetition
                    results.append(result)
    return results


def _validate_results(results: list[EpisodeResult]) -> None:
    if not results:
        raise ValueError("results must contain at least one episode")
    identities: set[tuple[str, str, int]] = set()
    coordinates: dict[tuple[int, int], int] = {}
    for result in results:
        if not isinstance(result, EpisodeResult):
            raise ValueError("results must contain EpisodeResult instances")
        for name in ("agent", "perturbation"):
            if not isinstance(getattr(result, name), str) or not getattr(result, name):
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(result.success, (bool, np.bool_)):
            raise ValueError("success must be boolean")
        for name in (
            "seed", "steps", "interventions", "true_failures", "detections",
            "true_positive_detections", "false_positive_detections",
            "recovery_attempts", "recovery_successes",
        ):
            value = getattr(result, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if result.detections != result.true_positive_detections + result.false_positive_detections:
            raise ValueError("detections must equal true positives plus false positives")
        if result.true_positive_detections > result.true_failures:
            raise ValueError("true-positive detections cannot exceed true failure steps")
        if result.recovery_successes > result.recovery_attempts:
            raise ValueError("recovery successes cannot exceed recovery attempts")
        for name in ("detections", "interventions", "true_failures", "recovery_attempts"):
            if getattr(result, name) > result.steps:
                raise ValueError(f"{name} cannot exceed executed steps")
        identity = (result.agent, result.perturbation, result.seed)
        if identity in identities:
            raise ValueError(f"duplicate episode: {identity}")
        identities.add(identity)
        if (result.root_seed is None) != (result.repetition is None):
            raise ValueError("root_seed and repetition must both be supplied or both omitted")
        if result.root_seed is not None:
            validate_seeds([result.root_seed])
            if type(result.repetition) is not int or result.repetition < 0:
                raise ValueError("repetition must be a nonnegative integer")
            coordinate = (result.root_seed, result.repetition)
            if coordinate in coordinates and coordinates[coordinate] != result.seed:
                raise ValueError("a root seed/repetition pair must use one shared episode seed")
            coordinates[coordinate] = result.seed


def _csv_text(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("cannot save an empty result table")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def save_episode_csv(results: list[EpisodeResult], path: Path) -> None:
    _validate_results(results)
    text = _csv_text([result.as_dict() for result in results])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def save_episode_jsonl(results: list[EpisodeResult], path: Path) -> None:
    """Save counters and the complete per-episode evidence behind them."""
    _validate_results(results)
    text = "".join(
        json.dumps({**result.as_dict(), "event_log": result.event_log}, allow_nan=False) + "\n"
        for result in results
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def aggregate(results: list[EpisodeResult], *, action_budget: int = 7) -> list[dict]:
    """Pool detection counts; success spread is descriptive episode-level SD.

    Failure units are unique injected-failure steps, with at most one matched
    detection per failure step. Undefined precision/recall are represented by
    None. Correlated repetitions are not independent uncertainty estimates.
    """
    if type(action_budget) is not int or action_budget < 1:
        raise ValueError("action_budget must be a positive integer")
    _validate_results(results)
    groups: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        groups[(result.agent, result.perturbation)].append(result)

    summary: list[dict] = []
    for (agent, perturbation), rows in sorted(groups.items()):
        successes = np.asarray([result.success for result in rows], dtype=np.float64)
        steps = np.asarray([result.steps for result in rows], dtype=np.float64)
        recoveries = np.asarray([result.recovery_attempts for result in rows], dtype=np.float64)
        detections = sum(result.detections for result in rows)
        true_positives = sum(result.true_positive_detections for result in rows)
        failures = sum(result.true_failures for result in rows)
        summary.append(
            {
                "agent": agent,
                "perturbation": perturbation,
                "episodes": len(rows),
                "success_rate": float(successes.mean()),
                "success_std": float(successes.std(ddof=0)),
                "action_budget": action_budget,
                "success_within_action_budget": float(
                    np.mean([result.success and result.steps <= action_budget for result in rows])
                ),
                "success_within_7_actions": float(
                    np.mean([result.success and result.steps <= 7 for result in rows])
                ),
                "mean_steps": float(steps.mean()),
                "mean_recovery_attempts": float(recoveries.mean()),
                "detections": detections,
                "true_positive_detections": true_positives,
                "false_positive_detections": sum(result.false_positive_detections for result in rows),
                "true_failures": failures,
                "detection_precision": true_positives / detections if detections else None,
                "detection_recall": true_positives / failures if failures else None,
            }
        )
    return summary


def save_summary(summary: list[dict], json_path: Path, csv_path: Path) -> None:
    if not summary:
        raise ValueError("summary must contain at least one row")
    # Serialise both files first, so invalid values cannot replace good output.
    json_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    csv_text = _csv_text(summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    csv_path.write_text(csv_text, encoding="utf-8", newline="")
