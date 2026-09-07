from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = {
    "world_model": {"seed": 7, "episodes": 500, "epochs": 120, "learning_rate": 0.002},
    "benchmark": {"seeds": 10, "episodes_per_condition": 3, "action_budget": 7},
}


def integer(name: str, value: Any, minimum: int = 1, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        limits = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {limits}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


def validate_config(config: dict) -> dict:
    import math
    from numbers import Real

    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError("configuration must contain world_model and benchmark objects")
    for section, defaults in DEFAULT_CONFIG.items():
        if not isinstance(config[section], dict) or set(config[section]) != set(defaults):
            raise ValueError(f"invalid fields in {section} configuration")
    wm, bench = config["world_model"], config["benchmark"]
    integer("world_model.seed", wm["seed"], 0, 2**32 - 1)
    integer("world_model.episodes", wm["episodes"], 2)
    integer("world_model.epochs", wm["epochs"])
    learning_rate = wm["learning_rate"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("world_model.learning_rate must be finite and positive")
    integer("benchmark.seeds", bench["seeds"], 1, 2**32)
    integer("benchmark.episodes_per_condition", bench["episodes_per_condition"])
    integer("benchmark.action_budget", bench["action_budget"])
    return config


def load_config(path: Path | None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        updates = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(updates, dict):
            raise ValueError("configuration must be a JSON object")
        for section, values in updates.items():
            if section not in config:
                raise ValueError(f"unknown configuration section: {section}")
            if not isinstance(values, dict):
                raise ValueError(f"{section} must be a JSON object")
            unknown = set(values) - set(config[section])
            if unknown:
                raise ValueError(f"unknown {section} settings: {', '.join(sorted(unknown))}")
            config[section].update(values)
    return validate_config(config)


def benchmark_outputs(results: list, summary: list[dict], output: Path, action_budget: int) -> None:
    from self_healing_embodied_agents.benchmark import save_episode_csv, save_episode_jsonl, save_summary
    from self_healing_embodied_agents.plotting import plot_budget_success, plot_success

    output.mkdir(parents=True, exist_ok=True)
    save_episode_csv(results, output / "episodes.csv")
    save_episode_jsonl(results, output / "episodes.jsonl")
    save_summary(summary, output / "summary.json", output / "summary.csv")
    plot_success(summary, output / "success_by_perturbation.svg")
    plot_budget_success(summary, output / f"success_within_{action_budget}_actions.svg")


def write_json(value: Any, path: Path) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def publish_files(stage: Path, destination: Path) -> None:
    """Publish a complete run, rolling back replaced files if publication fails.

    This protects ordinary execution/write failures. Multiple files cannot be
    replaced atomically for concurrent readers; callers should use separate run
    directories and should not run two publishers against the same destination.
    """
    sources = sorted(path for path in stage.rglob("*") if path.is_file())
    destination.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=".publication-backup-", dir=destination))
    cleanup_backup = False
    replaced: list[tuple[Path, Path | None]] = []
    try:
        for source in sources:
            relative = source.relative_to(stage)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            previous = None
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"output target is not a file: {target}")
                previous = backup / relative
                previous.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, previous)
            os.replace(source, target)
            replaced.append((target, previous))
        cleanup_backup = True
    except BaseException as publication_error:
        rollback_errors = []
        for target, previous in reversed(replaced):
            try:
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    # Keep the backup until every restoration has succeeded.
                    shutil.copy2(previous, target)
            except OSError as error:
                rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                f"publication and rollback failed; previous files remain in {backup}"
            ) from publication_error
        cleanup_backup = True
        raise
    finally:
        if cleanup_backup:
            shutil.rmtree(backup, ignore_errors=True)
