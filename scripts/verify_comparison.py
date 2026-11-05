"""Recompute complete study evidence and optionally replay recorded simulator actions.

This is a local consistency check, not authenticated provenance or independent
training reproduction. It does not prove the recorded agent selected an action.
"""

from __future__ import annotations
import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np
from self_healing_embodied_agents.benchmark import episode_seed
from self_healing_embodied_agents.comparison import METHODS, episode_metrics, summarize
from self_healing_embodied_agents.evidence import (
    read_json,
    strict_json,
    validate_episode,
    finite,
)
from self_healing_embodied_agents.env import EnvConfig, TabletopManipulationEnv
from self_healing_embodied_agents.release_contracts import validate_calibration_protocol, validate_provenance
from self_healing_embodied_agents.recovery import RecoveryMemory, TaskOutcomeMemory
from self_healing_embodied_agents.training import (
    TrainingMetrics,
    validate_training_metrics,
    load_bundle,
)
from self_healing_embodied_agents.types import Action
from self_healing_embodied_agents.provenance import file_sha256
from scripts.run_comparison import load_study, condition_orders, memory_snapshot


def same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-12)
        )
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            same(left[k], right[k]) for k in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def read_rows(path):
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 64 * 1024 * 1024
    ):
        raise ValueError("invalid compressed trace file")
    rows = []
    total = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        while True:
            line = stream.readline(512 * 1024 + 1)
            if not line:
                break
            total += len(line)
            if (
                len(line) > 512 * 1024
                or total > 256 * 1024 * 1024
                or len(rows) >= 100000
            ):
                raise ValueError("trace exceeds supported evidence budget")
            rows.append(strict_json(line))
    return rows


def replay_actions(row, max_steps):
    env = TabletopManipulationEnv(
        seed=row["seed"],
        perturbation=row["perturbation"],
        config=EnvConfig(max_steps=max_steps),
    )
    env.reset()
    events = []
    transitions = [e for e in row["event_log"] if e["event"] == "transition"]
    for item in transitions:
        if env.done:
            raise ValueError("recorded action after terminal state")
        observed = env.step(Action(item["action"]))
        events.extend(observed.events)
        if (
            observed.state.step_index != item["step"]
            or observed.action_succeeded is not item["action_succeeded"]
            or observed.state.success is not item["task_success"]
        ):
            raise ValueError(
                "transition result differs from deterministic action replay"
            )
        if not np.allclose(
            observed.state.vector(), item["observed_state"], rtol=2e-7, atol=2e-7
        ):
            raise ValueError("recorded observation differs from action replay")
    expected_events = [e for e in row["event_log"] if "failure" in e]
    if len(events) != len(expected_events):
        raise ValueError("fault event count mismatch")
    for a, b in zip(events, expected_events):
        if a.keys() != b.keys():
            raise ValueError("fault event fields mismatch")
        for key, value in a.items():
            if type(value) is float:
                if type(b[key]) is not float or not math.isclose(
                    value, b[key], rel_tol=2e-7, abs_tol=2e-7
                ):
                    raise ValueError("fault payload differs from action replay")
            elif type(value) is not type(b[key]) or value != b[key]:
                raise ValueError("fault event differs from action replay")
    return len(transitions)


def verify(directory, *, replay=False):
    root = Path(directory).resolve(strict=True)
    config = load_study(root / "config.json")
    summary = read_json(root / "summary.json")
    split = summary.get("split")
    if split not in ("development", "evaluation"):
        raise ValueError("invalid study split")
    settings = config[split]
    orders = condition_orders()
    expected_files = {
        "config.json",
        "summary.json",
        "memory_snapshots.json",
        "episodes.jsonl.gz",
    }
    expected_files |= {
        f"models/{seed}/{name}"
        for seed in settings["training_seeds"]
        for name in ("world_model.pt", "training_metrics.json", "provenance.json")
    }
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("redirected study artifacts are unsupported")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected_files:
        raise ValueError("incomplete or unexpected study file set")
    rows = read_rows(root / "episodes.jsonl.gz")
    if summary["trace_sha256"] != file_sha256(root / "episodes.jsonl.gz") or summary[
        "config_sha256"
    ] != file_sha256(root / "config.json"):
        raise ValueError("summary file binding mismatch")
    snapshots = read_json(root / "memory_snapshots.json")
    expected_snapshots = []
    expected = []
    for model in settings["training_seeds"]:
        for h, order in enumerate(orders):
            for phase, methods in [("fit", METHODS[6:8]), ("evaluation", METHODS)]:
                for condition in order:
                    for rep in range(config["episodes_per_condition"]):
                        expected.extend(
                            (model, h, phase, condition, rep, m) for m in methods
                        )
    if len(rows) != len(expected):
        raise ValueError("incomplete or extra declared study episodes")
    metrics_rows = []
    memories = {}
    action_count = 0
    for row, identity in zip(rows, expected):
        model, h, phase, condition, rep, method = identity
        actual = (
            row["training_seed"],
            row["history_id"],
            row["phase"],
            row["perturbation"],
            row["repetition"],
            row["agent"],
        )
        if any(type(a) is not type(b) or a != b for a, b in zip(actual, identity)):
            raise ValueError("row identity/order differs from declared design")
        seed_root = settings["fit_roots" if phase == "fit" else "evaluation_roots"][h]
        if (
            type(row["root_seed"]) is not int
            or row["root_seed"] != seed_root
            or row["seed"] != episode_seed(seed_root, rep)
            or row["order"] != orders[h]
        ):
            raise ValueError("unpaired root, episode seed or history order")
        if row["steps"] > config["max_steps"]:
            raise ValueError("executed actions exceed study budget")
        validate_episode(row, complete=True)
        metrics = episode_metrics(row)
        for key, value in metrics.items():
            if key not in row or not same(row[key], value):
                raise ValueError("stored trace metric differs from events")
        if replay:
            action_count += replay_actions(row, config["max_steps"])
        key = (model, h)
        if key not in memories:
            memories[key] = (RecoveryMemory(), TaskOutcomeMemory())
        updates = [e for e in row["event_log"] if e["event"] == "memory_update"]
        selections = [e for e in row["event_log"] if e["event"] == "recovery_selected"]
        if "task_memory" in method and not method.endswith("_frozen"):
            if len(updates) != len(selections):
                raise ValueError("task memory needs one terminal credit per selection")
            for selected, event in zip(selections, updates):
                for field in ("candidate", "failure_class"):
                    if event[field] != selected[field]:
                        raise ValueError("task memory credit candidate mismatch")
                remaining = row["steps"] - selected["step"]
                if (
                    type(event["task_success"]) is not bool
                    or event["task_success"] != bool(row["success"])
                    or type(event["selected_step"]) is not int
                    or event["selected_step"] != selected["step"]
                    or type(event["remaining_actions"]) is not int
                    or event["remaining_actions"] != remaining
                    or event["step"] != row["steps"]
                ):
                    raise ValueError("task memory terminal outcome mismatch")
                finite(event["reward"], "reward", 0, 1)
                if not math.isclose(
                    event["reward"], row["success"] / (1 + remaining), abs_tol=1e-12
                ):
                    raise ValueError("incorrect task reward")
        elif updates:
            raise ValueError("unexpected memory updates for this method")
        if phase == "fit":
            if method == "self_healing_memory":
                for event in row["event_log"]:
                    if event["event"] == "recovery_outcome":
                        memories[key][0].update(
                            event["failure_class"], event["candidate"], event["success"]
                        )
            else:
                for event in updates:
                    memories[key][1].update(
                        event["failure_class"],
                        event["candidate"],
                        event["task_success"],
                        event["remaining_actions"],
                    )
        metrics_rows.append(row)
    for model in settings["training_seeds"]:
        folder = root / "models" / str(model)
        bundle = load_bundle(folder / "world_model.pt")
        raw = read_json(folder / "training_metrics.json")
        metrics = TrainingMetrics(**raw)
        validate_training_metrics(metrics)
        validate_calibration_protocol(metrics, config["training"])
        if (
            metrics.seed != model
            or metrics.epochs != config["training"]["epochs"]
            or metrics.train_episodes + metrics.validation_episodes
            != config["training"]["episodes"]
        ):
            raise ValueError("training metrics contradict declared plan")
        if bundle.residual_threshold != metrics.residual_threshold:
            raise ValueError("calibration threshold differs from checkpoint")
        provenance = read_json(folder / "provenance.json")
        validate_provenance(provenance)
        if (
            provenance.get("checkpoint_sha256")
            != file_sha256(folder / "world_model.pt")
            or provenance.get("config_sha256") != summary["config_sha256"]
        ):
            raise ValueError("model provenance hash binding mismatch")
        checks = {
            "study_split": split,
            "effective_config": config,
            "evaluation_seeds": settings["evaluation_roots"],
            "fit_roots": settings["fit_roots"],
            "training_metrics": raw,
            "ordered_conditions": orders,
            "reference_commit": config["reference_commit"],
        }
        for name, value in checks.items():
            if not same(provenance.get(name), value):
                raise ValueError("model provenance does not match study")
        sources = provenance.get("source_sha256")
        if (
            not isinstance(sources, dict)
            or not sources
            or any(
                type(v) is not str
                or len(v) != 64
                or any(c not in "0123456789abcdef" for c in v)
                for v in sources.values()
            )
        ):
            raise ValueError("invalid source fingerprints")
        for h, order in enumerate(orders):
            expected_snapshots.append(
                {
                    "training_seed": model,
                    "history_id": h,
                    "fit_root": settings["fit_roots"][h],
                    "evaluation_root": settings["evaluation_roots"][h],
                    "order": order,
                    "memories": [memory_snapshot(m) for m in memories[(model, h)]],
                }
            )
    if not same(snapshots, expected_snapshots):
        raise ValueError("memory snapshots differ from fitting outcomes")
    computed = summarize(
        metrics_rows,
        max_steps=config["max_steps"],
        bootstrap_seed=config["bootstrap_seed"],
        bootstrap_samples=config["bootstrap_samples"],
    )
    extra = {
        "split",
        "interpretation",
        "total_elapsed_seconds",
        "trace_sha256",
        "config_sha256",
    }
    if set(summary) != set(computed) | extra:
        raise ValueError("unexpected summary fields")
    if (
        not same({k: summary[k] for k in computed}, computed)
        or summary["interpretation"] != config["interpretation"]
    ):
        raise ValueError("summary differs from complete recomputed evidence")
    finite(summary["total_elapsed_seconds"], "elapsed seconds", 0)
    return {
        "status": "PASS",
        "split": split,
        "training_models": len(settings["training_seeds"]),
        "histories": len(orders),
        "evaluation_episodes": computed["design"]["evaluation_episodes"],
        "fit_episodes": computed["design"]["fit_episodes"],
        "memory_snapshots_checked": len(expected_snapshots),
        "actions_replayed": action_count,
        "scope": "Declared complete design, trace/counter/summary consistency, calibration metadata, fitting-memory reconstruction and optional simulator action replay. Does not retrain models, authenticate sources or prove policy choices.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--replay-actions", action="store_true")
    args = p.parse_args()
    print(json.dumps(verify(args.directory, replay=args.replay_actions), indent=2))
