"""Separate model/data/arithmetic reference for recorded monitoring comparisons.

Uses NumPy and PyTorch primitives, not the production model, simulator, detector,
collector, metric, summary, or verification functions. This is not external certification.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ACTIONS = (
    "move_to_object",
    "grasp",
    "move_to_target",
    "place",
    "reobserve",
    "clear_path",
)


def initial(seed):
    rng = np.random.default_rng(seed)
    obj = rng.uniform([0.18, 0.18], [0.42, 0.82]).astype(np.float32)
    target = rng.uniform([0.63, 0.18], [0.88, 0.82]).astype(np.float32)
    return np.array([0.1, 0.5, *obj, *target, 0.0, 1.0, 0.0], dtype=np.float32)


def nominal_data(seed, episodes):
    optional = np.random.default_rng(seed)
    inputs, actions, targets, ids = [], [], [], []
    for episode in range(episodes):
        envseed = int(
            np.random.SeedSequence([seed, episode, 0]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        state = initial(envseed)
        plan = ([4] if optional.random() < 0.25 else []) + [0, 1, 2, 3]
        for action in plan:
            following = state.copy()
            if action == 0:
                following[:2] = state[2:4]
            elif action == 1:
                following[6] = 1.0
                following[2:4] = state[:2]
            elif action == 2:
                following[:2] = state[4:6]
                following[2:4] = state[4:6]
            elif action == 3:
                following[6] = 0.0
                following[2:4] = state[:2]
            inputs.append(state)
            actions.append(action)
            targets.append(following)
            ids.append(episode)
            state = following
    return np.asarray(inputs), np.asarray(actions), np.asarray(targets), np.asarray(ids)


def forward(weights, states, actions):
    """Three independently composed dense layers and the declared activation."""
    dtype = weights["net.0.weight"].dtype
    x = torch.as_tensor(states, dtype=dtype)
    encoded = F.one_hot(torch.as_tensor(actions, dtype=torch.int64), 6).to(dtype)
    x = torch.cat((x, encoded), dim=-1)
    with torch.no_grad():
        for layer in (0, 2):
            x = F.linear(
                x, weights[f"net.{layer}.weight"], weights[f"net.{layer}.bias"]
            )
            x = F.silu(x)
        return F.linear(x, weights["net.4.weight"], weights["net.4.bias"])


def close(a, b, label, tolerance=2e-6):
    if (
        not math.isfinite(a)
        or not math.isfinite(b)
        or not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
    ):
        raise AssertionError(f"{label}: {a!r} != {b!r}")


def check(directory):
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    with gzip.open(root / "episodes.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    settings = config[summary["split"]]
    calibration_predictions = monitor_predictions = alarm_checks = 0
    maxima = {
        "residual_absolute_error": 0.0,
        "calibration_mse_absolute_error": 0.0,
        "calibration_threshold_absolute_error": 0.0,
    }
    # Calibrate directly from saved tensors and independently regenerated episode data.
    for seed in settings["training_seeds"]:
        folder = root / "models" / str(seed)
        payload = torch.load(
            folder / "world_model.pt", map_location="cpu", weights_only=True
        )
        weights = payload["state_dict"]
        metrics = json.loads((folder / "training_metrics.json").read_text())
        states, actions, targets, ids = nominal_data(
            seed, config["training"]["episodes"]
        )
        order = np.random.default_rng(seed).permutation(config["training"]["episodes"])
        boundary = max(1, int(0.8 * len(order)))
        train = sorted(order[:boundary].tolist())
        calibration = sorted(order[boundary:].tolist())
        assert (
            metrics["train_episode_ids"] == train
            and metrics["validation_episode_ids"] == calibration
        )
        mask = np.isin(ids, calibration)
        assert (
            int(mask.sum()) == metrics["validation_samples"]
            and int((~mask).sum()) == metrics["train_samples"]
        )
        predictions = forward(weights, states[mask], actions[mask])
        squared = (
            predictions - torch.as_tensor(targets[mask], dtype=predictions.dtype)
        ).square()
        mse = squared.mean().item()
        threshold = float(
            max(np.quantile(squared.mean(1).sqrt().numpy(), 0.995) * 1.35, 0.035)
        )
        close(mse, metrics["validation_mse"], "calibration MSE")
        close(threshold, metrics["residual_threshold"], "calibration threshold")
        assert payload["residual_threshold"] == metrics["residual_threshold"]
        maxima["calibration_mse_absolute_error"] = max(
            maxima["calibration_mse_absolute_error"],
            abs(mse - metrics["validation_mse"]),
        )
        maxima["calibration_threshold_absolute_error"] = max(
            maxima["calibration_threshold_absolute_error"],
            abs(threshold - metrics["residual_threshold"]),
        )
        calibration_predictions += int(mask.sum())
        # Batch the reference dense operations but compare every recorded decision.
        selected = [
            r
            for r in rows
            if r["training_seed"] == seed
            and (
                r["agent"] == "learned_nominal" or r["agent"].startswith("self_healing")
            )
        ]
        inputs = []
        action_ids = []
        observed = []
        records = []
        for row in selected:
            previous = initial(row["seed"])
            alarms = {
                e["step"]: e for e in row["event_log"] if e["event"] == "detection"
            }
            for item in row["event_log"]:
                if item["event"] != "transition":
                    continue
                inputs.append(previous)
                action_ids.append(ACTIONS.index(item["action"]))
                observed.append(item["observed_state"])
                records.append((item, alarms.get(item["step"])))
                previous = np.asarray(item["observed_state"], dtype=np.float32)
        output = (
            forward(weights, np.asarray(inputs), np.asarray(action_ids))
            .numpy()
            .astype(np.float64)
        )
        scores = np.sqrt(
            np.mean((output - np.asarray(observed, dtype=np.float64)) ** 2, axis=1)
        )
        for score, (transition, alarm) in zip(scores, records, strict=True):
            expected = (
                "execution_failure"
                if not transition["action_succeeded"]
                else "residual_only"
                if transition["action"] not in ("reobserve", "clear_path")
                and score > payload["residual_threshold"]
                else None
            )
            assert (alarm["trigger"] if alarm else None) == expected, (
                seed,
                transition["step"],
                expected,
                alarm,
            )
            if alarm is not None:
                close(
                    alarm["threshold"],
                    payload["residual_threshold"],
                    "declared alarm threshold",
                    1e-12,
                )
                close(float(score), alarm["residual"], "monitor residual")
                maxima["residual_absolute_error"] = max(
                    maxima["residual_absolute_error"],
                    abs(float(score) - alarm["residual"]),
                )
                alarm_checks += 1
        monitor_predictions += len(records)
    # Independently aggregate complete model/history cells and resample their axes.
    evaluation = [r for r in rows if r["phase"] == "evaluation"]
    cells = defaultdict(list)
    for r in evaluation:
        cells[
            (r["agent"], r["training_seed"], r["history_id"], r["perturbation"])
        ].append(r)
    models = sorted(settings["training_seeds"])
    histories = range(len(settings["evaluation_roots"]))
    rng = np.random.default_rng(config["bootstrap_seed"])
    mw = rng.multinomial(
        len(models), [1 / len(models)] * len(models), size=config["bootstrap_samples"]
    ) / len(models)
    hw = rng.multinomial(
        len(histories),
        [1 / len(histories)] * len(histories),
        size=config["bootstrap_samples"],
    ) / len(histories)
    contrasts = 0
    for contrast in summary["paired_contrasts"]:
        conditions = [
            c
            for c in summary["design"]["conditions"]
            if (c == "none") == (contrast["condition"] == "nominal")
        ]
        for label in ("capped_actions", "success_rate", "raw_steps"):

            def value(r):
                if label == "success_rate":
                    return r["success"]
                if label == "capped_actions" and not r["success"]:
                    return config["max_steps"]
                return r["steps"]

            effects = []
            for model in models:
                vector = []
                for history in histories:
                    per_condition = []
                    for condition in conditions:
                        a = cells[(contrast["method"], model, history, condition)]
                        b = cells[(contrast["reference"], model, history, condition)]
                        per_condition.append(
                            sum(value(r) for r in a) / len(a)
                            - sum(value(r) for r in b) / len(b)
                        )
                    vector.append(sum(per_condition) / len(per_condition))
                effects.append(vector)
            effects = np.asarray(effects)
            draws = np.sum((mw @ effects) * hw, axis=1)
            target = contrast["effects"][label]
            close(float(effects.mean()), target["estimate"], "paired effect", 1e-12)
            bounds = np.quantile(draws, [0.025, 0.975])
            for actual, expected in zip(bounds, target["ci95"], strict=True):
                close(float(actual), expected, "crossed-bootstrap interval", 1e-12)
            contrasts += 1
    return {
        "status": "PASS",
        "split": summary["split"],
        "training_models": len(models),
        "independent_histories": len(histories),
        "calibration_transition_predictions": calibration_predictions,
        "monitor_transition_predictions": monitor_predictions,
        "recorded_alarm_scores_checked": alarm_checks,
        "paired_estimates_and_intervals_checked": contrasts,
        "numerical_error_maxima": maxima,
        "scope": "Independent nominal-data construction, direct-tensor inference, detector decisions and crossed-bootstrap arithmetic. NumPy/PyTorch primitives remain shared. Does not certify real robots or enlarge the study sample.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(check(parser.parse_args().directory), indent=2))
