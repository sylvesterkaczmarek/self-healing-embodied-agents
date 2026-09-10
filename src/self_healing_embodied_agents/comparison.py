"""Paired comparisons for the crossed model and recovery-history pilot."""

from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np

from .types import EpisodeResult


METHODS = (
    "always_replan", "failure_nominal", "postcondition_nominal",
    "analytical_nominal", "learned_nominal", "self_healing",
    "self_healing_memory", "self_healing_task_memory",
    "self_healing_memory_frozen", "self_healing_task_memory_frozen",
)
CONTRASTS = (
    ("learned_nominal", "failure_nominal"),
    ("postcondition_nominal", "failure_nominal"),
    ("analytical_nominal", "failure_nominal"),
    ("learned_nominal", "postcondition_nominal"),
    ("self_healing", "learned_nominal"),
    ("self_healing", "always_replan"),
    *((method, "self_healing") for method in METHODS[6:]),
)
TRIGGERS = ("execution_failure", "residual_only", "postcondition_only", "analytical_only")
COSTS = (
    "failed_actions", "reobserve_actions", "clear_path_actions", "nominal_planner_calls",
    "ranked_recovery_selections", *(f"{trigger}_alarms" for trigger in TRIGGERS),
)


def episode_metrics(result: EpisodeResult | dict) -> dict:
    """Count actions and alarms from traces, retaining the historical onset window.

    Unmatched alarms are temporal matching outcomes, not evidence that an
    intervention was unnecessary. Continuous replanning does not itself count
    as detection. Only alarms matched to an onset have a detection delay.
    """
    row = ({**result.as_dict(), "event_log": result.event_log}
           if isinstance(result, EpisodeResult) else result)
    trace = row["event_log"]
    transitions = [event for event in trace if event.get("event") == "transition"]
    alarms = [event for event in trace if event.get("event") == "detection"]
    onsets = {event["step"] for event in trace if event.get("failure")}
    if len(transitions) != row["steps"]:
        raise ValueError("transition count must equal executed steps")
    if [event["step"] for event in transitions] != list(range(1, row["steps"] + 1)):
        raise ValueError("transition steps must be consecutive")
    if any(type(event["action_succeeded"]) is not bool for event in transitions):
        raise ValueError("action_succeeded must be boolean")
    matched = [event for event in alarms if event.get("matched_failure_step") is not None]
    matched_steps = [event["matched_failure_step"] for event in matched]
    if len(set(matched_steps)) != len(matched_steps) or not set(matched_steps) <= onsets:
        raise ValueError("alarms must match distinct recorded failure onset steps")
    delays = [event["step"] - event["matched_failure_step"] for event in matched]
    if any(delay < 0 for delay in delays):
        raise ValueError("an alarm cannot precede its matched failure onset")
    for field, value in (
        ("true_failures", len(onsets)), ("detections", len(alarms)),
        ("true_positive_detections", len(matched)),
        ("false_positive_detections", len(alarms) - len(matched)),
    ):
        if row[field] != value:
            raise ValueError(f"{field} disagrees with event log")
    counts = {f"{trigger}_alarms": 0 for trigger in TRIGGERS}
    for event in alarms:
        trigger = event.get("trigger")
        if trigger not in TRIGGERS:
            raise ValueError(f"unknown detection trigger: {trigger!r}")
        counts[f"{trigger}_alarms"] += 1
    return {
        "failed_actions": sum(not event["action_succeeded"] for event in transitions),
        "reobserve_actions": sum(event["action"] == "reobserve" for event in transitions),
        "clear_path_actions": sum(event["action"] == "clear_path" for event in transitions),
        "nominal_planner_calls": sum(event.get("event") == "planning" for event in trace),
        "ranked_recovery_selections": sum(
            event.get("event") == "recovery_selected" and event.get("recovery_policy") == "ranked"
            for event in trace
        ),
        **counts,
        "matched_onset_alarms": len(matched),
        "unmatched_onset_alarms": len(alarms) - len(matched),
        "matched_detection_delays": delays,
    }


def _validate_rows(rows: list[dict], max_steps: int) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("rows must contain evaluation episodes")
    evaluation = []
    paired = defaultdict(dict)
    histories = {}
    identities = set()
    for row in rows:
        if row.get("phase") not in {"fit", "evaluation"}:
            raise ValueError("phase must be fit or evaluation")
        for field in ("training_seed", "history_id", "root_seed", "repetition", "seed", "steps"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if type(row.get("success")) not in (int, bool) or row["success"] not in (0, 1):
            raise ValueError("success must be zero or one")
        if row["steps"] > max_steps:
            raise ValueError("executed steps exceed max_steps")
        if row.get("agent") not in METHODS:
            raise ValueError("unknown comparison method")
        order = row.get("order")
        if (not isinstance(order, list) or not order
                or any(not isinstance(value, str) for value in order)
                or len(order) != len(set(order)) or row.get("perturbation") not in order):
            raise ValueError("order must list distinct conditions including this perturbation")
        key = (row["training_seed"], row["history_id"], row["perturbation"], row["repetition"])
        identity = (row["phase"], *key, row["agent"])
        if identity in identities:
            raise ValueError(f"duplicate episode: {identity}")
        identities.add(identity)
        metrics = episode_metrics(row)
        if row["phase"] != "evaluation":
            continue
        history = row["history_id"]
        coordinates = (row["root_seed"], tuple(order))
        if history in histories and histories[history] != coordinates:
            raise ValueError("a history must share its root seed and order across all models and methods")
        histories[history] = coordinates
        paired[key][row["agent"]] = row
        evaluation.append({**row, **metrics,
                           "capped_actions": row["steps"] if row["success"] else max_steps})
    if not evaluation:
        raise ValueError("rows must contain evaluation episodes")
    for key, methods in paired.items():
        if set(methods) != set(METHODS):
            raise ValueError(f"incomplete paired methods at {key}")
        if len({row["seed"] for row in methods.values()}) != 1:
            raise ValueError("paired methods must share an episode seed")
    models = sorted({row["training_seed"] for row in evaluation})
    history_ids = sorted(histories)
    cases = {(key[2], key[3]) for key in paired}
    expected = {(model, history, condition, rep)
                for model, history, (condition, rep) in product(models, history_ids, cases)}
    if set(paired) != expected:
        raise ValueError("incomplete crossed model/history/condition/repetition design")
    conditions = sorted({condition for condition, _ in cases})
    if any(set(order) != set(conditions) for _, order in histories.values()):
        raise ValueError("evaluation condition coverage disagrees with history order")
    seeds = {}
    for key, methods in paired.items():
        case = key[1:]
        seed = next(iter(methods.values()))["seed"]
        if case in seeds and seeds[case] != seed:
            raise ValueError("a history case must share its episode seed across training models")
        seeds[case] = seed
    return evaluation, {"training_seeds": models, "history_ids": history_ids,
                        "conditions": conditions}


def _group_summary(agent: str, condition: str, rows: list[dict], max_steps: int) -> dict:
    n = len(rows)
    successes = sum(row["success"] for row in rows)
    matched = sum(row["matched_onset_alarms"] for row in rows)
    unmatched = sum(row["unmatched_onset_alarms"] for row in rows)
    onsets = sum(row["true_failures"] for row in rows)
    delays = [delay for row in rows for delay in row["matched_detection_delays"]]
    totals = {cost: sum(row[cost] for row in rows) for cost in COSTS}
    return {
        "agent": agent, "condition": condition, "episodes": n,
        "training_models": len({row["training_seed"] for row in rows}),
        "histories": len({row["history_id"] for row in rows}),
        "successes": successes, "failures": n - successes,
        "success_rate": successes / n,
        "mean_steps": sum(row["steps"] for row in rows) / n,
        "mean_capped_actions": sum(row["capped_actions"] for row in rows) / n,
        "cost_totals": totals, "cost_means": {cost: value / n for cost, value in totals.items()},
        "detection": {
            "failure_onsets": onsets, "matched_onset_alarms": matched,
            "unmatched_onset_alarms": unmatched,
            "onset_precision": matched / (matched + unmatched) if matched + unmatched else None,
            "onset_recall": matched / onsets if onsets else None,
            "matched_delay_count": len(delays),
            "matched_delay_mean_steps": sum(delays) / len(delays) if delays else None,
        },
        "completion_curve": [
            {"budget": budget,
             "completed": sum(row["success"] and row["steps"] <= budget for row in rows),
             "success_rate": sum(row["success"] and row["steps"] <= budget for row in rows) / n}
            for budget in range(1, max_steps + 1)
        ],
    }


def summarize(rows: list[dict], *, max_steps: int = 24, bootstrap_seed: int = 0,
              bootstrap_samples: int = 2000) -> dict:
    """Summarize complete paired evaluation histories with a two-way bootstrap.

    Within each model/history cell, average repetitions within conditions and
    then conditions. Resample model seeds and whole histories independently.
    These percentile intervals describe this crossed pilot, including shared
    model and history variation; episodes are not independent replicates.
    """
    for field, value, minimum in (("max_steps", max_steps, 1),
                                  ("bootstrap_seed", bootstrap_seed, 0),
                                  ("bootstrap_samples", bootstrap_samples, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    evaluation, design = _validate_rows(rows, max_steps)
    groups = defaultdict(list)
    pools = defaultdict(list)
    cells = defaultdict(list)
    for row in evaluation:
        groups[row["agent"], row["perturbation"]].append(row)
        pool = "nominal" if row["perturbation"] == "none" else "faults"
        pools[row["agent"], pool].append(row)
        cells[row["agent"], row["training_seed"], row["history_id"], row["perturbation"]].append(row)
    models, histories = design["training_seeds"], design["history_ids"]
    rng = np.random.default_rng(bootstrap_seed)
    model_weights = rng.multinomial(len(models), np.full(len(models), 1 / len(models)),
                                   size=bootstrap_samples) / len(models)
    history_weights = rng.multinomial(len(histories), np.full(len(histories), 1 / len(histories)),
                                     size=bootstrap_samples) / len(histories)
    contrasts = []
    for pool in ("faults", "nominal"):
        conditions = [condition for condition in design["conditions"]
                      if (condition == "none") == (pool == "nominal")]
        if not conditions:
            continue
        for method, reference in CONTRASTS:
            effects = {}
            for field, label in (("capped_actions", "capped_actions"),
                                 ("success", "success_rate"), ("steps", "raw_steps")):
                differences = np.asarray([
                    [np.mean([
                        np.mean([row[field] for row in cells[method, model, history, condition]])
                        - np.mean([row[field] for row in cells[reference, model, history, condition]])
                        for condition in conditions
                    ]) for history in histories] for model in models
                ])
                draws = np.einsum("bi,ij,bj->b", model_weights, differences, history_weights)
                effects[label] = {"estimate": float(differences.mean()),
                                  "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}
            contrasts.append({"method": method, "reference": reference, "condition": pool,
                              "paired_episodes": len(pools[method, pool]), "effects": effects})
    return {
        "design": {
            **design, "methods": list(METHODS), "max_steps": max_steps,
            "evaluation_episodes": len(evaluation), "fit_episodes": len(rows) - len(evaluation),
            "bootstrap_seed": bootstrap_seed, "bootstrap_samples": bootstrap_samples,
            "uncertainty": "Exploratory pilot 95% percentile intervals from independent resampling "
                           "of training-model seeds and complete histories; no independent episode assumption.",
            "effect_direction": "Method minus reference; negative capped actions and positive success favour method.",
            "cost_definition": "Capped actions equal executed steps on success and max_steps on failure.",
            "curve_interpretation": "Descriptive completion times under the fixed max_steps policy; "
                                    "budgets are not separately executed policies.",
            "memory_protocols": "Cold online and fitted frozen memory differ in both prior experience and updating.",
            "baseline_replication": "Repeated deterministic baseline results across model seeds are shared observations.",
        },
        "by_condition": [_group_summary(agent, condition, group, max_steps)
                         for (agent, condition), group in sorted(groups.items())],
        "pooled": [_group_summary(agent, condition, group, max_steps)
                   for (agent, condition), group in sorted(pools.items())],
        "paired_contrasts": contrasts,
    }
