from __future__ import annotations

import copy
import json

import pytest

from self_healing_embodied_agents.comparison import METHODS, episode_metrics, summarize
from self_healing_embodied_agents.types import EpisodeResult


def result(*, steps=4, success=True, agent="self_healing", **kwargs):
    trace = [{"event": "planning", "step": 0}]
    trace.extend({"event": "transition", "step": step, "action": "move_to_object",
                  "action_succeeded": True, "task_success": success and step == steps}
                 for step in range(1, steps + 1))
    return EpisodeResult(success=success, steps=steps, perturbation="none", agent=agent,
                         seed=1, event_log=trace, **kwargs)


def crossed_rows(*, repetitions=1):
    rows = []
    for mi, model in enumerate((11, 13)):
        for hi, history in enumerate((101, 102)):
            for condition in ("none", "grasp_slip"):
                for rep in range(repetitions):
                    for method in METHODS:
                        # The paired effect has independent model and history components.
                        steps = 4 + (4 * mi + 2 * hi if method == "self_healing" else 0)
                        episode = result(steps=steps, agent=method)
                        rows.append({**episode.as_dict(), "event_log": episode.event_log,
                                     "phase": "evaluation", "training_seed": model,
                                     "history_id": history, "root_seed": history,
                                     "seed": history * 100 + rep, "repetition": rep,
                                     "perturbation": condition, "order": ["none", "grasp_slip"]})
    return rows


def contrast(summary, method="self_healing", reference="always_replan", condition="faults"):
    return next(row for row in summary["paired_contrasts"]
                if (row["method"], row["reference"], row["condition"])
                == (method, reference, condition))


def test_trace_costs_and_historical_alarm_matching_remain_distinct():
    episode = result(steps=4, true_failures=1, detections=2,
                     true_positive_detections=1, false_positive_detections=1)
    episode.event_log[2].update(action="grasp", action_succeeded=False)
    episode.event_log[3]["action"] = "reobserve"
    episode.event_log[4]["action"] = "clear_path"
    episode.event_log += [
        {"event": "grasp_slip", "step": 1, "failure": True},
        {"event": "detection", "step": 2, "trigger": "execution_failure", "matched_failure_step": 1},
        {"event": "detection", "step": 4, "trigger": "residual_only", "matched_failure_step": None},
        {"event": "planning", "step": 2},
        {"event": "recovery_selected", "step": 2, "recovery_policy": "ranked"},
    ]
    metrics = episode_metrics(episode)
    assert metrics == {
        "failed_actions": 1, "reobserve_actions": 1, "clear_path_actions": 1,
        "nominal_planner_calls": 2, "ranked_recovery_selections": 1,
        "execution_failure_alarms": 1, "residual_only_alarms": 1,
        "postcondition_only_alarms": 0, "analytical_only_alarms": 0,
        "matched_onset_alarms": 1, "unmatched_onset_alarms": 1,
        "matched_detection_delays": [1],
    }
    assert all(type(value) is int for key, value in metrics.items() if key != "matched_detection_delays")


def test_always_replanning_is_a_compute_cost_not_an_alarm():
    episode = result(agent="always_replan")
    episode.event_log += [{"event": "planning", "step": step} for step in range(1, 4)]
    episode.event_log[2]["action_succeeded"] = False
    metrics = episode_metrics(episode)
    assert metrics["nominal_planner_calls"] == 4
    assert metrics["failed_actions"] == 1
    assert metrics["execution_failure_alarms"] == metrics["matched_onset_alarms"] == 0


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row.update(steps=5), "transition count"),
    (lambda row: row.update(true_failures=1), "true_failures disagrees"),
    (lambda row: row["event_log"][1].update(action_succeeded=1), "must be boolean"),
    (lambda row: row["event_log"][1].update(step=0), "must be consecutive"),
])
def test_trace_inconsistencies_are_rejected(mutation, match):
    episode = result()
    row = {**episode.as_dict(), "event_log": episode.event_log}
    mutation(row)
    with pytest.raises(ValueError, match=match):
        episode_metrics(row)


def test_crossed_bootstrap_recovers_known_effect_and_is_deterministic():
    rows = crossed_rows()
    summary = summarize(rows, bootstrap_seed=97, bootstrap_samples=2000)
    effects = contrast(summary)["effects"]
    assert effects["capped_actions"] == {"estimate": 3.0, "ci95": [0.0, 6.0]}
    assert effects["success_rate"] == {"estimate": 0.0, "ci95": [0.0, 0.0]}
    assert summary == summarize(list(reversed(rows)), bootstrap_seed=97, bootstrap_samples=2000)
    # Repeating observations inside a model/history cell adds no independent replication.
    repeated = summarize(crossed_rows(repetitions=3), bootstrap_seed=97, bootstrap_samples=2000)
    assert contrast(repeated)["effects"] == effects
    assert repeated["design"]["history_ids"] == [101, 102]
    json.dumps(summary, allow_nan=False)


def test_failed_early_stops_receive_full_cost_and_undefined_detection_is_null():
    rows = crossed_rows()
    for row in rows:
        if row["agent"] == "self_healing":
            episode = result(steps=1, success=False)
            row.update(steps=1, success=0, event_log=episode.event_log)
    summary = summarize(rows, bootstrap_samples=100)
    effects = contrast(summary)["effects"]
    assert effects["capped_actions"]["estimate"] == 20
    assert effects["raw_steps"]["estimate"] == -3
    assert effects["success_rate"]["estimate"] == -1
    group = next(row for row in summary["pooled"]
                 if row["agent"] == "self_healing" and row["condition"] == "faults")
    assert group["failures"] == group["episodes"] == 4
    assert group["mean_steps"] == 1 and group["mean_capped_actions"] == 24
    assert all(point["completed"] == 0 for point in group["completion_curve"])
    assert group["detection"]["onset_precision"] is None
    assert group["detection"]["onset_recall"] is None
    assert group["detection"]["matched_delay_mean_steps"] is None


def test_fit_rows_do_not_enter_held_out_effects():
    rows = crossed_rows()
    fitted = copy.deepcopy(next(row for row in rows if row["agent"] == "self_healing_memory_frozen"))
    fitted["phase"] = "fit"
    summary = summarize([*rows, fitted], bootstrap_samples=100)
    assert summary["design"]["fit_episodes"] == 1
    assert summary["paired_contrasts"] == summarize(rows, bootstrap_samples=100)["paired_contrasts"]


@pytest.mark.parametrize("case,match", [
    ("duplicate", "duplicate"), ("missing_method", "incomplete paired"),
    ("missing_cell", "incomplete crossed"), ("seed", "share an episode seed"),
    ("model_seed", "across training models"), ("order", "share its root seed and order"),
])
def test_pairing_errors_cannot_silently_drop_data(case, match):
    rows = crossed_rows()
    if case == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif case == "missing_method":
        rows.pop()
    elif case == "missing_cell":
        rows = [row for row in rows if (row["training_seed"], row["history_id"]) != (11, 101)]
    elif case == "seed":
        rows[0]["seed"] += 1
    elif case == "model_seed":
        for row in rows:
            if row["training_seed"] == 13:
                row["seed"] += 1
    elif case == "order":
        rows[0]["order"].reverse()
    with pytest.raises(ValueError, match=match):
        summarize(rows, bootstrap_samples=100)


@pytest.mark.parametrize("settings", [{"max_steps": 0}, {"bootstrap_samples": 0},
                                      {"bootstrap_seed": -1}, {"max_steps": True}])
def test_invalid_analysis_settings_fail(settings):
    with pytest.raises(ValueError):
        summarize(crossed_rows(), **settings)
