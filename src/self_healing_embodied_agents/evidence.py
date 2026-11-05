"""Typed research contracts. Local evidence consistency is not authentication."""

from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from .types import ActionKind


def integer(value, name, minimum=0, maximum=None):
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be a built-in integer in its permitted range")
    return value


def finite(value, name, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, excluding booleans")
    if (minimum is not None and value < minimum) or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON constant")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
        stack = [(value, 0)]
        count = 0
        while stack:
            item, depth = stack.pop()
            count += 1
            if depth > 64 or count > 1000000:
                raise ValueError("excessive evidence structure")
            if type(item) is dict:
                stack.extend((v, depth + 1) for v in item.values())
            elif type(item) is list:
                stack.extend((v, depth + 1) for v in item)
            elif type(item) is float:
                finite(item, "JSON number")
            elif type(item) is str:
                item.encode("utf-8")
        return value
    except (RecursionError, UnicodeError, OverflowError) as exc:
        raise ValueError("invalid JSON evidence") from exc


def read_json(path):
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 64 * 1024 * 1024
    ):
        raise ValueError("expected a bounded regular evidence file")
    return strict_json(path.read_bytes())


STUDY_FIELDS = {
    "schema_version",
    "study",
    "interpretation",
    "reference_commit",
    "hypotheses",
    "primary_outcomes",
    "secondary_outcomes",
    "sampling_unit",
    "replication_rationale",
    "precision_target_actions",
    "training",
    "development",
    "evaluation",
    "episodes_per_condition",
    "max_steps",
    "condition_schedule",
    "memory_protocol",
    "task_memory_reward",
    "methods",
    "bootstrap_seed",
    "bootstrap_samples",
}


def validate_study(config, methods, history_count=14):
    if type(config) is not dict or set(config) != STUDY_FIELDS:
        raise ValueError(
            "comparison configuration requires its complete supported fields"
        )
    integer(config["schema_version"], "schema_version", 1, 1)
    for key in ("episodes_per_condition", "max_steps", "bootstrap_samples"):
        integer(config[key], key, 1)
    integer(config["bootstrap_seed"], "bootstrap_seed", 0, 2**32 - 1)
    finite(config["precision_target_actions"], "precision_target_actions", 0)
    for key in (
        "study",
        "interpretation",
        "sampling_unit",
        "replication_rationale",
        "condition_schedule",
        "memory_protocol",
        "task_memory_reward",
    ):
        if type(config[key]) is not str or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if (
        type(config["reference_commit"]) is not str
        or re.fullmatch("[0-9a-f]{40}", config["reference_commit"]) is None
    ):
        raise ValueError("reference_commit must be a commit SHA")
    for key in ("hypotheses", "primary_outcomes", "secondary_outcomes"):
        if (
            type(config[key]) is not list
            or not config[key]
            or any(type(v) is not str or not v.strip() for v in config[key])
        ):
            raise ValueError(f"{key} must be a nonempty string list")
    if type(config["training"]) is not dict or set(config["training"]) != {
        "episodes",
        "epochs",
    }:
        raise ValueError("training must contain episodes and epochs")
    integer(config["training"]["episodes"], "training.episodes", 2)
    integer(config["training"]["epochs"], "training.epochs", 1)
    roots, models = set(), set()
    for split in ("development", "evaluation"):
        settings = config[split]
        if type(settings) is not dict or set(settings) != {
            "training_seeds",
            "fit_roots",
            "evaluation_roots",
        }:
            raise ValueError("invalid split fields")
        for key in settings:
            values = settings[key]
            if type(values) is not list or not values:
                raise ValueError("split seeds must be nonempty lists")
            for v in values:
                integer(v, key, 0, 2**32 - 1)
            if len(set(values)) != len(values):
                raise ValueError("split seeds must be distinct")
            used = models if key == "training_seeds" else roots
            if used.intersection(values):
                raise ValueError("study splits and phases must have disjoint seeds")
            used.update(values)
            if key != "training_seeds" and len(values) != history_count:
                raise ValueError(
                    "each phase needs the complete counterbalanced history set"
                )
    if type(config["methods"]) is not list or config["methods"] != list(methods):
        raise ValueError("configured methods differ from the implemented design")
    return deepcopy(config)


COUNTERS = (
    "steps",
    "interventions",
    "true_failures",
    "detections",
    "true_positive_detections",
    "false_positive_detections",
    "recovery_attempts",
    "recovery_successes",
)


def validate_episode(row, *, detection_window=1, complete=False):
    integer(detection_window, "detection_window")
    for name in COUNTERS:
        integer(row[name], name)
    integer(row["seed"], "seed", 0, 2**64 - 1)
    if type(row["success"]) not in (int, bool) or row["success"] not in (0, 1):
        raise ValueError("success must be zero or one")
    steps = row["steps"]
    for name in ("interventions", "detections", "true_failures", "recovery_attempts"):
        if row[name] > steps:
            raise ValueError(f"{name} exceeds executed steps")
    if row["recovery_successes"] > row["recovery_attempts"]:
        raise ValueError("recovery successes exceed attempts")
    trace = row["event_log"]
    if type(trace) is not list or any(type(e) is not dict for e in trace):
        raise ValueError("event_log must contain event objects")
    for event in trace:
        integer(event.get("step"), "event step", 0, steps)
        if type(event.get("event")) is not str:
            raise ValueError("event name must be a string")
        if "failure" in event and type(event["failure"]) is not bool:
            raise ValueError("failure marker must be boolean")
    transitions = [e for e in trace if e["event"] == "transition"]
    if len(transitions) != steps:
        raise ValueError("transition count must equal executed steps")
    if [e["step"] for e in transitions] != list(range(1, steps + 1)):
        raise ValueError("transition steps must be consecutive")
    for i, event in enumerate(transitions):
        if event.get("action") not in {a.value for a in ActionKind}:
            raise ValueError("unknown transition action")
        for field in ("action_succeeded", "task_success"):
            if type(event.get(field)) is not bool:
                raise ValueError(f"{field} must be boolean")
        if event["task_success"] and i != len(transitions) - 1:
            raise ValueError("execution continued after task success")
        if "observed_state" in event:
            values = event["observed_state"]
            if type(values) is not list or len(values) != 9:
                raise ValueError("invalid observed state")
            for v in values:
                finite(v, "observed state")
            if any(v not in (0.0, 1.0) for v in values[6:]):
                raise ValueError("invalid observed binary flag")
        elif complete:
            raise ValueError("complete evidence requires observed states")
    if bool(row["success"]) != (
        transitions[-1]["task_success"] if transitions else False
    ):
        raise ValueError("reported success contradicts final observation")
    onsets = {e["step"] for e in trace if e.get("failure")}
    if any(step < 1 for step in onsets):
        raise ValueError("failure onset must follow an action")
    alarms = [e for e in trace if e["event"] == "detection"]
    matched = set()
    for event in alarms:
        integer(event["step"], "alarm step", 1, steps)
        step = event.get("matched_failure_step")
        if step is not None:
            integer(step, "matched onset", 1, steps)
            if (
                step not in onsets
                or step in matched
                or not 0 <= event["step"] - step <= detection_window
            ):
                raise ValueError("invalid or repeated alarm onset match")
            matched.add(step)
        trigger = event.get("trigger")
        if trigger not in {
            "execution_failure",
            "residual_only",
            "postcondition_only",
            "analytical_only",
        }:
            raise ValueError("unknown detection trigger")
        actual_failure = not transitions[event["step"] - 1]["action_succeeded"]
        if (trigger == "execution_failure") != actual_failure:
            raise ValueError("alarm trigger contradicts action result")
        if "execution_failure" in event and (
            type(event["execution_failure"]) is not bool
            or event["execution_failure"] != actual_failure
        ):
            raise ValueError("incorrect execution failure flag")
        if event.get("residual") is not None:
            finite(event["residual"], "residual", 0)
        if event.get("threshold") is not None:
            finite(event["threshold"], "threshold", 0)
        if trigger in {"residual_only", "analytical_only"} and (
            complete or "residual" in event or "threshold" in event
        ):
            if (
                event.get("residual") is None
                or event.get("threshold") is None
                or event["residual"] <= event["threshold"]
            ):
                raise ValueError("residual alarm does not exceed threshold")
    for field, value in [
        ("true_failures", len(onsets)),
        ("detections", len(alarms)),
        ("true_positive_detections", len(matched)),
        ("false_positive_detections", len(alarms) - len(matched)),
    ]:
        if row[field] != value:
            raise ValueError(f"{field} disagrees with event log")
    for field, expected in [
        ("detection_precision", len(matched) / len(alarms) if alarms else None),
        ("detection_recall", len(matched) / len(onsets) if onsets else None),
    ]:
        value = row[field]
        if expected is None:
            if value is not None:
                raise ValueError("undefined detection ratio must be null")
        elif (
            type(value) not in (int, float)
            or isinstance(value, bool)
            or not math.isclose(finite(value, field, 0, 1), expected, abs_tol=1e-12)
        ):
            raise ValueError("detection ratio contradicts counters")
    if complete:
        if [e["step"] for e in trace] != sorted(e["step"] for e in trace):
            raise ValueError("complete trace events must follow execution order")
        active = None
        for event in trace:
            if event["event"] == "recovery_selected":
                if active is not None:
                    raise ValueError("new recovery has no outcome for its predecessor")
                actions = event.get("actions")
                if (
                    type(actions) is not list
                    or not actions
                    or any(a not in {v.value for v in ActionKind} for a in actions)
                ):
                    raise ValueError("invalid selected recovery actions")
                if type(event.get("candidate")) is not str or not event["candidate"]:
                    raise ValueError("invalid recovery identity")
                active = event
            elif event["event"] == "recovery_outcome":
                if active is None:
                    raise ValueError("recovery outcome has no selection")
                if row["agent"] != "reactive_replan" and any(
                    event.get(k) != active.get(k)
                    for k in ("candidate", "failure_class")
                ):
                    raise ValueError("recovery outcome belongs to another candidate")
                if event.get("reason") not in {
                    "completed",
                    "interrupted",
                    "budget_exhausted",
                    "execution_error",
                }:
                    raise ValueError("unknown recovery outcome reason")
                if type(event.get("success")) is not bool or event["success"] != (
                    event["reason"] == "completed"
                ):
                    raise ValueError("recovery outcome reason contradicts success")
                active = None
        if active is not None:
            raise ValueError("recovery selection lacks a final outcome")
        selections = [e for e in trace if e["event"] == "recovery_selected"]
        outcomes = [e for e in trace if e["event"] == "recovery_outcome"]
        if len(selections) != row["recovery_attempts"] or len(outcomes) != len(
            selections
        ):
            raise ValueError("recovery attempt/outcome accounting mismatch")
        if (
            any(type(e.get("success")) is not bool for e in outcomes)
            or sum(e["success"] for e in outcomes) != row["recovery_successes"]
        ):
            raise ValueError("recovery success accounting mismatch")
        if row["agent"] not in {
            "open_loop",
            "reactive_replan",
            "always_replan",
        } and row["interventions"] != len(alarms):
            raise ValueError("intervention count differs from detection count")
        if len({e["step"] for e in alarms}) != len(alarms):
            raise ValueError("duplicate alarms on one action")
    return row
