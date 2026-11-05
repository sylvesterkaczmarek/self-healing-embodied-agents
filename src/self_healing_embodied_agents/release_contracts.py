"""Complete reference-execution and declared calibration/provenance contracts."""

from __future__ import annotations
from dataclasses import asdict
import re
import numpy as np
from .evidence import integer
from .types import ActionKind
from .training import TrainingMetrics

ACTIONS = {kind.value for kind in ActionKind}
EVENT_FIELDS = {
    "planning": {"event", "step", "actions"},
    "transition": {
        "event",
        "step",
        "action",
        "action_succeeded",
        "observed_state",
        "task_success",
    },
    "detection": {
        "event",
        "step",
        "execution_failure",
        "matched_failure_step",
        "residual",
        "threshold",
        "trigger",
    },
    "recovery_selected": {
        "event",
        "step",
        "actions",
        "candidate",
        "failure_class",
        "recovery_policy",
        "remaining_steps",
    },
    "recovery_outcome": {
        "event",
        "step",
        "candidate",
        "failure_class",
        "reason",
        "success",
    },
    "memory_update": {
        "event",
        "step",
        "selected_step",
        "failure_class",
        "candidate",
        "task_success",
        "remaining_actions",
        "reward",
    },
}
for name in ("grasp_slip", "object_displacement", "stale_observation"):
    EVENT_FIELDS[name] = {"event", "step", "failure", "dx", "dy"}
for name in ("reobserve", "path_cleared", "blocked_path", "transient_occlusion"):
    EVENT_FIELDS[name] = {"event", "step", "failure"}


def validate_execution(row):
    """Check recorded action execution, not whether a planner chose the best plan."""
    method = row["agent"]
    legacy = method == "reactive_replan"
    if method in ("open_loop", "always_replan", "reactive_replan"):
        monitor = None
    elif method.startswith("self_healing"):
        monitor = "learned"
    else:
        monitor = method.split("_")[0]
    allowed = {"execution_failure"}
    residual_trigger = {
        "learned": "residual_only",
        "postcondition": "postcondition_only",
        "analytical": "analytical_only",
    }.get(monitor)
    if residual_trigger is not None:
        allowed.add(residual_trigger)
    active = None
    pending = []
    used = 0
    transition = None
    alarm_step = None
    selected_step = None
    for event in row["event_log"]:
        name = event["event"]
        fields = EVENT_FIELDS.get(name)
        if fields is None:
            raise ValueError("unknown complete-trace event")
        if legacy and name == "recovery_selected":
            fields = {"event", "step", "candidate", "actions"}
        if legacy and name == "recovery_outcome":
            fields = {"event", "step", "success", "reason"}
        if set(event) != fields:
            raise ValueError("unexpected or missing complete-trace event fields")
        if name in ("planning", "recovery_selected"):
            if type(event["actions"]) is not list or any(
                type(a) is not str or a not in ACTIONS for a in event["actions"]
            ):
                raise ValueError("invalid recorded plan")
        if name == "transition":
            transition = event
            if active is not None:
                if (
                    event["step"] <= active["step"]
                    or not pending
                    or event["action"] != pending.pop(0)
                ):
                    raise ValueError(
                        "recovery action differs from the selected sequence"
                    )
                used += 1
        elif name == "detection":
            if (
                transition is None
                or event["step"] != transition["step"]
                or monitor is None
                or event["trigger"] not in allowed
            ):
                raise ValueError(
                    "alarm is not attributable to this observation and detector"
                )
            if event["trigger"] in ("residual_only", "analytical_only") and transition[
                "action"
            ] in ("reobserve", "clear_path"):
                raise ValueError(
                    "repair actions cannot generate residual-only alarms in this protocol"
                )
            alarm_step = event["step"]
        elif name == "recovery_selected":
            if (
                active is not None
                or transition is None
                or event["step"] != transition["step"]
                or transition["task_success"]
                or selected_step == event["step"]
                or (not legacy and alarm_step != event["step"])
                or (legacy and transition["action_succeeded"])
            ):
                raise ValueError(
                    "recovery selection lacks its preceding failed observation/alarm"
                )
            if not event["actions"]:
                raise ValueError("recovery plan must contain actions")
            if not legacy:
                integer(event["remaining_steps"], "remaining_steps", 1)
                if event["recovery_policy"] not in ("nominal", "ranked"):
                    raise ValueError("unknown recorded recovery policy")
                expected_policy = (
                    "ranked"
                    if method.startswith("self_healing") or method.endswith("_ranked")
                    else "nominal"
                )
                if event["recovery_policy"] != expected_policy:
                    raise ValueError("recovery policy contradicts method")
            active = event
            pending = list(event["actions"])
            used = 0
            selected_step = event["step"]
        elif name == "recovery_outcome":
            if (
                active is None
                or transition is None
                or event["step"] != transition["step"]
                or used < 1
            ):
                raise ValueError("recovery outcome precedes executed recovery actions")
            reason = event["reason"]
            if reason == "completed" and pending and not transition["task_success"]:
                raise ValueError("recovery claims completion with unexecuted actions")
            if reason == "interrupted" and (
                transition["task_success"]
                or (
                    transition["action_succeeded"]
                    if legacy
                    else alarm_step != event["step"]
                )
            ):
                raise ValueError(
                    "interruption lacks a corresponding failed observation/alarm"
                )
            if reason in ("budget_exhausted", "execution_error") and (
                event["step"] != row["steps"] or transition["task_success"]
            ):
                raise ValueError(
                    "terminal recovery failure precedes the terminal record"
                )
            active = None
            pending = []


def validate_calibration_protocol(metrics, training):
    """Recompute the fixed collector's split and counts without fitting a model."""
    defaults = asdict(TrainingMetrics(0, 1, 1, 0.0, 0.1))
    for field in (
        "split_unit",
        "validation_role",
        "environment_seed_rule",
        "learning_rate",
        "calibration_quantile",
        "calibration_multiplier",
        "calibration_floor",
    ):
        value = getattr(metrics, field)
        if type(value) is not type(defaults[field]) or value != defaults[field]:
            raise ValueError(
                "calibration metadata differs from the declared fixed training protocol"
            )
    episodes = training["episodes"]
    permutation = np.random.default_rng(metrics.seed).permutation(episodes)
    train = sorted(permutation[: max(1, int(episodes * 0.8))].tolist())
    calibration = sorted(permutation[max(1, int(episodes * 0.8)) :].tolist())
    lengths = 4 + (np.random.default_rng(metrics.seed).random(episodes) < 0.25).astype(
        np.int64
    )
    if (
        metrics.train_episode_ids != train
        or metrics.validation_episode_ids != calibration
        or metrics.train_samples != int(lengths[train].sum())
        or metrics.validation_samples != int(lengths[calibration].sum())
    ):
        raise ValueError(
            "calibration partition or sample count differs from deterministic collection"
        )


REQUIRED_SOURCE = {
    f"src/self_healing_embodied_agents/{name}.py"
    for name in (
        "__init__",
        "agents",
        "benchmark",
        "comparison",
        "env",
        "plotting",
        "provenance",
        "recovery",
        "training",
        "types",
        "world_model",
    )
}
REQUIRED_SOURCE |= {
    f"scripts/{name}.py"
    for name in (
        "__init__",
        "_common",
        "reproduce",
        "run_benchmark",
        "run_comparison",
        "smoke_demo",
        "train_world_model",
    )
}
PROVENANCE_FIELDS = {
    "schema_version",
    "effective_config",
    "model_origin",
    "training_metrics",
    "checkpoint_sha256",
    "source_sha256",
    "runtime",
    "evaluation_seeds",
    "evaluation_seed_rule",
    "paired_across",
    "detection_aggregation",
    "memory_scope",
    "ordered_conditions",
    "episode_order",
    "study_split",
    "reference_commit",
    "fit_roots",
    "config_sha256",
}


def validate_provenance(value):
    if type(value) is not dict or set(value) != PROVENANCE_FIELDS:
        raise ValueError("incomplete or unsupported provenance fields")
    integer(value["schema_version"], "provenance schema", 1, 1)
    exact = {
        "model_origin": "trained",
        "memory_scope": "independent model/history; cold online or separately fitted frozen",
        "episode_order": "training seed, history, phase (fit/evaluation), condition, repetition, method",
        "evaluation_seed_rule": "SeedSequence([root_seed, repetition, 0x4556414C]) generates one uint64",
        "paired_across": "agents and perturbations",
        "detection_aggregation": "pooled true-positive counts / pooled detections or injected failure steps",
    }
    if any(value[k] != expected for k, expected in exact.items()):
        raise ValueError("provenance protocol contradicts the comparison")
    sources = value["source_sha256"]
    if type(sources) is not dict or not REQUIRED_SOURCE <= sources.keys():
        raise ValueError("source fingerprints omit required executed modules")
    for name, digest in sources.items():
        if (
            type(name) is not str
            or re.fullmatch(
                r"(?:src/self_healing_embodied_agents|scripts)/[A-Za-z0-9_]+\.py", name
            )
            is None
            or type(digest) is not str
            or re.fullmatch("[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("invalid source path or fingerprint")
    runtime = value["runtime"]
    if type(runtime) is not dict or set(runtime) != {
        "python",
        "platform",
        "dependencies",
        "device",
        "torch_num_threads",
        "torch_num_interop_threads",
        "deterministic_algorithms",
        "deterministic_warn_only",
    }:
        raise ValueError("invalid recorded runtime")
    for key in ("python", "platform"):
        if type(runtime[key]) is not str or not runtime[key]:
            raise ValueError("invalid runtime identity")
    if runtime["device"] != "cpu":
        raise ValueError("comparison is a CPU protocol")
    for key in ("torch_num_threads", "torch_num_interop_threads"):
        integer(runtime[key], key, 1)
    for key in ("deterministic_algorithms", "deterministic_warn_only"):
        if type(runtime[key]) is not bool:
            raise ValueError("runtime determinism fields must be boolean")
    deps = runtime["dependencies"]
    if (
        type(deps) is not dict
        or set(deps) != {"self-healing-embodied-agents", "numpy", "torch"}
        or any(type(deps[k]) is not str or not deps[k] for k in ("numpy", "torch"))
        or (
            deps["self-healing-embodied-agents"] is not None
            and (
                type(deps["self-healing-embodied-agents"]) is not str
                or not deps["self-healing-embodied-agents"]
            )
        )
    ):
        raise ValueError("invalid dependency provenance")
