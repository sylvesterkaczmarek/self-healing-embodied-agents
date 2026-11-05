"""Independent release probes, written against the unchanged implementation candidate."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import pytest
from self_healing_embodied_agents.agents import SelfHealingAgent
from self_healing_embodied_agents.env import TabletopManipulationEnv
from self_healing_embodied_agents.evidence import validate_episode
from self_healing_embodied_agents.training import load_bundle
from scripts.run_comparison import load_study, run_study
from scripts.verify_comparison import verify

REPO = Path(__file__).resolve().parents[1]


def episode():
    r = SelfHealingAgent(detector="failure", recovery="nominal").run_episode(
        TabletopManipulationEnv(seed=17, perturbation="blocked_path")
    )
    return {**r.as_dict(), "event_log": deepcopy(r.event_log)}


@pytest.mark.parametrize(
    "damage",
    [
        "unexecuted-plan",
        "wrong-action",
        "early-completion",
        "extra-event",
        "extra-event-field",
        "false-interruption",
        "selection-before-alarm",
        "alarm-before-observation",
        "recovery-after-terminal",
    ],
)
def test_complete_recovery_requires_actual_execution(damage):
    r = episode()
    trace = r["event_log"]
    selection = next(e for e in trace if e["event"] == "recovery_selected")
    outcome = next(e for e in trace if e["event"] == "recovery_outcome")
    if damage == "unexecuted-plan":
        selection["actions"].append("grasp")
        r["success"] = 0
        trace[-2]["task_success"] = (
            False if trace[-2]["event"] == "transition" else False
        )
    elif damage == "wrong-action":
        selection["actions"][0] = "grasp"
    elif damage == "early-completion":
        trace.remove(outcome)
        outcome["step"] = selection["step"]
        trace.insert(trace.index(selection) + 1, outcome)
    elif damage == "extra-event":
        trace.append({"event": "invented_verified_success", "step": r["steps"]})
    elif damage == "extra-event-field":
        selection["approval_granted"] = True
    elif damage == "false-interruption":
        outcome.update(success=False, reason="interrupted")
        r["recovery_successes"] = 0
    elif damage == "selection-before-alarm":
        trace.remove(selection)
        alarm = next(e for e in trace if e["event"] == "detection")
        trace.insert(trace.index(alarm), selection)
    elif damage == "alarm-before-observation":
        alarm = next(e for e in trace if e["event"] == "detection")
        trace.remove(alarm)
        transition = next(
            e
            for e in trace
            if e["event"] == "transition" and e["step"] == alarm["step"]
        )
        trace.insert(trace.index(transition), alarm)
    else:
        selection.update(step=r["steps"])
        trace.remove(selection)
        trace.insert(len(trace) - 1, selection)
    with pytest.raises(ValueError):
        validate_episode(r, complete=True)


def test_learned_method_cannot_claim_other_monitor_alarm():
    bundle = load_bundle(REPO / "artifacts/world_model.pt")
    r = SelfHealingAgent(bundle).run_episode(
        TabletopManipulationEnv(seed=17, perturbation="grasp_slip")
    )
    row = {**r.as_dict(), "event_log": deepcopy(r.event_log)}
    alarm = next(
        e
        for e in row["event_log"]
        if e["event"] == "detection" and e["trigger"] == "residual_only"
    )
    alarm["trigger"] = "postcondition_only"
    with pytest.raises(ValueError):
        validate_episode(row, complete=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    cfg = load_study(REPO / "configs/comparison.json")
    cfg["development"]["training_seeds"] = [11]
    cfg["training"] = {"episodes": 12, "epochs": 2}
    cfg.update(episodes_per_condition=1, bootstrap_samples=10)
    output = tmp_path_factory.mktemp("release-study")
    run_study(cfg, "development", output)
    return output


def get(path):
    return json.loads(path.read_text())


def put(path, obj):
    path.write_text(json.dumps(obj))


@pytest.mark.parametrize(
    "damage",
    [
        "split-unit",
        "seed-rule",
        "sample-count",
        "partition-swap",
        "learning-rate",
        "calibration-quantile",
        "calibration-multiplier",
        "calibration-floor",
    ],
)
def test_rebound_calibration_metadata_matches_executed_protocol(
    study, tmp_path, damage
):
    root = tmp_path / "study"
    shutil.copytree(study, root)
    path = root / "models/11/training_metrics.json"
    metrics = get(path)
    if damage == "split-unit":
        metrics["split_unit"] = "transition"
    elif damage == "seed-rule":
        metrics["environment_seed_rule"] = "other seed protocol"
    elif damage == "sample-count":
        metrics["validation_samples"] += 1
    elif damage == "partition-swap":
        metrics["train_episode_ids"][0], metrics["validation_episode_ids"][0] = (
            metrics["validation_episode_ids"][0],
            metrics["train_episode_ids"][0],
        )
    elif damage == "learning-rate":
        metrics["learning_rate"] = 0.1
    elif damage == "calibration-quantile":
        metrics["calibration_quantile"] = 0.5
    elif damage == "calibration-multiplier":
        metrics["calibration_multiplier"] = 2.0
    else:
        metrics["calibration_floor"] = 0.5
    put(path, metrics)
    p = root / "models/11/provenance.json"
    data = get(p)
    data["training_metrics"] = metrics
    put(p, data)
    with pytest.raises(ValueError):
        verify(root)


@pytest.mark.parametrize(
    "damage",
    [
        "schema-bool",
        "runtime-bool",
        "runtime-missing",
        "source-omission",
        "source-extra",
        "origin",
        "memory-scope",
    ],
)
def test_invalid_provenance_cannot_pass_complete_check(study, tmp_path, damage):
    root = tmp_path / "study"
    shutil.copytree(study, root)
    p = root / "models/11/provenance.json"
    v = get(p)
    if damage == "schema-bool":
        v["schema_version"] = True
    elif damage == "runtime-bool":
        v["runtime"]["torch_num_threads"] = True
    elif damage == "runtime-missing":
        v.pop("runtime")
    elif damage == "source-omission":
        v["source_sha256"].pop("src/self_healing_embodied_agents/agents.py")
    elif damage == "source-extra":
        v["source_sha256"]["../../elsewhere.py"] = "a" * 64
    elif damage == "origin":
        v["model_origin"] = "reused"
    else:
        v["memory_scope"] = "shared across all evaluation roots"
    put(p, v)
    with pytest.raises(ValueError):
        verify(root)


@pytest.mark.parametrize(
    "bad", ["unknown-detector", "unknown-recovery", "update-alias", "foreign-memory"]
)
def test_mutated_runtime_settings_rejected_before_reset(bad):
    agent = SelfHealingAgent(detector="failure", recovery="nominal")
    if bad == "unknown-detector":
        agent.detector = "unsupported"
    elif bad == "unknown-recovery":
        agent.recovery = "unsupported"
    elif bad == "update-alias":
        agent.update_memory = "false"
    else:
        agent.memory = object()

    class Env(TabletopManipulationEnv):
        def reset(self):
            raise AssertionError("reset before validation")

    with pytest.raises((ValueError, TypeError)):
        agent.run_episode(Env())


def test_original_valid_episode_and_study_remain_accepted(study):
    validate_episode(episode(), complete=True)
    assert verify(study, replay=True)["evaluation_episodes"] == 980


@pytest.mark.parametrize("detector", ["failure", "postcondition", "analytical"])
@pytest.mark.parametrize(
    "fault", ["none", "grasp_slip", "blocked_path", "compound_slip_block"]
)
def test_valid_controls_and_partial_plan_completion_still_pass(detector, fault):
    result = SelfHealingAgent(detector=detector, recovery="nominal").run_episode(
        TabletopManipulationEnv(seed=17, perturbation=fault)
    )
    validate_episode({**result.as_dict(), "event_log": result.event_log}, complete=True)


@pytest.mark.parametrize("value", [True, 0, -1, "1"])
def test_provenance_thread_counts_remain_typed_and_positive(study, value):
    from self_healing_embodied_agents.release_contracts import validate_provenance

    p = get(study / "models/11/provenance.json")
    p["runtime"]["torch_num_interop_threads"] = value
    with pytest.raises(ValueError):
        validate_provenance(p)


def test_historical_unknown_package_version_is_retained_as_unknown(study):
    from self_healing_embodied_agents.release_contracts import validate_provenance

    p = get(study / "models/11/provenance.json")
    p["runtime"]["dependencies"]["self-healing-embodied-agents"] = None
    validate_provenance(p)


def test_candidate_completion_does_not_require_task_completion():
    row = episode()
    selected = next(e for e in row["event_log"] if e["event"] == "recovery_selected")
    # The selected path-clearing candidate ends before the task. Preserve that
    # legitimate short-candidate completion, rather than counting it as task success.
    assert selected["actions"] == ["clear_path"]
    outcome = next(e for e in row["event_log"] if e["event"] == "recovery_outcome")
    assert outcome["success"] and outcome["step"] < row["steps"]
    validate_episode(row, complete=True)
