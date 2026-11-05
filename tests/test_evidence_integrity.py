from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import pytest
from self_healing_embodied_agents.evidence import strict_json, validate_episode
from self_healing_embodied_agents.provenance import file_sha256
from self_healing_embodied_agents.agents import (
    SelfHealingAgent,
    AgentConfig,
    AlwaysReplanAgent,
)
from self_healing_embodied_agents.env import TabletopManipulationEnv
from self_healing_embodied_agents.types import Action, ActionKind
from self_healing_embodied_agents.world_model import ModelBundle, TransitionMLP
from scripts.run_comparison import run_study, load_study
from scripts.verify_comparison import verify, read_rows


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    cfg = load_study(Path(__file__).resolve().parents[1] / "configs/comparison.json")
    cfg["development"]["training_seeds"] = [11]
    cfg["training"] = {"episodes": 12, "epochs": 2}
    cfg.update(episodes_per_condition=1, bootstrap_samples=10)
    root = tmp_path_factory.mktemp("complete")
    run_study(cfg, "development", root)
    return root


def refresh(root):
    summary = json.loads((root / "summary.json").read_text())
    summary["trace_sha256"] = file_sha256(root / "episodes.jsonl.gz")
    summary["config_sha256"] = file_sha256(root / "config.json")
    (root / "summary.json").write_text(json.dumps(summary))


def test_complete_small_study_and_action_replay(completed):
    result = verify(completed, replay=True)
    assert result["evaluation_episodes"] == 980 and result["fit_episodes"] == 196
    assert result["memory_snapshots_checked"] == 14 and result["actions_replayed"] > 0


@pytest.mark.parametrize(
    "damage",
    [
        "missing-model",
        "missing-method",
        "missing-repetition",
        "missing-history",
        "duplicate",
        "root",
        "order",
        "episode-seed",
        "observed-state",
        "memory-reward",
        "stored-metric",
        "recovery-count",
        "success",
        "terminal",
        "late-alarm",
    ],
)
def test_mutated_traces_cannot_enter_verified_evidence(tmp_path, completed, damage):
    root = tmp_path / "study"
    shutil.copytree(completed, root)
    rows = read_rows(root / "episodes.jsonl.gz")
    if damage == "missing-model":
        (root / "models/11/world_model.pt").unlink()
    elif damage == "missing-method":
        rows = [r for r in rows if r["agent"] != "always_replan"]
    elif damage == "missing-repetition":
        rows = rows[:-10]
    elif damage == "missing-history":
        rows = [r for r in rows if r["history_id"] != 13]
    elif damage == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif damage == "root":
        rows[0]["root_seed"] += 1
    elif damage == "order":
        rows[0]["order"].reverse()
    elif damage == "episode-seed":
        rows[0]["seed"] += 1
    elif damage == "observed-state":
        next(e for e in rows[0]["event_log"] if e["event"] == "transition")[
            "observed_state"
        ][0] += 0.05
    elif damage == "memory-reward":
        e = next(
            e for r in rows for e in r["event_log"] if e["event"] == "memory_update"
        )
        e["reward"] += 0.1
    elif damage == "stored-metric":
        rows[0]["failed_actions"] += 1
    elif damage == "recovery-count":
        rows[0]["recovery_attempts"] += 1
    elif damage == "success":
        rows[0]["success"] = 0
    elif damage == "terminal":
        next(e for e in rows[0]["event_log"] if e["event"] == "transition")[
            "task_success"
        ] = True
    elif damage == "late-alarm":
        next(e for r in rows for e in r["event_log"] if e["event"] == "detection")[
            "step"
        ] = 100
    with gzip.open(root / "episodes.jsonl.gz", "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    refresh(root)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(root, replay=True)


@pytest.mark.parametrize(
    "damage",
    [
        "snapshot-count",
        "snapshot-reward",
        "checkpoint-hash",
        "training-overlap",
        "threshold",
        "provenance",
        "summary",
        "extra-file",
    ],
)
def test_mutated_supporting_records_rejected(tmp_path, completed, damage):
    root = tmp_path / "study"
    shutil.copytree(completed, root)
    if damage.startswith("snapshot"):
        p = root / "memory_snapshots.json"
        value = json.loads(p.read_text())
        if damage == "snapshot-count":
            value.pop()
        else:
            value[0]["memories"][1]["reward_sums"][0][1] += 0.1
    elif damage == "extra-file":
        (root / "other.json").write_text("{}")
        p = None
    elif damage == "summary":
        p = root / "summary.json"
        value = json.loads(p.read_text())
        value["pooled"][0]["mean_steps"] += 0.1
    else:
        p = (
            root
            / "models/11"
            / (
                "training_metrics.json"
                if damage in ("training-overlap", "threshold")
                else "provenance.json"
            )
        )
        value = json.loads(p.read_text())
        if damage == "training-overlap":
            value["validation_episode_ids"][0] = value["train_episode_ids"][0]
        if damage == "threshold":
            value["residual_threshold"] += 0.1
        if damage == "checkpoint-hash":
            value["checkpoint_sha256"] = "0" * 64
        if damage == "provenance":
            value["source_sha256"] = {}
    if p:
        p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify(root)


@pytest.mark.parametrize(
    "raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "[" * 100 + "0" + "]" * 100]
)
def test_ambiguous_or_unbounded_json_rejected(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1])
def test_mutated_threshold_rejected_before_reset(value):
    class Fake:
        residual_threshold = 0.01

        def predict(self, state, action):
            return state.vector()

    class Env(TabletopManipulationEnv):
        def reset(self):
            pytest.fail("reset before validation")

    bundle = Fake()
    agent = SelfHealingAgent(bundle)
    bundle.residual_threshold = value
    with pytest.raises(ValueError):
        agent.run_episode(Env())


def test_mutated_budget_rejected_before_reset():
    agent = AlwaysReplanAgent(config=AgentConfig())
    agent.config.max_steps = True

    class Env(TabletopManipulationEnv):
        def reset(self):
            pytest.fail("reset before validation")

    with pytest.raises(ValueError):
        agent.run_episode(Env())


def test_mutated_observation_flags_rejected():
    model = ModelBundle(TransitionMLP(), 0.1)
    state = TabletopManipulationEnv().reset()
    state.holding = 2
    with pytest.raises(ValueError):
        model.predict(state, Action(ActionKind.GRASP))


def test_declared_detection_window_is_enforced():
    from self_healing_embodied_agents.comparison import episode_metrics

    r = SelfHealingAgent(detector="failure", recovery="nominal").run_episode(
        TabletopManipulationEnv(perturbation="blocked_path")
    )
    row = {**r.as_dict(), "event_log": deepcopy(r.event_log)}
    onset = next(e for e in row["event_log"] if e.get("failure"))
    onset["step"] = 1
    alarm = next(e for e in row["event_log"] if e["event"] == "detection")
    alarm["matched_failure_step"] = 1
    with pytest.raises(ValueError):
        episode_metrics(row)
    assert episode_metrics(row, detection_window=3)["matched_detection_delays"] == [2]


def test_rejected_direct_study_config_leaves_output_empty(tmp_path):
    cfg = load_study(Path(__file__).resolve().parents[1] / "configs/comparison.json")
    cfg["schema_version"] = True
    with pytest.raises(ValueError):
        run_study(cfg, "development", tmp_path)
    assert not list(tmp_path.iterdir())


def test_showcase_executes_six_contracts(tmp_path):
    from scripts.showcase import demonstrate

    root = Path(__file__).resolve().parents[1]
    assert (
        demonstrate(root / "artifacts/world_model.pt", tmp_path / "showcase")["passed"]
        == 6
    )


def test_runtime_version_matches_installation():
    import importlib.metadata as md
    import self_healing_embodied_agents as package

    assert md.version("self-healing-embodied-agents") == package.__version__


@pytest.mark.parametrize("damage", ["candidate", "reason", "missing-actions", "order"])
def test_complete_recovery_outcomes_belong_to_the_selected_candidate(damage):
    agent = SelfHealingAgent(detector="failure", recovery="nominal")
    result = agent.run_episode(
        TabletopManipulationEnv(seed=17, perturbation="blocked_path")
    )
    row = {**result.as_dict(), "event_log": deepcopy(result.event_log)}
    outcome = next(e for e in row["event_log"] if e["event"] == "recovery_outcome")
    selected = next(e for e in row["event_log"] if e["event"] == "recovery_selected")
    if damage == "candidate":
        outcome["candidate"] = "different-plan"
    if damage == "reason":
        outcome["reason"] = "interrupted"
    if damage == "missing-actions":
        selected["actions"] = []
    if damage == "order":
        row["event_log"].reverse()
    with pytest.raises(ValueError):
        validate_episode(row, complete=True)
