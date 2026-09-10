from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts import run_comparison as runner
from self_healing_embodied_agents.env import PERTURBATIONS, nominal_transition


def _study() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs/comparison.json"
    return json.loads(path.read_text())


@dataclass
class _Metrics:
    seed: int


class _NominalBundle:
    residual_threshold = 0.01

    def predict(self, state, action):
        return nominal_transition(state, action).state.vector()


@pytest.fixture
def small_study(monkeypatch, tmp_path):
    config = _study()
    config.update(episodes_per_condition=2, bootstrap_samples=10)
    config["development"] = {
        "training_seeds": [1, 2], "fit_roots": [10, 11], "evaluation_roots": [20, 21],
    }
    orders = [["grasp_slip", "compound_slip_block"], ["compound_slip_block", "grasp_slip"]]
    monkeypatch.setattr(runner, "condition_orders", lambda: orders)
    monkeypatch.setattr(runner, "train_transition_model",
                        lambda seed, **kwargs: (_NominalBundle(), _Metrics(seed)))

    def save(bundle, metrics, checkpoint, metrics_path):
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"test-checkpoint")
        runner.write_json({"seed": metrics.seed}, metrics_path)

    monkeypatch.setattr(runner, "save_bundle", save)
    sources = {"scripts/run_comparison.py": "fixed-test-source"}
    monkeypatch.setattr(runner, "source_snapshot", lambda: sources.copy())
    monkeypatch.setattr(runner, "build_provenance", lambda **kwargs: {"source_sha256": sources.copy()})
    output = tmp_path / "study"
    output.mkdir()
    return config, output


def _rows(output):
    with gzip.open(output / "episodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def test_condition_orders_balance_positions_and_adjacent_pairs():
    orders = runner.condition_orders()
    assert len(orders) == 14
    assert all(set(order) == set(PERTURBATIONS) and len(order) == 7 for order in orders)
    for position in range(7):
        assert Counter(order[position] for order in orders) == Counter({name: 2 for name in PERTURBATIONS})
    pairs = Counter(pair for order in orders for pair in zip(order, order[1:]))
    assert pairs == Counter({(left, right): 2 for left in PERTURBATIONS
                             for right in PERTURBATIONS if left != right})


@pytest.mark.parametrize("overlap", ["training", "within_split_roots", "between_split_roots"])
def test_study_rejects_leakage_between_training_fit_and_evaluation(tmp_path, overlap):
    config = _study()
    if overlap == "training":
        config["evaluation"]["training_seeds"][0] = config["development"]["training_seeds"][0]
    elif overlap == "within_split_roots":
        config["evaluation"]["evaluation_roots"][0] = config["evaluation"]["fit_roots"][0]
    else:
        config["evaluation"]["fit_roots"][0] = config["development"]["evaluation_roots"][0]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="disjoint"):
        runner.load_study(path)


def test_existing_output_is_preserved_before_training(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_study()))
    output = tmp_path / "existing"
    output.mkdir()
    (output / "summary.json").write_text("previous evidence")
    monkeypatch.setattr(runner, "train_transition_model",
                        lambda **kwargs: pytest.fail("existing evidence must prevent training"))
    with pytest.raises(SystemExit) as error:
        runner.main(["--config", str(path), "--out", str(output)])
    assert error.value.code == 2
    assert (output / "summary.json").read_text() == "previous evidence"
    assert list(output.iterdir()) == [output / "summary.json"]


def test_histories_keep_fit_cold_and_frozen_memories_separate(small_study, monkeypatch):
    config, output = small_study
    factory = runner.agents_for_history
    histories = []

    def capture(bundle, agent_config, fitted):
        agents = factory(bundle, agent_config, fitted)
        assert all(memory.attempts for memory in fitted)
        assert all(not agent.memory.attempts for agent in agents[6:8])
        histories.append((fitted, agents[6:]))
        return agents

    monkeypatch.setattr(runner, "agents_for_history", capture)
    summary = runner.run_study(config, "development", output)
    snapshots = json.loads((output / "memory_snapshots.json").read_text())
    memories = [memory for fitted, agents in histories for memory in [*fitted, *[agent.memory for agent in agents]]]
    assert len(histories) == 4
    assert len({id(memory) for memory in memories}) == len(memories)
    assert len({id(memory.attempts) for memory in memories}) == len(memories)
    for snapshot, (fitted, agents) in zip(snapshots, histories, strict=True):
        assert all(agent.memory.attempts for agent in agents[:2])
        assert snapshot["memories"] == [runner.memory_snapshot(memory) for memory in fitted]
        assert snapshot["memories"] == [runner.memory_snapshot(agent.memory) for agent in agents[-2:]]
    rows = _rows(output)
    assert {row["root_seed"] for row in rows if row["phase"] == "fit"} == {10, 11}
    assert {row["root_seed"] for row in rows if row["phase"] == "evaluation"} == {20, 21}
    pairs = defaultdict(list)
    for row in rows:
        if row["phase"] == "evaluation":
            pairs[row["training_seed"], row["history_id"], row["perturbation"], row["repetition"]].append(row)
    assert all({row["agent"] for row in case} == set(config["methods"]) for case in pairs.values())
    assert all(len({row["seed"] for row in case}) == 1 for case in pairs.values())
    assert summary["design"]["fit_episodes"] == 32
    assert summary["design"]["evaluation_episodes"] == 160
    assert summary["trace_sha256"] == runner.file_sha256(output / "episodes.jsonl.gz")
    for seed in (1, 2):
        provenance = json.loads((output / f"models/{seed}/provenance.json").read_text())
        assert provenance["config_sha256"] == summary["config_sha256"]
        assert provenance["fit_roots"] == [10, 11]


def test_budget_reaches_environment_and_limits_executed_actions(small_study, monkeypatch):
    config, output = small_study
    config["max_steps"] = 3
    environment = runner.TabletopManipulationEnv
    limits = []

    def capture(**kwargs):
        env = environment(**kwargs)
        limits.append(env.config.max_steps)
        return env

    monkeypatch.setattr(runner, "TabletopManipulationEnv", capture)
    summary = runner.run_study(config, "development", output)
    assert limits and set(limits) == {3}
    assert all(row["steps"] == 3 and not row["success"] for row in _rows(output))
    assert all(row["mean_capped_actions"] == 3 for row in summary["pooled"])


def test_source_change_prevents_completed_summary(small_study, monkeypatch):
    config, output = small_study
    snapshots = iter([{"scripts/run_comparison.py": "fixed-test-source"}, {"scripts/run_comparison.py": "changed"}])
    monkeypatch.setattr(runner, "source_snapshot", lambda: next(snapshots))
    with pytest.raises(RuntimeError, match="source changed during comparison"):
        runner.run_study(config, "development", output)
    assert _rows(output)
    assert not (output / "summary.json").exists()


def test_frozen_memory_mutation_prevents_completed_summary(small_study, monkeypatch):
    config, output = small_study
    factory = runner.agents_for_history

    def unfreeze(bundle, agent_config, fitted):
        agents = factory(bundle, agent_config, fitted)
        agents[-1].update_memory = True
        return agents

    monkeypatch.setattr(runner, "agents_for_history", unfreeze)
    with pytest.raises(AssertionError, match="frozen evaluation changed fitted memory"):
        runner.run_study(config, "development", output)
    assert not (output / "summary.json").exists()
