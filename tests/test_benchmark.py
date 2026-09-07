from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from self_healing_embodied_agents import benchmark
from self_healing_embodied_agents.plotting import plot_budget_success, plot_success
from self_healing_embodied_agents.types import EpisodeResult


def episode(**kwargs) -> EpisodeResult:
    values = dict(success=True, steps=6, perturbation="none", agent="self_healing", seed=1)
    values.update(kwargs)
    return EpisodeResult(**values)


def test_configured_budget_changes_metric_without_relabelling_legacy_metric():
    rows = [episode(steps=6), episode(seed=2, steps=8), episode(seed=3, success=False, steps=4)]
    summary = benchmark.aggregate(rows, action_budget=9)[0]
    assert summary["action_budget"] == 9
    assert summary["success_within_action_budget"] == pytest.approx(2 / 3)
    assert summary["success_within_7_actions"] == pytest.approx(1 / 3)
    assert summary["success_rate"] == pytest.approx(2 / 3)


def test_detection_rates_pool_events_across_episodes():
    rows = [
        episode(true_failures=1, detections=1, true_positive_detections=1),
        episode(seed=2, steps=10, true_failures=9, detections=9,
                true_positive_detections=1, false_positive_detections=8),
    ]
    row = benchmark.aggregate(rows)[0]
    assert row["detection_precision"] == pytest.approx(2 / 10)
    assert row["detection_recall"] == pytest.approx(2 / 10)
    assert row["detections"] == row["true_failures"] == 10
    assert row["true_positive_detections"] == 2


def test_undefined_rates_are_null_and_zero_recall_is_defined(tmp_path):
    no_events = episode()
    assert no_events.as_dict()["detection_precision"] is None
    assert no_events.as_dict()["detection_recall"] is None
    summary = benchmark.aggregate([no_events])
    benchmark.save_summary(summary, tmp_path / "nested/summary.json", tmp_path / "other/summary.csv")
    assert json.loads((tmp_path / "nested/summary.json").read_text())[0]["detection_precision"] is None
    row = benchmark.aggregate([episode(true_failures=1)])[0]
    assert row["detection_recall"] == 0
    assert row["detection_precision"] is None


@pytest.mark.parametrize("changes", [
    {"steps": -1}, {"steps": True}, {"seed": 1.5}, {"success": 1},
    {"detections": 1}, {"true_positive_detections": 1, "detections": 1},
    {"recovery_successes": 1}, {"steps": 1, "true_failures": 2},
    {"root_seed": 1}, {"root_seed": 1, "repetition": -1},
])
def test_inconsistent_episode_counters_are_rejected(changes):
    with pytest.raises(ValueError):
        benchmark.aggregate([episode(**changes)])


def test_aggregate_rejects_duplicates_and_conflicting_run_coordinates():
    row = episode()
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.aggregate([row, replace(row)])
    with pytest.raises(ValueError, match="shared episode seed"):
        benchmark.aggregate([
            episode(root_seed=7, repetition=0),
            episode(seed=2, perturbation="grasp_slip", root_seed=7, repetition=0),
        ])


@pytest.mark.parametrize("budget", [0, -1, True, 7.5, "7"])
def test_invalid_action_budgets_are_rejected(budget):
    with pytest.raises(ValueError, match="action_budget"):
        benchmark.aggregate([episode()], action_budget=budget)


@pytest.mark.parametrize("seeds,repetitions", [
    ([], 1), ([1, 1], 1), ([True], 1), ([1.5], 1), ([2**32], 1), ([-1], 1),
    ([1], 0), ([1], True), ([1], 1.5),
])
def test_invalid_benchmark_settings_fail_before_loading_model(monkeypatch, seeds, repetitions):
    def unexpected_load(path):
        raise AssertionError("invalid benchmark must not load a model")
    monkeypatch.setattr(benchmark, "load_bundle", unexpected_load)
    with pytest.raises(ValueError):
        benchmark.run_benchmark(model_path=Path("unused.pt"), seeds=seeds,
                                episodes_per_condition=repetitions)


def test_seed_derivation_avoids_old_offset_collision():
    # The former root_seed * 100 + repetition mapped both coordinates to 200.
    assert benchmark.episode_seed(1, 100) != benchmark.episode_seed(2, 0)
    assert benchmark.episode_seed(7, 0) == benchmark.episode_seed(7, 0)
    assert 0 <= benchmark.episode_seed(2**32 - 1, 3) < 2**64


def test_memory_is_independent_between_root_seeds_and_persists_within_run(monkeypatch):
    class FakeAgent:
        def __init__(self, bundle=None, memory=None):
            self.memory = memory
            self.name = "self_healing_memory" if memory is not None else "stateless"
        def run_episode(self, env):
            count = 0
            if self.memory is not None:
                count = self.memory.get("visits", 0)
                self.memory["visits"] = count + 1
            return episode(seed=env.seed, agent=self.name, perturbation=env.perturbation,
                           event_log=[{"prior_memory_episodes": count}])
    monkeypatch.setattr(benchmark, "load_bundle", lambda path: object())
    monkeypatch.setattr(benchmark, "RecoveryMemory", dict)
    monkeypatch.setattr(benchmark, "OpenLoopAgent", FakeAgent)
    monkeypatch.setattr(benchmark, "ReactiveReplanAgent", FakeAgent)
    monkeypatch.setattr(benchmark, "SelfHealingAgent", FakeAgent)
    monkeypatch.setattr(benchmark, "PERTURBATIONS", ["none", "grasp_slip"])
    rows = benchmark.run_benchmark(model_path=Path("unused.pt"), seeds=[7, 9], episodes_per_condition=2)
    memory_rows = [row for row in rows if row.agent == "self_healing_memory"]
    for seed in [7, 9]:
        own = [row for row in memory_rows if row.root_seed == seed]
        assert [row.event_log[0]["prior_memory_episodes"] for row in own] == [0, 1, 2, 3]
        assert [row.repetition for row in own] == [0, 1, 0, 1]
        assert own[0].seed == own[2].seed
    for start in range(0, len(rows), 4):
        assert len({row.seed for row in rows[start:start + 4]}) == 1


@pytest.mark.parametrize("save", ["csv", "jsonl", "summary"])
def test_empty_output_rejected_without_overwriting_existing_file(tmp_path, save):
    path = tmp_path / "result"
    path.write_text("previous result")
    with pytest.raises(ValueError):
        if save == "csv":
            benchmark.save_episode_csv([], path)
        elif save == "jsonl":
            benchmark.save_episode_jsonl([], path)
        else:
            benchmark.save_summary([], path, tmp_path / "result.csv")
    assert path.read_text() == "previous result"


def test_jsonl_retains_decision_evidence_and_rejects_nonfinite_values(tmp_path):
    path = tmp_path / "episodes.jsonl"
    result = episode(event_log=[{"kind": "recovery_selected", "candidate": "reobserve"}])
    benchmark.save_episode_jsonl([result], path)
    saved = json.loads(path.read_text())
    assert saved["event_log"] == result.event_log
    assert saved["detection_precision"] is None
    previous = path.read_text()
    result.event_log.append({"score": float("nan")})
    with pytest.raises(ValueError):
        benchmark.save_episode_jsonl([result], path)
    assert path.read_text() == previous


def test_invalid_summary_json_preserves_existing_files(tmp_path):
    json_path, csv_path = tmp_path / "summary.json", tmp_path / "summary.csv"
    json_path.write_text("old JSON")
    csv_path.write_text("old CSV")
    with pytest.raises(ValueError):
        benchmark.save_summary([{"value": float("inf")}], json_path, csv_path)
    assert json_path.read_text() == "old JSON"
    assert csv_path.read_text() == "old CSV"


def test_plots_include_memory_agent_and_configured_budget(tmp_path):
    rows = [episode(agent=name) for name in
            ["open_loop", "reactive_replan", "self_healing", "self_healing_memory"]]
    summary = benchmark.aggregate(rows, action_budget=9)
    plot_budget_success(summary, tmp_path / "budget.svg")
    plot_success(summary, tmp_path / "success.svg")
    for name in ["budget.svg", "success.svg"]:
        assert "self healing memory" in (tmp_path / name).read_text()
    assert "Success within 9 actions" in (tmp_path / "budget.svg").read_text()
    assert "Success within 7 actions" not in (tmp_path / "budget.svg").read_text()


def test_plot_rejects_duplicate_incomplete_and_nonfinite_tables(tmp_path):
    summary = benchmark.aggregate([episode()])
    with pytest.raises(ValueError, match="duplicate"):
        plot_success(summary * 2, tmp_path / "x.svg")
    with pytest.raises(ValueError, match="combination"):
        plot_success(summary + [{**summary[0], "agent": "other", "perturbation": "grasp_slip"}],
                     tmp_path / "x.svg")
    with pytest.raises(ValueError, match="finite rate"):
        plot_success([{**summary[0], "success_rate": float("nan")}], tmp_path / "x.svg")
    assert not (tmp_path / "x.svg").exists()
