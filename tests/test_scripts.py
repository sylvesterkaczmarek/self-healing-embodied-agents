from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts import _common, reproduce, run_benchmark, train_world_model
from self_healing_embodied_agents.provenance import build_provenance


def _config(tmp_path: Path, **updates) -> Path:
    settings = {"world_model": {"seed": 7, "episodes": 6, "epochs": 2}, "benchmark": {"seeds": 1, "episodes_per_condition": 1, "action_budget": 4}}
    for section, values in updates.items():
        settings[section].update(values)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path


@pytest.mark.parametrize("section,key,value", [
    ("world_model", "seed", True),
    ("world_model", "seed", 2**32),
    ("world_model", "seed", -1),
    ("world_model", "episodes", 1),
    ("world_model", "episodes", 6.9),
    ("world_model", "epochs", "2"),
    ("world_model", "learning_rate", float("nan")),
    ("world_model", "learning_rate", float("inf")),
    ("world_model", "learning_rate", True),
    ("benchmark", "seeds", 0),
    ("benchmark", "seeds", 2**32 + 1),
    ("benchmark", "episodes_per_condition", -2),
    ("benchmark", "episodes_per_condition", 1.5),
    ("benchmark", "action_budget", False),
    ("benchmark", "action_budget", 0),
])
def test_config_rejects_invalid_values_before_execution(tmp_path, monkeypatch, section, key, value):
    config = _config(tmp_path, **{section: {key: value}})
    monkeypatch.setattr(reproduce, "train_transition_model", lambda **kwargs: pytest.fail("training must not begin"))
    with pytest.raises(SystemExit) as error:
        reproduce.main(["--config", str(config), "--out", str(tmp_path / "output")])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("content", [
    '[]',
    '{"unknown": {}}',
    '{"benchmark": {"action_buget": 2}}',
    '{"world_model": []}',
    '{"benchmark": {"seeds": 1, "seeds": 2}}',
])
def test_config_rejects_unknown_malformed_and_duplicate_keys(tmp_path, content):
    config = tmp_path / "config.json"
    config.write_text(content)
    with pytest.raises(ValueError):
        _common.load_config(config)


@dataclass
class _Metrics:
    validation_mse: float = 0.125
    residual_threshold: float = 0.25


def _mock_reproduction(monkeypatch, budget: int = 4):
    calls = {}
    def train(**settings):
        calls["training"] = settings
        return object(), _Metrics()
    def save(bundle, metrics, path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new-checkpoint")
        _common.write_json({"validation_mse": metrics.validation_mse}, metrics_path)
    def benchmark(**settings):
        calls["benchmark"] = settings
        assert settings["model_path"].read_bytes() == b"new-checkpoint"
        return [object()]
    def aggregate(results, *, action_budget):
        assert action_budget == budget
        return [{"action_budget": action_budget}]
    def outputs(results, summary, output, action_budget):
        assert action_budget == budget
        _common.write_json(summary, output / "summary.json")
    monkeypatch.setattr(reproduce, "train_transition_model", train)
    monkeypatch.setattr(reproduce, "save_bundle", save)
    monkeypatch.setattr(reproduce, "run_benchmark", benchmark)
    monkeypatch.setattr(reproduce, "aggregate", aggregate)
    monkeypatch.setattr(reproduce, "benchmark_outputs", outputs)
    return calls


def test_reproduce_honours_config_budget_and_publishes_provenance(tmp_path, monkeypatch):
    config = _config(tmp_path)
    calls = _mock_reproduction(monkeypatch)
    output = tmp_path / "output"
    reproduce.main(["--config", str(config), "--out", str(output)])
    assert calls["training"] == {"seed": 7, "episodes": 6, "epochs": 2, "learning_rate": 0.002}
    assert calls["benchmark"]["seeds"] == [0]
    assert calls["benchmark"]["episodes_per_condition"] == 1
    assert json.loads((output / "results/summary.json").read_text()) == [{"action_budget": 4}]
    metadata = json.loads((output / "results/provenance.json").read_text())
    assert metadata["effective_config"]["benchmark"]["action_budget"] == 4
    assert metadata["training_metrics"]["validation_mse"] == 0.125
    assert metadata["checkpoint_sha256"] == hashlib.sha256(b"new-checkpoint").hexdigest()
    assert metadata["model_origin"] == "trained"
    assert metadata["memory_scope"] == "per_root_seed"
    assert "scripts/reproduce.py" in metadata["source_sha256"]
    assert not list(output.glob(".reproduce-*"))


@pytest.mark.parametrize("failure_at", ["training", "benchmark", "report"])
def test_reproduce_failure_preserves_previous_checkpoint_and_results(tmp_path, monkeypatch, failure_at):
    config = _config(tmp_path)
    _mock_reproduction(monkeypatch)
    output = tmp_path / "output"
    checkpoint = output / "artifacts/world_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"previous-checkpoint")
    summary = output / "results/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("previous-summary")
    symbol = {"training": "train_transition_model", "benchmark": "run_benchmark", "report": "benchmark_outputs"}[failure_at]
    def failure(*args, **kwargs):
        raise RuntimeError("experiment failed")
    monkeypatch.setattr(reproduce, symbol, failure)
    with pytest.raises(RuntimeError, match="experiment failed"):
        reproduce.main(["--config", str(config), "--out", str(output)])
    assert checkpoint.read_bytes() == b"previous-checkpoint"
    assert summary.read_text() == "previous-summary"
    assert not (output / "results/world_model_metrics.json").exists()
    assert not list(output.glob(".reproduce-*"))


def test_reproduce_reused_model_does_not_claim_it_was_trained(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _mock_reproduction(monkeypatch)
    monkeypatch.setattr(reproduce, "train_transition_model", lambda **kw: pytest.fail("supplied model must be reused"))
    supplied = tmp_path / "supplied.pt"
    supplied.write_bytes(b"new-checkpoint")
    output = tmp_path / "output"
    reproduce.main(["--config", str(config), "--out", str(output), "--model", str(supplied)])
    metadata = json.loads((output / "results/provenance.json").read_text())
    assert metadata["model_origin"] == "reused"
    assert metadata["effective_config"]["world_model"] is None
    assert metadata["training_metrics"] is None
    assert json.loads((output / "results/world_model_metrics.json").read_text())["training_metrics"] is None
    assert supplied.read_bytes() == b"new-checkpoint"


def test_publication_rolls_back_all_earlier_replacements(tmp_path, monkeypatch):
    stage, destination = tmp_path / "stage", tmp_path / "destination"
    stage.mkdir()
    destination.mkdir()
    for name in ("a", "b", "c"):
        (stage / name).write_text("new")
    (destination / "a").write_text("old")
    original_replace = _common.os.replace
    def fail_last(source, target):
        if Path(source) == stage / "c":
            raise OSError("disk failure")
        return original_replace(source, target)
    monkeypatch.setattr(_common.os, "replace", fail_last)
    with pytest.raises(OSError, match="disk failure"):
        _common.publish_files(stage, destination)
    assert (destination / "a").read_text() == "old"
    assert not (destination / "b").exists()
    assert not (destination / "c").exists()


def test_benchmark_cli_overrides_config_and_uses_requested_output(tmp_path, monkeypatch):
    config = _config(tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    def benchmark(**settings):
        assert settings["seeds"] == [0, 1]
        assert settings["episodes_per_condition"] == 2
        assert settings["model_path"] == model
        return [object()]
    def aggregate(results, *, action_budget):
        assert action_budget == 9
        return [{"agent": "test", "perturbation": "unused", "action_budget": action_budget}]
    monkeypatch.setattr(run_benchmark, "run_benchmark", benchmark)
    monkeypatch.setattr(run_benchmark, "aggregate", aggregate)
    monkeypatch.setattr(run_benchmark, "benchmark_outputs", lambda results, summary, output, budget: _common.write_json(summary, output / "summary.json"))
    output = tmp_path / "report"
    run_benchmark.main(["--config", str(config), "--model", str(model), "--seeds", "2", "--episodes-per-condition", "2", "--action-budget", "9", "--out", str(output)])
    assert json.loads((output / "summary.json").read_text())[0]["action_budget"] == 9
    assert (output / "provenance.json").is_file()


def test_training_cli_overrides_config_and_preserves_output_options(tmp_path, monkeypatch):
    config = _config(tmp_path)
    calls = {}
    def train(**kwargs):
        calls["training"] = kwargs
        return object(), _Metrics()
    def save(bundle, metrics, path, metrics_path):
        calls["paths"] = path, metrics_path
    monkeypatch.setattr(train_world_model, "train_transition_model", train)
    monkeypatch.setattr(train_world_model, "save_bundle", save)
    output, metrics = tmp_path / "chosen.pt", tmp_path / "chosen.json"
    train_world_model.main(["--config", str(config), "--epochs", "3", "--learning-rate", "0.004", "--output", str(output), "--metrics", str(metrics)])
    assert calls["training"] == {"seed": 7, "episodes": 6, "epochs": 3, "learning_rate": 0.004}
    assert calls["paths"] == (output, metrics)


@pytest.mark.parametrize("script", ["reproduce", "run_benchmark", "train_world_model", "smoke_demo"])
@pytest.mark.parametrize("as_module", [False, True])
def test_script_and_module_entrypoints_show_help(script, as_module):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", f"scripts.{script}"] if as_module else [sys.executable, str(root / "scripts" / f"{script}.py")]
    result = subprocess.run([*command, "--help"], cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_provenance_records_exact_source_and_runtime(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint")
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    metadata = build_provenance(config={"benchmark": {"seeds": 2}}, model_path=model, seeds=[0, 1], model_origin="reused", scripts_path=scripts)
    assert metadata["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert metadata["source_sha256"]["scripts/reproduce.py"] == hashlib.sha256((scripts / "reproduce.py").read_bytes()).hexdigest()
    assert metadata["runtime"]["device"] == "cpu"
    assert metadata["runtime"]["torch_num_threads"] >= 1
    assert metadata["evaluation_seeds"] == [0, 1]


def test_failed_rollback_retains_previous_files_for_recovery(tmp_path, monkeypatch):
    stage, destination = tmp_path / "stage", tmp_path / "destination"
    stage.mkdir()
    destination.mkdir()
    for name in ("a", "b"):
        (stage / name).write_text("new")
        (destination / name).write_text("old")
    original_replace, original_copy = _common.os.replace, _common.shutil.copy2
    def fail_second(source, target):
        if Path(source) == stage / "b":
            raise OSError("publication failed")
        return original_replace(source, target)
    def fail_restore(source, target):
        if Path(source).parent.name.startswith(".publication-backup-"):
            raise OSError("restore failed")
        return original_copy(source, target)
    monkeypatch.setattr(_common.os, "replace", fail_second)
    monkeypatch.setattr(_common.shutil, "copy2", fail_restore)
    with pytest.raises(RuntimeError, match="previous files remain"):
        _common.publish_files(stage, destination)
    backups = list(destination.glob(".publication-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "a").read_text() == "old"
    assert (backups[0] / "b").read_text() == "old"
