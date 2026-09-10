"""Run the separately versioned monitoring study without replacing reference data."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import time
from dataclasses import asdict
from pathlib import Path

import self_healing_embodied_agents
from self_healing_embodied_agents.agents import AgentConfig, AlwaysReplanAgent, SelfHealingAgent
from self_healing_embodied_agents.benchmark import episode_seed, run_benchmark, validate_seeds
from self_healing_embodied_agents.comparison import METHODS, episode_metrics, summarize
from self_healing_embodied_agents.env import PERTURBATIONS, EnvConfig, TabletopManipulationEnv
from self_healing_embodied_agents.provenance import build_provenance, file_sha256
from self_healing_embodied_agents.recovery import RecoveryMemory, TaskOutcomeMemory
from self_healing_embodied_agents.training import save_bundle, train_transition_model

if __package__:
    from ._common import integer, write_json
else:
    from _common import integer, write_json


def condition_orders() -> list[list[str]]:
    """Balance each condition's position and directed adjacent condition pairs."""
    base = [0, 1, 6, 2, 5, 3, 4]
    if len(PERTURBATIONS) != 7:
        raise ValueError("the frozen Williams schedule requires seven conditions")
    forward = [[PERTURBATIONS[(i + shift) % 7] for i in base] for shift in range(7)]
    return forward + [list(reversed(order)) for order in forward]


def load_study(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported comparison configuration")
    for key in ("episodes_per_condition", "max_steps", "bootstrap_samples"):
        integer(key, config[key])
    integer("bootstrap_seed", config["bootstrap_seed"], 0, 2**32 - 1)
    integer("training.episodes", config["training"]["episodes"], 2)
    integer("training.epochs", config["training"]["epochs"])
    roots: set[int] = set()
    model_seeds: set[int] = set()
    for split in ("development", "evaluation"):
        settings = config[split]
        validate_seeds(settings["training_seeds"])
        if model_seeds.intersection(settings["training_seeds"]):
            raise ValueError("training seeds must be disjoint between study splits")
        model_seeds.update(settings["training_seeds"])
        for key in ("fit_roots", "evaluation_roots"):
            validate_seeds(settings[key])
            if len(settings[key]) != len(condition_orders()):
                raise ValueError("each split needs fourteen independent roots per phase")
            if roots.intersection(settings[key]):
                raise ValueError("fit and evaluation root sets must be disjoint")
            roots.update(settings[key])
    if config["methods"] != list(METHODS):
        raise ValueError("configured methods do not match the implemented comparison")
    return config


def source_snapshot() -> dict[str, str]:
    package_root = Path(self_healing_embodied_agents.__file__).resolve().parent
    return {f"{prefix}/{path.relative_to(base).as_posix()}": file_sha256(path)
            for base, prefix in ((package_root, "src/self_healing_embodied_agents"),
                                 (Path(__file__).resolve().parent, "scripts"))
            for path in sorted(base.rglob("*.py"))}


def agents_for_history(bundle, config: AgentConfig, fitted: list) -> list:
    return [
        AlwaysReplanAgent(config=config),
        SelfHealingAgent(detector="failure", recovery="nominal", config=config),
        SelfHealingAgent(detector="postcondition", recovery="nominal", config=config),
        SelfHealingAgent(detector="analytical", recovery="nominal", config=config),
        SelfHealingAgent(bundle, recovery="nominal", config=config),
        SelfHealingAgent(bundle, config=config),
        SelfHealingAgent(bundle, memory=RecoveryMemory(), config=config),
        SelfHealingAgent(bundle, memory=TaskOutcomeMemory(), config=config),
        *[SelfHealingAgent(bundle, memory=copy.deepcopy(memory), update_memory=False, config=config)
          for memory in fitted],
    ]


def run_study(config: dict, split: str, output: Path) -> dict:
    settings = config[split]
    orders = condition_orders()
    agent_config = AgentConfig(max_steps=config["max_steps"])
    all_rows: list[dict] = []
    snapshots: list[dict] = []
    sources = source_snapshot()
    started = time.monotonic()
    # The caller reserves an empty directory. A failed run has no summary.json.
    write_json(config, output / "config.json")
    with gzip.open(output / "episodes.jsonl.gz", "wt", encoding="utf-8") as traces:
        for model_seed in settings["training_seeds"]:
            model_start = time.monotonic()
            bundle, metrics = train_transition_model(seed=model_seed, **config["training"])
            model_dir = output / "models" / str(model_seed)
            checkpoint = model_dir / "world_model.pt"
            save_bundle(bundle, metrics, checkpoint, model_dir / "training_metrics.json")
            provenance = build_provenance(
                config=config, model_path=checkpoint, seeds=settings["evaluation_roots"],
                model_origin="trained", scripts_path=Path(__file__).parent,
                training_metrics=asdict(metrics),
            )
            provenance.update({
                "study_split": split, "reference_commit": config["reference_commit"],
                "memory_scope": "independent model/history; cold online or separately fitted frozen",
                "ordered_conditions": orders,
                "episode_order": "training seed, history, phase (fit/evaluation), condition, repetition, method",
                "fit_roots": settings["fit_roots"],
                "config_sha256": file_sha256(output / "config.json"),
            })
            if provenance["source_sha256"] != sources:
                raise RuntimeError("source changed during comparison")
            write_json(provenance, model_dir / "provenance.json")
            print(f"trained seed {model_seed} in {time.monotonic() - model_start:.1f}s", flush=True)

            for history_id, order in enumerate(orders):
                fitted = [RecoveryMemory(), TaskOutcomeMemory()]
                fit_agents = [SelfHealingAgent(bundle, memory=memory, config=agent_config) for memory in fitted]
                for phase, root, agents in (
                    ("fit", settings["fit_roots"][history_id], fit_agents),
                    ("evaluation", settings["evaluation_roots"][history_id], None),
                ):
                    if phase == "evaluation":
                        agents = agents_for_history(bundle, agent_config, fitted)
                        names = [agent.name for agent in agents[:-2]] + [agent.name + "_frozen" for agent in agents[-2:]]
                        if names != config["methods"]:
                            raise ValueError("configured methods do not match the implemented comparison")
                        snapshots.append({
                            "training_seed": model_seed, "history_id": history_id,
                            "fit_root": settings["fit_roots"][history_id], "evaluation_root": root,
                            "order": order, "memories": [memory_snapshot(memory) for memory in fitted],
                        })
                    else:
                        names = [agent.name for agent in agents]
                    for condition in order:
                        for repetition in range(config["episodes_per_condition"]):
                            for name, agent in zip(names, agents, strict=True):
                                env = TabletopManipulationEnv(
                                    seed=episode_seed(root, repetition), perturbation=condition,
                                    config=EnvConfig(max_steps=config["max_steps"]),
                                )
                                result = agent.run_episode(env)
                                result.agent, result.root_seed, result.repetition = name, root, repetition
                                row = {
                                    **result.as_dict(), **episode_metrics(result),
                                    "phase": phase, "training_seed": model_seed, "history_id": history_id,
                                    "order": order, "event_log": result.event_log,
                                }
                                traces.write(json.dumps(row, allow_nan=False) + "\n")
                                all_rows.append(row)
                    if phase == "evaluation":
                        if [memory_snapshot(agent.memory) for agent in agents[-2:]] != snapshots[-1]["memories"]:
                            raise AssertionError("frozen evaluation changed fitted memory")
                print(f"seed {model_seed}, history {history_id + 1}/14 complete", flush=True)
    write_json(snapshots, output / "memory_snapshots.json")
    summary = summarize(all_rows, max_steps=config["max_steps"],
                        bootstrap_seed=config["bootstrap_seed"], bootstrap_samples=config["bootstrap_samples"])
    summary.update({
        "split": split, "interpretation": config["interpretation"],
        "total_elapsed_seconds": time.monotonic() - started,
        "trace_sha256": file_sha256(output / "episodes.jsonl.gz"),
        "config_sha256": file_sha256(output / "config.json"),
    })
    if source_snapshot() != sources:
        raise RuntimeError("source changed during comparison")
    write_json(summary, output / "summary.json")
    return summary


def memory_snapshot(memory) -> dict:
    # Tuple keys have an unambiguous structured representation in JSON.
    return {field: [[list(key), value] for key, value in sorted(counts.items())]
            for field, counts in asdict(memory).items()}


def audit_reference(root: Path, output: Path) -> dict:
    reference = [json.loads(line) for line in (root / "results/episodes.jsonl").read_text().splitlines()]
    lookup = {(row["agent"], row["perturbation"], row["seed"]): row for row in reference}
    model_path = root / "artifacts/world_model.pt"
    results = run_benchmark(model_path=model_path, seeds=list(range(10)), episodes_per_condition=3)
    mismatches = []
    for result in results:
        row = result.as_dict()
        expected = lookup[(result.agent, result.perturbation, result.seed)]
        if any(value != expected[key] for key, value in row.items()):
            mismatches.append({"agent": result.agent, "seed": result.seed, "condition": result.perturbation})
    diagnostics = {"always_replan_vs_self_healing": [], "failure_ranked_vs_reactive": []}
    for root_seed in range(10):
        for condition in PERTURBATIONS:
            for repetition in range(3):
                seed = episode_seed(root_seed, repetition)
                for key, agent, baseline in (
                    ("always_replan_vs_self_healing", AlwaysReplanAgent(), "self_healing"),
                    ("failure_ranked_vs_reactive", SelfHealingAgent(detector="failure"), "reactive_replan"),
                ):
                    result = agent.run_episode(TabletopManipulationEnv(seed=seed, perturbation=condition))
                    expected = lookup[(baseline, condition, seed)]
                    if (int(result.success), result.steps) != (expected["success"], expected["steps"]):
                        diagnostics[key].append({"seed": seed, "condition": condition})
    memory_differences = [row["steps"] - lookup[("self_healing", row["perturbation"], row["seed"])]["steps"]
                          for row in reference if row["agent"] == "self_healing_memory" and row["perturbation"] != "none"]
    report = {
        "reference_commit": "bfbc77386b5e37f20a3c257dc606c7a0104bba0e",
        "reference_episodes": len(reference), "replayed_episodes": len(results),
        "counter_mismatches": mismatches, "paired_cases_per_diagnostic": 210,
        "diagnostic_mismatches": diagnostics,
        "memory_fault_episodes_slower": sum(delta > 0 for delta in memory_differences),
        "memory_fault_episodes_faster": sum(delta < 0 for delta in memory_differences),
        "reference_sha256": {str(path.relative_to(root)): file_sha256(path)
                             for path in sorted((root / "results").glob("*")) if path.is_file()},
        "checkpoint_sha256": file_sha256(model_path),
        "note": "Historical counters are replayed; additional planning and trigger annotations are not historical trace fields.",
    }
    write_json(report, output / "reference_audit.json")
    if mismatches or any(diagnostics.values()):
        raise AssertionError("reference audit changed; inspect reference_audit.json")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/comparison.json")
    parser.add_argument("--split", choices=("development", "evaluation"), default="development")
    parser.add_argument("--reference-audit", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="new output directory; existing paths are never overwritten")
    args = parser.parse_args(argv)
    try:
        config = load_study(args.config)
        args.out.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.reference_audit:
        audit_reference(root, args.out)
        print("reference counters and paired diagnostics match")
    else:
        run_study(config, args.split, args.out)
        print(f"completed {args.split} comparison in {args.out}")


if __name__ == "__main__":
    main()
