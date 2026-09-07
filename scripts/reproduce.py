from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from self_healing_embodied_agents.benchmark import aggregate, run_benchmark
from self_healing_embodied_agents.provenance import build_provenance
from self_healing_embodied_agents.training import save_bundle, train_transition_model

if __package__:
    from ._common import benchmark_outputs, load_config, publish_files, write_json
else:
    from _common import benchmark_outputs, load_config, publish_files, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and reproduce the complete benchmark in one output directory.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/reproduce.json")
    parser.add_argument("--out", type=Path, default=Path("."), help="root directory for artifacts/ and results/")
    parser.add_argument("--model", type=Path, help="reuse this checkpoint instead of training a new model")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    wm, bench = config["world_model"], config["benchmark"]
    seeds = list(range(bench["seeds"]))
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reproduce-", dir=output) as temporary:
        stage = Path(temporary) / "run"
        model_path = stage / "artifacts/world_model.pt"
        metrics_path = stage / "results/world_model_metrics.json"
        training_metrics = None
        if args.model is None:
            bundle, metrics = train_transition_model(**wm)
            training_metrics = asdict(metrics)
            save_bundle(bundle, metrics, model_path, metrics_path)
        else:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.model, model_path)
            write_json({
                "model_origin": "reused",
                "training_metrics": None,
                "note": "Training metrics are unavailable for this supplied checkpoint.",
            }, metrics_path)
        results = run_benchmark(
            model_path=model_path,
            seeds=seeds,
            episodes_per_condition=bench["episodes_per_condition"],
        )
        summary = aggregate(results, action_budget=bench["action_budget"])
        benchmark_outputs(results, summary, stage / "results", bench["action_budget"])
        provenance = build_provenance(
            config={"world_model": wm if args.model is None else None, "benchmark": bench},
            model_path=model_path,
            seeds=seeds,
            model_origin="trained" if args.model is None else "reused",
            scripts_path=Path(__file__).resolve().parent,
            training_metrics=training_metrics,
        )
        write_json(provenance, stage / "results/provenance.json")
        publish_files(stage, output)
    print(f"reproduced {len(results)} episodes in {output}")


if __name__ == "__main__":
    main()
