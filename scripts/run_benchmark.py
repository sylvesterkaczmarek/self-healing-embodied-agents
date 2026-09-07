from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from self_healing_embodied_agents.benchmark import aggregate, run_benchmark
from self_healing_embodied_agents.provenance import build_provenance

if __package__:
    from ._common import benchmark_outputs, load_config, publish_files, validate_config, write_json
else:
    from _common import benchmark_outputs, load_config, publish_files, validate_config, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model and save complete benchmark evidence.")
    parser.add_argument("--config", type=Path, help="JSON configuration; command-line values override it")
    parser.add_argument("--model", type=Path, default=Path("artifacts/world_model.pt"))
    parser.add_argument("--seeds", type=int, help="number of benchmark root seeds, starting at zero")
    parser.add_argument("--episodes-per-condition", type=int)
    parser.add_argument("--action-budget", type=int)
    parser.add_argument("--out", type=Path, default=Path("results"), help="directory for benchmark results")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        bench = config["benchmark"]
        for name in ("seeds", "episodes_per_condition", "action_budget"):
            if getattr(args, name) is not None:
                bench[name] = getattr(args, name)
        validate_config(config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    seeds = list(range(bench["seeds"]))
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=output) as temporary:
        stage = Path(temporary) / "run"
        results = run_benchmark(
            model_path=args.model,
            seeds=seeds,
            episodes_per_condition=bench["episodes_per_condition"],
        )
        summary = aggregate(results, action_budget=bench["action_budget"])
        benchmark_outputs(results, summary, stage, bench["action_budget"])
        write_json(build_provenance(
            config={"world_model": None, "benchmark": bench},
            model_path=args.model,
            seeds=seeds,
            model_origin="reused",
            scripts_path=Path(__file__).resolve().parent,
        ), stage / "provenance.json")
        publish_files(stage, output)
    print(f"episodes: {len(results)}; results: {output}")
    for row in summary:
        if row["perturbation"] in {"none", "grasp_slip", "compound_slip_block"}:
            print(f"{row['agent']:>20} | {row['perturbation']:<20} | success={row['success_rate']:.3f}")


if __name__ == "__main__":
    main()
