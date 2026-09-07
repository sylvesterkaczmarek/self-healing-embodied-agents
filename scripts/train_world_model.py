from __future__ import annotations

import argparse
from pathlib import Path

from self_healing_embodied_agents.training import save_bundle, train_transition_model

if __package__:
    from ._common import load_config, validate_config
else:
    from _common import load_config, validate_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a transition model and record its calibration measurements.")
    parser.add_argument("--config", type=Path, help="JSON configuration; command-line values override it")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--out", type=Path, default=Path("."), help="root directory for default artifact and metrics paths")
    parser.add_argument("--output", type=Path, help="explicit checkpoint path, overriding --out")
    parser.add_argument("--metrics", type=Path, help="explicit metrics path, overriding --out")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        wm = config["world_model"]
        for name in ("seed", "episodes", "epochs", "learning_rate"):
            if getattr(args, name) is not None:
                wm[name] = getattr(args, name)
        validate_config(config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    model_path = args.output if args.output is not None else args.out / "artifacts/world_model.pt"
    metrics_path = args.metrics if args.metrics is not None else args.out / "results/world_model_metrics.json"
    if model_path.resolve() == metrics_path.resolve():
        parser.error("checkpoint and metrics paths must be different")
    bundle, metrics = train_transition_model(**wm)
    save_bundle(bundle, metrics, model_path, metrics_path)
    print(f"saved model: {model_path}")
    print(f"calibration MSE: {metrics.validation_mse:.6f}")
    print(f"residual threshold: {metrics.residual_threshold:.6f}")


if __name__ == "__main__":
    main()
