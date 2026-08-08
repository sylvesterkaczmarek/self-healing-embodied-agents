from __future__ import annotations

import argparse
from pathlib import Path

from self_healing_embodied_agents.training import save_bundle, train_transition_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("artifacts/world_model.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("results/world_model_metrics.json"))
    args = parser.parse_args()

    bundle, metrics = train_transition_model(seed=args.seed, episodes=args.episodes, epochs=args.epochs)
    save_bundle(bundle, metrics, args.output, args.metrics)
    print(f"saved model: {args.output}")
    print(f"validation MSE: {metrics.validation_mse:.6f}")
    print(f"residual threshold: {metrics.residual_threshold:.6f}")


if __name__ == "__main__":
    main()
