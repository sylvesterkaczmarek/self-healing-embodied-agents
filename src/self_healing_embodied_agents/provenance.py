from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

from .env import PERTURBATIONS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    *,
    config: dict,
    model_path: Path,
    seeds: list[int],
    model_origin: str,
    scripts_path: Path | None = None,
    training_metrics: dict | None = None,
) -> dict:
    """Record the effective experiment and the exact code/checkpoint it used."""
    if model_origin not in {"trained", "reused"}:
        raise ValueError("model_origin must be trained or reused")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be distinct integers in 0..2**32-1")
    package_root = Path(__file__).resolve().parent
    source_hashes = {
        f"src/self_healing_embodied_agents/{path.relative_to(package_root).as_posix()}": file_sha256(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    if scripts_path is not None:
        for path in sorted(Path(scripts_path).rglob("*.py")):
            source_hashes[f"scripts/{path.relative_to(scripts_path).as_posix()}"] = file_sha256(path)
    versions = {}
    for distribution in ("self-healing-embodied-agents", "numpy", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    value = {
        "schema_version": 1,
        "effective_config": config,
        "model_origin": model_origin,
        "training_metrics": training_metrics,
        "checkpoint_sha256": file_sha256(model_path),
        "source_sha256": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": versions,
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
        "evaluation_seeds": seeds,
        "evaluation_seed_rule": "SeedSequence([root_seed, repetition, 0x4556414C]) generates one uint64",
        "paired_across": "agents and perturbations",
        "detection_aggregation": "pooled true-positive counts / pooled detections or injected failure steps",
        "memory_scope": "per_root_seed",
        "ordered_conditions": list(PERTURBATIONS),
        "episode_order": "root seed, perturbation, repetition, agent",
    }
    # Copy the input structures and reject non-finite metadata before writing.
    return json.loads(json.dumps(value, allow_nan=False))
