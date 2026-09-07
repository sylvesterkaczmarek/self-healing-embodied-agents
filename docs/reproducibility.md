# Reproducibility

## Run the reference experiment

Use Python 3.11 or later. From the repository root:

```bash
python -m pip install -e ".[dev]"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/reproduce.py \
  --config configs/reproduce.json --out runs/reproduction
python -m pytest -q
```

The reference configuration trains on 700 nominal episodes for 180 epochs with seed 7, partitioned into 560 training and 140 calibration episodes. It then evaluates four agent variants over seven conditions, ten root seeds and three repetitions per condition, totalling 840 episodes. The figure's reporting threshold is seven actions; execution is capped at 24 actions.

`--out` is the root for a run's `artifacts/` and `results/` directories. Use a separate output directory for each experiment. The reproduction command stages all outputs and restores previous files if ordinary publication fails. Concurrent writers to the same directory are unsupported. `make reproduce` writes to `runs/reproduction/` and runs the tests afterwards. `make train` and `make benchmark` use `runs/local/`. `make clean` removes those generated runs and build caches while preserving the checked-in model and reference results.

## Reuse the saved model

The checked-in `artifacts/world_model.pt` supports an immediate recovery trace after installation:

```bash
python scripts/smoke_demo.py --perturbation compound_slip_block --seed 13
```

To rerun all configured episodes with that checkpoint:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/reproduce.py \
  --config configs/reproduce.json --model artifacts/world_model.pt \
  --out runs/replay
```

A replay records the supplied checkpoint's hash and marks its origin as reused. It leaves training metrics unspecified because the supplied file alone does not establish how it was trained. The original reference's training evidence remains in its own provenance and metrics files.

For a shorter, 56-episode evaluation:

```bash
python scripts/run_benchmark.py --model artifacts/world_model.pt \
  --seeds 2 --episodes-per-condition 1 --out runs/quick
```

The benchmark command writes results directly under `--out`. The training command instead uses it as a root for `artifacts/` and `results/`; explicit `--output` and `--metrics` paths override those defaults. Both training and benchmark commands accept `--config`, with explicit command-line settings taking precedence.

## Evidence files

| Path in a complete reproduction | Contents |
|---|---|
| `artifacts/world_model.pt` | Model weights, detector threshold, architecture, precision and action encoding. |
| `results/world_model_metrics.json` | Training settings, complete episode split, calibration MSE and threshold settings. |
| `results/episodes.csv` | Every episode's identity, outcomes and detection/recovery counters. |
| `results/episodes.jsonl` | The same episode counters plus action transitions, injected events, detections, selected candidates and their outcomes. |
| `results/summary.csv`, `results/summary.json` | Per-method/per-condition summaries with pooled detection counts and ratios. |
| `results/provenance.json` | Effective settings, checkpoint/source SHA-256 hashes, runtime versions, thread settings, deterministic mode and evaluation order. |
| `results/success_by_perturbation.svg` | Task completion by condition. |
| `results/success_within_7_actions.svg` | Completion within the configured reference reporting threshold. |

For another reporting threshold, the second figure's filename uses that number. The summary's `success_within_action_budget` uses the requested threshold; `success_within_7_actions` remains a compatibility field that always uses seven.

## Recorded runtime

The checked-in reference ran on Linux x86-64 with Python 3.12.13, NumPy 2.3.5 and PyTorch 2.10.0+cpu. PyTorch recorded one computation thread and nine inter-operation threads, with deterministic algorithms enabled and warning-only mode disabled. The complete platform string and package versions are in `results/provenance.json`. The recorded calibration MSE is 0.004946196 and residual threshold is 0.211759388, computed from 601 transitions across the 140 calibration episodes; training uses 2,371 transitions from the other 560 episodes.

## Seeds and interpretation

Training environments use a uint64 seed derived from `SeedSequence([training_seed, episode_id, 0])`. Evaluation uses `SeedSequence([root_seed, repetition, 0x4556414C])`. This separates the nominal collection stream from evaluation and avoids arithmetic-offset overlap. Root seeds must be distinct integers in `[0, 2**32)`; derived environment seeds are stored with each episode.

Each root-seed/repetition pair supplies the same initial geometry across conditions and methods. Memory starts empty for each root seed and follows the recorded condition order. Comparisons therefore use paired, partly dependent observations. The reference's single training seed and ten evaluation root seeds do not establish generalisation to other robot tasks or distributions.

Training enables deterministic PyTorch algorithms and fixes Python, NumPy and PyTorch seeds. Exact weights and residuals can still differ across package versions, platforms and thread settings. The saved checkpoint enables replay of the recorded model, while source hashes and runtime metadata identify the implementation used for each run.

Input and output validation reject duplicate seeds, malformed settings and nonfinite predictions or metrics. A failed inference stops before the next environment action. Undefined precision or recall is stored as JSON `null`, keeping output parseable by standard JSON readers.
