# Evidence verification

## Commands

From an installed source checkout:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.verify_comparison results/comparison-v1/development --replay-actions
python -m scripts.verify_comparison results/comparison-v1/evaluation --replay-actions
python -m scripts.showcase --model artifacts/world_model.pt --out /tmp/healing-demo-new
```

`verify_comparison` returns a JSON PASS only after validating the full declared
row sequence, counters, fitting-memory snapshots, checkpoint/calibration bindings,
source fingerprint syntax, and recomputed summaries. `--replay-actions` additionally
executes the recorded action sequences in new simulator instances. Omitting the
option does not claim action replay; the reported action count is zero.

The checker limits compressed trace size, decompressed data, line length and JSON
structure. It rejects redirected files, duplicate JSON keys, nonfinite values,
missing/extra study files and mismatched trace/config hashes. File fingerprints
establish internal consistency, not the identity or honesty of the producer.

## Numerical contracts

Counts, types, identities, settings and hashes are exact. Recomputed floating
summaries use relative tolerance 1e-11 and absolute tolerance 1e-12. Simulator
float32 observations and fault magnitudes permit 2e-7 absolute/relative roundoff.
These allowances are numerical comparison tolerances, not physical sensor noise
models or a guarantee of cross-platform bitwise execution.

Onset matching is one-to-one and bounded by the declared detection window. The
study uses one step. `episode_metrics(..., detection_window=N)` permits an explicit
alternative when analysing separately configured agents. Conditional detection
ratios are null when their denominators are zero. Recovery-sequence completion
remains distinct from completing the object-moving task.

## Reproduce training and decisions

Use absent output paths; the CLI creates them and the Python API requires a
pre-created empty directory. Neither may replace earlier comparison evidence.

```bash
python -m scripts.run_comparison --config configs/comparison.json --split development --out /tmp/healing-development-new
python -m scripts.run_comparison --config configs/comparison.json --split evaluation --out /tmp/healing-evaluation-new
python -m scripts.verify_comparison /tmp/healing-evaluation-new --replay-actions
```

Full execution retrains models and runs the policy; consistency verification alone
does neither. Recorded compressed bytes may differ because gzip headers contain
time/path information, even when decoded records are exactly equal. Compare the
complete records and model tensors, and separately report elapsed time, checkpoint
container hashes and source/runtime provenance.

## Compatibility and trust

Unknown configuration/checkpoint fields and duplicate JSON keys now fail. Existing
schema-0 and schema-1 model checkpoints remain inference artifacts; loading is not
an exact optimizer/training restart. Calibration metadata with specified episode
partitions must be complete and disjoint. Zero episode counts continue to denote
legacy unspecified partitions, not proven train/calibration separation.

Use trusted local model files only. PyTorch restricted weights-only loading does
not eliminate denial-of-service or every memory-safety risk. The file bounds here
are not a general serialization sandbox. PyTorch guidance:

- https://docs.pytorch.org/docs/stable/notes/serialization.html
- https://docs.pytorch.org/docs/stable/notes/randomness.html

No real robot, credentials, human participant or paid inference service is used.
The adapter sketches remain unimplemented integration points. Fresh installed-wheel
verification runs from a separate directory so source scripts cannot mask missing
packaged modules. The recorded runtime requirements support advisory checking;
they are not a hash-pinned dependency lock or supply-chain assurance.
