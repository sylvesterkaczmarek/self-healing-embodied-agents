# Independent release review

## Starting point

The review used a fresh checkout and environment at candidate
`a414e3046f2414ef56611a9de0ce1cc3fd31575a`, on the existing PR #2. The original
382 tests passed. Before modifying the candidate, a separately written 30-case
probe reproduced 27 failures and three passing controls. The untouched probe and
failure logs remain in the local handoff. All cases pass after repair; additional
valid-policy and provenance controls are included in the committed review suite.

The expanded independent suite passes 48 cases; the complete source suite passes
430 tests. The unchanged original 30-case probe and the original 26-case
implementation audit also pass. The numerical reference checks both archived
study splits. Final exact-head CI and public-package verification are recorded
in the release workflow and PR.

This is a separate verification pass, not an external third-party certification.

## Findings and repairs

Complete recovery records could attribute success to a plan containing different
executed actions, or report completion before any recovery action. Outcome names
and counts alone were insufficient. The reference trace verifier now follows the
selected action queue, observed action order, monitor identity and terminal reason.
A path-clearing candidate may legitimately complete before the overall task does.
A task reaching its goal may end a longer recovery sequence early.

The study verifier formerly checked calibration counts only for internal range and
partition consistency. A different split or invented sample count could survive
when its duplicate provenance record was updated. It now reconstructs the declared
fixed collector's episode partition and sample counts, and checks the supported
learning rate, split semantics and calibration procedure. A separate numerical
reference checks the actual saved-weight calibration predictions and threshold.

Some unvalidated source/runtime fields and mutated detector/recovery/memory settings
could pass. Complete provenance now requires typed runtime fields, the declared
study protocol and safe source paths containing all originally required modules.
Runtime controls are checked before resetting the environment. Historical unknown
package-version metadata remains explicitly unknown.

## Independent numerical reference

`scripts/independent_reference.py` has no imports from the runtime, experiment
runner or evidence verifier. It independently constructs nominal geometry and
transition data, composes the three dense layers and SiLU activations from saved
tensors, rederives alarm decisions, and computes model/history paired effects and
the crossed percentile bootstrap using separate matrix arithmetic.

It checks all development and evaluation models. The evaluation split includes
2,960 calibration predictions, 75,602 monitor-transition predictions, 13,266
recorded alarm residuals and 60 paired estimates with their intervals. All monitor
decisions match. Linear operations and the NumPy/PyTorch dependencies remain shared;
this is not independently implemented numerical infrastructure.

Recorded float32 residuals and saved-model calibration values allow 2e-6 numerical
rounding tolerance, while alarm choices must agree exactly. Paired estimate/interval
comparison uses 1e-12 tolerance. Source hashes and file identities remain exact.
These are numerical compatibility tolerances, not physical sensing tolerances.

## Reproduction and evidence preservation

An untouched-candidate full evaluation retrains all five models and executes all
14,700 evaluation plus 2,940 fitting episodes. The repaired release repeats that
same fixed design. Complete episode records, models, calibration, fitted memory
and numerical summaries are compared, while elapsed times, archive compression
metadata and changed source/runtime provenance are excluded from byte-identity claims.
All original 43 result/checkpoint/configuration files remain unchanged.

No favourable subset or replacement seeds are introduced. Model seeds and complete
histories remain the resampling units. Continuous replanning/postcondition controls
still match learned task performance, and tested memory variants still add actions.

## Commands

```sh
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
python -m ruff check src scripts tests
python -m scripts.verify_comparison results/comparison-v1/evaluation --replay-actions
python scripts/independent_reference.py results/comparison-v1/evaluation
python scripts/independent_reference.py results/comparison-v1/development
python -m scripts.run_comparison --config configs/comparison.json --split evaluation --out /tmp/healing-release-replay-new
python -m scripts.verify_comparison /tmp/healing-release-replay-new --replay-actions
python -m build
python -m twine check --strict dist/*
python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz --report /tmp/healing-release-install.json
```

Use a previously absent output path. The installed-package verifier uses an isolated
non-editable environment and a working directory outside the source checkout.
CI covers Linux Python 3.11-3.14 and macOS 15. Publication repeats the exact-source
checks and then downloads and tests the public artifacts. Package-index publication
is outside this GitHub release because no PyPI publisher access was verified.

## Limits

Action replay uses the same simulator. The numerical reference is separate but
shares numerical libraries. Neither authenticates a producer or proves real-robot
behavior. The method uses ideal observations and guaranteed symbolic repair skills.
The original held-out split has been examined during audit and is not newly unseen.
All uncertainty and negative conclusions remain bounded to this small pilot.

Primary dependency documentation:

- https://docs.pytorch.org/docs/stable/notes/randomness.html
- https://docs.pytorch.org/docs/stable/notes/serialization.html
