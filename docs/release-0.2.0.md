# Self-Healing Embodied Agents 0.2.0

A verified release of the existing public monitoring/recovery benchmark and
controlled comparison, with stricter experimental evidence and execution contracts.

## Changes

- Recovery completion now follows the recorded selected action sequence. Interrupted
  outcomes require a corresponding observation/alarm; task completion and partial
  candidate completion remain distinct.
- Complete traces reject unsupported events/fields and detector/policy mismatches.
- Calibration records are checked against their fixed collector, sample counts,
  episode-level split and threshold procedure. Complete provenance has typed runtime
  settings, supported source paths and the required simulation/experiment modules.
- Mutated detector, recovery and memory settings are rejected before environment reset.
- Separate reference calculations check nominal data, direct-tensor model inference,
  every learned-monitor decision and the crossed bootstrap comparisons.

## Evidence

The original reference and pilot files are preserved. All evaluation and fitting
records, trained models, fitted memories and numerical summaries remain unchanged
in same-environment candidate/release reproductions. The original held-out split
was retained; this is not a newly unseen evaluation dataset.

Simple observation-based controls still match learned monitoring and ranked recovery
in task outcomes. Tested memory variants add actions. The benchmark has five evaluation
model seeds, fourteen histories and shared deterministic baselines. Replays and
per-action predictions do not increase the number of independent observations.

## Distribution

The GitHub release contains the wheel, source distribution, exact-commit source
archive, checksums and review documentation. No PyPI upload is performed. Source
archives include the reference assets; installed commands accept explicit paths.
Use the independent-review and migration documents for full verification details.

This is not physical-robot testing, human-influence evidence or safety certification.
