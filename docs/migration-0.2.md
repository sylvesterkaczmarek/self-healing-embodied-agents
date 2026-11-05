# Migration to 0.2.0

Valid schema-0 and schema-1 inference checkpoints remain supported. No archived
result or checkpoint is migrated or overwritten. Configuration values and policies
remain unchanged for valid runs.

The complete-trace verifier now requires the reference event schema and validates
executed recovery actions and chronological monitor decisions. The low-level
metric API retains its partial-trace mode; callers claiming complete reference
execution must supply the full fields. Custom event schemas need an explicit
adapter rather than silently being treated as reference evidence.

The study verifier recomputes the fixed training protocol's episode partition and
sample counts. It rejects metadata describing an unimplemented calibration rule.
It preserves a null historical package version as unknown, rather than inventing
an installed version. Unkeyed fingerprints establish consistency, not authenticity.

All mutable detector, recovery and memory controls are revalidated before reset.
Use a fresh output directory for every comparison. Legacy reference reproduction
commands keep their documented staged publication behavior.

Restricted PyTorch loading reduces executable-content exposure but is not a sandbox
for arbitrary malicious files. Use trusted local/released study assets. Cross-platform
bitwise training equality is not promised. The separately implemented reference uses
NumPy/PyTorch primitives, including linear operations and the SiLU activation.
