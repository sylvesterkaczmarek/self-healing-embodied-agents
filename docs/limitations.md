# Limitations

The benchmark uses a two-dimensional symbolic tabletop with instantaneous discrete skills. It contains no rigid-body dynamics, contact forces, collision geometry, camera model, continuous controller or hardware. Workspace distances have arbitrary units. Task completion here provides evidence about this recovery loop's behaviour on synthetic cases.

Observations expose object and target coordinates and Boolean task state directly. Occlusion hides visibility for a fixed number of reads, and an explicit re-observation restores it immediately. Path clearing always works. These are controlled recovery assumptions that require separate validation in a physical system.

Recovery candidates and diagnoses are hand-written for a small known fault taxonomy. The symbolic planner has access to the testbed's exact nominal skill contract. It does not forecast future injected faults, and its state estimate can itself be stale or incomplete. Diagnosis labels are not separately scored for accuracy.

The learned model sees nominal task trajectories, with some re-observation examples. Recovery states and many action combinations are outside that training distribution. Residuals mix coordinate errors and Boolean-flag errors with equal weights, and the threshold is an empirical heuristic. Changing workspace scale, observation units or the action representation requires renewed training and calibration.

Detection precision and recall include explicit action failures. They do not isolate the contribution of the learned model. Task-performance differences also include differences in recovery candidate selection, so comparing self-healing with reactive replanning cannot attribute an improvement entirely to residual-based detection. Fault matching depends on a one-action timing window and scores each injected step once.

The seven-action result is a reporting cutoff applied to episodes executed with a 24-action cap. It is not a separate experiment with a seven-action planning horizon. Mean actions include unsuccessful episodes, including open-loop runs that stop after four actions.

The reference uses one model-training seed and a small set of paired evaluation geometries. Repetitions share geometry across conditions and methods, and the memory variant shares history within each root seed. Reported episode spread is descriptive; uncertainty across training seeds, condition orders, tasks and fault distributions remains unmeasured.

Recovery memory rewards completion of a candidate sequence, which can be a short sensing or clearing action. Candidate completion and task completion are separate outcomes. The fixed-order online ablation does not establish that memory helps on unseen tasks or arbitrary failure sequences.

The files under `adapters/` define protocol sketches. Runnable ManiSkill and LeRobot integrations, vision-language-action policy comparisons and physical-robot safety validation remain future work.
