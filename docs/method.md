# Method

## State and execution

The tabletop task moves one object to a target using six discrete skills. The observation vector contains nine values, in this order: end-effector position (x, y), object position (x, y), target position (x, y), holding, object visibility and path obstruction. Positions use arbitrary workspace units; flags are represented as zero or one. These values come from the testbed's state, with injected observation faults.

The default workspace spans 0.05 to 0.95 on both axes. Grasping requires visibility and an end-effector/object distance of at most 0.06. Placing requires holding the object and a final object/target distance of at most 0.08. A carried object follows a successful move to the target. Actions consume one step, including failed actions and sensing or path-clearing actions. Successful tasks and exhausted step budgets are absorbing: further calls do not change state or increment the counter. Agent execution stops at the smaller of its own limit and the environment limit, both 24 by default.

`nominal_transition` defines the common skill contract used by the environment and symbolic recovery model. Custom grasp and placement radii therefore apply to both execution and planning. The environment adds faults around that transition; the symbolic model forecasts fault-free execution from the supplied observation.

## Learned transition model

A three-layer MLP with two 64-unit hidden layers and SiLU activations predicts the nine next-state values from the current state and a six-way one-hot action. It is trained on nominal trajectories using mean squared error and AdamW. Nominal data include the task sequence and an optional initial re-observation. They do not cover every recovery state or action combination.

The split is made over episode IDs: 80% of episodes train the model and the remaining episodes calibrate the detector. All transitions from a trajectory remain in one partition. The saved metrics include both lists of episode IDs and sample counts.

For predicted state \(\hat{x}_{t+1}\) and observed state \(x_{t+1}\), the residual is

\[
r_t = \sqrt{\frac{1}{9}\sum_{j=1}^{9}(\hat{x}_{t+1,j}-x_{t+1,j})^2}.
\]

The threshold is the larger of 0.035 and 1.35 times the empirical 99.5th percentile of nominal calibration residuals. This is a calibration heuristic with no finite-sample false-alarm guarantee. Calibration MSE measures the same held-out episodes used to set the threshold; detector performance is assessed separately in the benchmark. Continuous coordinates and flags contribute with equal weight, so changes in feature units or workspace scale require renewed calibration.

An explicit action failure or an above-threshold residual triggers a detection. Successful `REOBSERVE` and `CLEAR_PATH` actions do not trigger residual-based detections, because their corrective changes are expected. Model inference still runs for these actions, and invalid predictions stop execution before another action is taken.

## Recovery selection and outcomes

A hand-written diagnosis maps the observed discrepancy to perception loss, path obstruction, grasp loss, object movement, execution failure or generic state divergence. It produces a small set of explicit candidate action sequences.

Nominal replanning skips a move only when its destination is already reached exactly: approach is omitted when end-effector and object coordinates are equal, and transport is omitted when the object is already at the target after acquisition. Grasping and placing still execute. Re-observation and path clearing remain explicit actions. This avoids repeatedly selecting an already completed move after an alarm, while preserving the simulator's required task steps.

Candidate scoring uses

\[
3\,\mathbf{1}[\text{predicted task success}]
- \lVert p_{\mathrm{object}}-p_{\mathrm{target}}\rVert_2
- 0.07\,N_{\mathrm{executed}}
+ 0.8\,(q-0.5),
\]

where the memory term is omitted for the variant without memory. Rollouts stop at task success or the remaining execution budget, so actions after completion or beyond the budget do not earn a completion bonus or add predicted cost. Equal scores retain candidate order.

A recovery attempt starts when a candidate is selected with time left to execute it. It completes when its sequence finishes without a new detection, or when the task succeeds. A fresh detection interrupts the current candidate; exceptions and an exhausted budget also finish it unsuccessfully. Every started attempt receives one recorded outcome. A successful one-step re-observation can therefore count as a completed recovery even while the object still needs moving. `recovery_successes` counts completed candidates; the separate `success` field records task completion.

The memory variant uses the posterior mean \(q=(s+1)/(n+2)\), where \(s\) and \(n\) count successful and total candidate attempts for a diagnosis/candidate pair. Each benchmark root seed starts with empty memory. It persists across that seed's fixed sequence of conditions and repetitions. This is an online, order-dependent ablation using candidate completion as its reward.

## Injected faults

| Condition | Trigger and effect |
|---|---|
| `none` | Nominal execution. |
| `grasp_slip` | After the first successful target transport while holding, release and displace the object. |
| `object_displacement` | Displace the object immediately before the first grasp attempt. |
| `transient_occlusion` | At the first object-approach attempt, obscure the object for two observation reads; explicit re-observation restores visibility immediately. |
| `blocked_path` | Block the first target transport attempted while holding until a path-clearing action. |
| `stale_observation` | Displace the object before the first grasp attempt and return the last observation once, with the current executed-step index. |
| `compound_slip_block` | First obstruct transport; inject grasp slip only after a later transport succeeds. |

Each fault is injected at most once per episode. Realised injections depend on the actions a policy reaches. For example, a policy that never clears an obstruction never reaches the later slip in the compound condition. The same episode seed supplies paired initial geometry and fault randomness across methods, without forcing every method to experience the same realised event sequence.

## Evaluation

Task success, executed actions and completed candidate attempts are reported separately. The default seven-action figure counts episodes that finish successfully in at most seven executed actions. Episodes actually run with a 24-action cap, and recovery planning uses that execution cap. Changing the reporting `action_budget` changes the summary and figure threshold; it does not change the execution policy's limit.

Ground-truth detection units are unique executed steps containing an injected fault. Each detection matches the oldest unmatched fault within the configured window, which by default includes the current and immediately preceding action. A fault step and a detection can each be matched once. Unmatched detections are false positives, including later alarms about an already matched or out-of-window fault. This timing-based score does not assess diagnosis accuracy or prove the alarm's causal origin.

Precision is pooled matched detections divided by pooled detections. Recall is pooled matched detections divided by pooled injected-fault steps. Explicit action failures contribute to these scores alongside learned residual alarms. Undefined values are JSON `null` and empty CSV cells. The open-loop and reactive baselines do not emit scored detection events, although the reactive baseline responds to explicit action failures. Their precision is undefined, and recall is zero where faults occur, because no events enter this scoring interface. Reactive failure handling is recorded separately in its action/recovery trace.

`success_std` is the descriptive population standard deviation of episode success indicators. Reused initial geometries, paired policies and memory histories make episode rows dependent; that standard deviation is not a confidence interval. The reference uses one trained model, so it does not measure variation over model-training seeds.

## Controlled comparisons

`configs/comparison.json` defines a separate, frozen pilot. The historical reference
above remains tied to commit `bfbc77386b5e37f20a3c257dc606c7a0104bba0e`.
The additional controls share the same ideal-state observations, skill execution
and action limits. Fault labels are used for evaluation only.

| Method | Monitor | Recovery selection |
|---|---|---|
| `always_replan` | None; replan after every observation | First nominal action |
| `failure_nominal` | Explicit execution failure | Nominal plan |
| `postcondition_nominal` | Execution failure or observed skill endpoint mismatch | Nominal plan |
| `analytical_nominal` | Execution failure or nominal analytical residual | Nominal plan |
| `learned_nominal` | Execution failure or learned residual | Nominal plan |
| `self_healing` | Same learned monitor | Ranked candidates |
| `self_healing_memory` | Same learned monitor | Ranked candidates with historical memory |
| `self_healing_task_memory` | Same learned monitor | Ranked candidates with task-outcome memory |

Postconditions check visibility, holding and coordinate endpoints. A tolerance of
eight float32 epsilons at coordinate scale handles numerical rounding. The
analytical control uses the simulator's known nominal dynamics and the same
numerical tolerance for its residual. Neither control models physical sensor
noise. Learned and analytical residuals retain the historical suppression on
successful re-observation and clearing actions; postconditions also check those
actions' observable endpoints.

Detection traces identify `execution_failure`, `residual_only`,
`postcondition_only` and `analytical_only`. Explicit failed actions are counted
for every method, including continuous replanning, which emits no detection
alarms. Matched and unmatched alarms retain the historical onset-window meaning.
An unmatched alarm does not establish that an intervention was unnecessary.
`nominal_planner_calls` counts calls to the nominal plan interface; ranked
selections are counted separately. These counters are not total planning compute.

The experimental task-outcome memory assigns each selected candidate terminal
credit `task_success / (1 + actions_since_selection)`. It uses the same prior and
ranking weight as historical memory. Interrupted candidates also receive terminal
credit, while budget exhaustion and execution errors receive zero. This shared
outcome credit is not a causal estimate of each candidate's contribution. The
historical memory and default agent remain available unchanged in behaviour.

Cold online memories start empty for each model/history pair. Two additional
`_frozen` variants fit their memory on a disjoint geometry root, then evaluate a
copied memory with updates disabled. Thus cold and frozen variants differ in both
prior experience and permission to update. Fourteen Williams orders balance all
seven conditions' positions and directed adjacent pairs. Each order runs three
repetitions per condition; fit and evaluation histories use the same order.

The held-out pilot crosses five independent model-training seeds with fourteen
histories, giving 42 evaluation geometry seeds reused across conditions and
methods. The 14,700 evaluation rows are not independent replications. Paired
95% percentile intervals resample whole model seeds and histories independently,
retaining the repetitions, order and fit/evaluation relationship. Five training
seeds provide limited uncertainty information; this remains a pilot. Development
uses separate model and geometry seeds, with a descriptive 0.25-action interval
half-width target and no sample-size adjustment after evaluation.

Primary outcomes are completion under the actual 24-action limit and capped
action cost: executed steps on success, 24 on failure. Raw action counts and
completion curves for reporting cutoffs 1 through 24 are also retained. The curves
do not represent separately executed planning horizons. The testbed provides no
physical collision, force or damage measurements, and this study makes no
wall-clock efficiency claim.

Run each command into a new directory:

```bash
python scripts/run_comparison.py --reference-audit --out runs/reference-audit
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_comparison.py \
  --split development --out runs/comparison-development
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_comparison.py \
  --split evaluation --out runs/comparison-evaluation
```

Each study directory contains its configuration, model checkpoints and calibration
partitions, source/checkpoint hashes, fitted-memory snapshots, compressed complete
episode traces and an auditable summary. Existing output paths are refused. A run
whose source files change or whose frozen memory mutates cannot publish a completed
summary. The source archive includes the comparison runner and saved evidence.

The study records the original local protocol and implementation commits. Their
published equivalents have identical Git trees and different commit metadata;
`study.json` maps both pairs. The compact `source.bundle` preserves the original
commits. After cloning this repository, recover them with:

```bash
git fetch results/comparison-v1/source.bundle \
  refs/heads/archive/comparison-source:refs/heads/comparison-recorded-source
```

The bundle requires the historical reference commit already present in the clone.
