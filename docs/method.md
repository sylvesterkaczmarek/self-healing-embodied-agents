# Method

## Problem formulation

A nominal embodied policy generates an action sequence for a manipulation task. The system maintains a learned one-step transition model trained only on nominal trajectories. After each action, it compares the predicted next state with the observed state.

A divergence event is raised when the residual crosses a threshold calibrated from held-out nominal transitions or when execution explicitly fails. The recovery layer then:

1. classifies the mismatch into a small failure taxonomy,
2. generates recovery candidates,
3. rolls each candidate forward with a deterministic counterfactual skill model,
4. scores predicted task completion, remaining goal distance, action cost, and optional recovery-memory evidence,
5. resumes execution from the selected recovery sequence.

## Why two world models

The learned transition model is used for anomaly detection because model error provides a measurable signal for unexpected state transitions. The symbolic model is used for short counterfactual recovery rollouts because the testbed has known skill semantics and this makes candidate scoring transparent and reproducible.

The separation is deliberate. It prevents the benchmark from confusing transition-prediction quality with recovery-policy quality.

## Failure taxonomy

The default testbed injects:

- grasp loss after transport begins,
- object displacement before grasp,
- temporary observation loss,
- path obstruction,
- stale state observations,
- a compound obstruction plus grasp-loss case.

## Metrics

The benchmark reports task success rate, episode length, recovery attempts, and for divergence-aware agents, detection precision and recall against known injected failures.
