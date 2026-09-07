# Integration path

The default benchmark runs without a robotics simulator dependency. The files under `adapters/` are protocol sketches:

- `adapters/maniskill.py` sketches environment `reset` and `step` methods.
- `adapters/lerobot.py` sketches a policy action-selection method.

They do not implement runnable simulator or robot connections. The current agents also use `nominal_plan`, `config.max_steps`, `done`, `events`, `seed` and `perturbation`; a concrete environment adapter must provide those contracts or introduce an explicit replacement interface.

A physical integration needs an observation estimator, executable motion/skill controllers and independent safety constraints. Object visibility, path clearing, grasp radius and placement tolerance must reflect the real system. Replace the exact symbolic transition with a validated skill model, retrain the learned predictor on the new observations, and calibrate its residual scale.

Keep action failures, sensor observations and task completion separate in the interface. Preserve action timestamps, actual step budgets and recovery outcomes so that interruptions, stale observations and failed actions can be audited. Real fault labels also need an independent measurement process; the benchmark currently obtains them from its own fault injector.

A useful next validation would repeat the existing perturbation protocol in a rigid-body simulator while reporting model-training variation and the additional failures the simulator introduces. Hardware evaluation would then require its own motion constraints, collision checks and emergency stopping arrangements.

Upstream documentation is available for [ManiSkill](https://maniskill.readthedocs.io/en/latest/) and [LeRobot](https://huggingface.co/docs/lerobot/index).
