# What this repository does not claim

- The tabletop environment is a controlled manipulation testbed, not a rigid-body physics simulator.
- Recovery success in this benchmark is not evidence of deployment safety on real robots.
- The learned transition model is intentionally small and is not presented as a general-purpose robotics world model.
- The failure taxonomy is incomplete and the perturbations are synthetic.
- Counterfactual skill rollouts use known task semantics, which will not be available in this form for every real system.
- The benchmark does not establish state-of-the-art performance against large vision-language-action policies.

The purpose is to make failure detection, diagnosis, recovery selection, memory and evaluation inspectable in one reproducible system.
