# Experiments

The default benchmark compares four conditions under identical deterministic seeds:

- open-loop execution
- reactive replanning after explicit action failure
- learned-divergence self-healing
- learned-divergence self-healing with episodic recovery memory

The benchmark covers nominal execution plus six controlled perturbation families. Run it with `make benchmark`.
