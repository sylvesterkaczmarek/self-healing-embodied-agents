# Reproducibility

The default benchmark is designed to be reproducible on CPU.

- NumPy, Python and PyTorch seeds are fixed.
- `torch.use_deterministic_algorithms(True)` is requested during model training.
- Perturbations are generated from per-episode NumPy generators.
- Benchmark outputs are saved as CSV and JSON.
- The checked-in result figure is generated from those machine-readable outputs.

Exact floating-point values can vary across PyTorch versions and platforms. Re-run `make reproduce` to regenerate the model and benchmark results in the local environment.
