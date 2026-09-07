.PHONY: install train benchmark test demo reproduce clean

install:
	python -m pip install -e ".[dev]"

train:
	python scripts/train_world_model.py --out runs/local

benchmark:
	python scripts/run_benchmark.py --model runs/local/artifacts/world_model.pt --out runs/local/results

test:
	pytest -q

demo:
	python scripts/smoke_demo.py

reproduce:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/reproduce.py --out runs/reproduction
	pytest -q

clean:
	rm -rf runs .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
