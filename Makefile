.PHONY: install train benchmark test demo reproduce clean

install:
	python -m pip install -e .[dev]

train:
	python scripts/train_world_model.py

benchmark:
	python scripts/run_benchmark.py

test:
	pytest -q

demo:
	python scripts/smoke_demo.py

reproduce:
	python scripts/reproduce.py
	pytest -q

clean:
	rm -rf artifacts .pytest_cache build dist *.egg-info src/*.egg-info
