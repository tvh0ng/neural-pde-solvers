.PHONY: install lint format test test-pinn test-surrogate notebooks clean

install:            ## editable install with every extra
	pip install -e ".[all]"

lint:
	ruff check .
	ruff format --check src tests scripts

format:
	ruff check . --fix
	ruff format src tests scripts notebooks

test: test-pinn test-surrogate   ## PyTorch and TensorFlow suites run in separate processes

test-pinn:
	pytest tests/test_pinn.py

test-surrogate:
	pytest tests/test_surrogate.py

notebooks:          ## execute all notebooks with short runs (as in CI)
	for nb in notebooks/pinn/*.ipynb; do PINN_QUICK=1 jupyter nbconvert --to notebook --execute --output-dir /tmp/nb "$$nb"; done
	cd notebooks/surrogate && SURROGATE_QUICK=1 jupyter nbconvert --to notebook --execute --output-dir /tmp/nb train_surrogate.ipynb

clean:
	rm -rf results .pytest_cache .ruff_cache build dist src/*.egg-info
