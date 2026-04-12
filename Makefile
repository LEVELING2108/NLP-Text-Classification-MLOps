PYTHON ?= python

.PHONY: install install-dev test train serve

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

train:
	$(PYTHON) pipelines/run_train.py --config configs/config.yaml

serve:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000
