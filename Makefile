PYTHON ?= python3
PROFILE ?= ci
DATA_DIR ?= data/generated/$(PROFILE)

.PHONY: install data validate classical unsupervised deep vision nlp graph tracking api test lint notebooks reproduce

install:
	$(PYTHON) -m pip install -e '.[full,dev]'

data:
	$(PYTHON) scripts/prepare_data.py --profile $(PROFILE) --output $(DATA_DIR)

validate:
	$(PYTHON) scripts/validate_data.py --data-dir $(DATA_DIR)

classical:
	$(PYTHON) scripts/run_classical.py --data-dir $(DATA_DIR) --profile $(PROFILE)

unsupervised:
	$(PYTHON) scripts/run_unsupervised.py --data-dir $(DATA_DIR) --profile $(PROFILE)

deep:
	$(PYTHON) scripts/run_deep.py --data-dir $(DATA_DIR) --profile $(PROFILE)

vision:
	$(PYTHON) scripts/run_vision.py --data-dir $(DATA_DIR) --profile $(PROFILE)

nlp:
	$(PYTHON) scripts/run_nlp.py --data-dir $(DATA_DIR) --profile $(PROFILE)

graph:
	$(PYTHON) scripts/run_graph.py --data-dir $(DATA_DIR) --profile $(PROFILE)

tracking:
	$(PYTHON) scripts/run_tracking_demo.py --data-dir $(DATA_DIR)

api:
	ROOKIE_LAB_DATA_DIR=$(DATA_DIR) $(PYTHON) -m uvicorn rookie_lab.serving.api:app --reload

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests notebooks

notebooks:
	$(PYTHON) scripts/build_learning_notebooks.py --check notebooks
	$(PYTHON) scripts/build_system_notebooks.py --check notebooks

reproduce:
	$(PYTHON) scripts/reproduce_all.py --profile $(PROFILE)
