VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.DEFAULT_GOAL := help

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[dev]"

.PHONY: install
install: $(VENV)/bin/activate ## Install the core harness and dev tools

.PHONY: install-all
install-all: $(VENV)/bin/activate ## Install every extra (dense models, BEIR loaders, API clients)
	$(PIP) install --quiet -e ".[dev,dense,beir,api]"

.PHONY: demo
demo: install ## Run the offline smoke benchmark -- no network, no API keys, ~5 seconds
	$(PY) -m bakeoff run configs/smoke.yaml --out results/smoke
	$(PY) -m bakeoff report results/smoke --format markdown

.PHONY: bench
bench: install-all ## Run the real BEIR benchmark (downloads models and datasets)
	$(PY) -m bakeoff run configs/beir-v1.yaml --out results/beir-v1
	$(PY) -m bakeoff report results/beir-v1 --format markdown --write-readme

.PHONY: test
test: install ## Run the test suite
	$(VENV)/bin/pytest -q

.PHONY: lint
lint: install ## Lint and format-check
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf $(VENV) .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'
