.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test check clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create the virtualenv and install all dependencies
	$(UV) sync --locked

lint: ## Lint and check formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Apply formatting and autofixable lint fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## Run mypy in strict mode
	$(UV) run mypy

test: ## Run the test suite with coverage
	$(UV) run pytest --cov --cov-report=term --cov-report=xml

check: lint typecheck test ## Run everything CI runs

clean: ## Remove build and cache artifacts
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
