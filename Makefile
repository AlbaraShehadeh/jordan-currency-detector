.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck test check clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv, install deps, install git hooks
	uv sync
	uv run pre-commit install

format: ## Auto-format and auto-fix lint
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint + format check (no changes)
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## Static type check
	uv run mypy

test: ## Run tests with coverage
	uv run pytest --cov

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and build outputs
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
