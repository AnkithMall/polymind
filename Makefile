.PHONY: help install install-dev update upgrade test test-quick lint format check clean build publish

# Default target
help: ## Show this help message
	@echo "Polymind - Hardware-aware LLM runtime"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Installation
# ============================================================

install: ## Install polymind in production mode
	uv sync --no-dev
	@echo "✓ Installed polymind"

install-dev: ## Install polymind with dev dependencies
	uv sync
	@echo "✓ Installed polymind (dev mode)"

update: ## Update all dependencies
	uv lock --upgrade
	uv sync
	@echo "✓ Dependencies updated"

upgrade: update ## Alias for update

# ============================================================
# Testing
# ============================================================

test: ## Run all tests with verbose output
	uv run pytest tests/ -v

test-quick: ## Run tests excluding slow/network tests
	uv run pytest tests/ -v -m "not slow" --ignore=tests/test_model.py

test-hardware: ## Run hardware tests only
	uv run pytest tests/test_hardware.py -v

test-model: ## Run model tests only
	uv run pytest tests/test_model.py -v

test-runtime: ## Run runtime tests only
	uv run pytest tests/test_runtime.py -v

test-config: ## Run config tests only
	uv run pytest tests/test_config.py -v

test-cov: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=polymind --cov-report=term-missing

test-html: ## Run tests and generate HTML coverage report
	uv run pytest tests/ -v --cov=polymind --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/"

# ============================================================
# Code Quality
# ============================================================

lint: ## Run linting checks
	uv run ruff check src/ tests/
	@echo "✓ Linting passed"

lint-fix: ## Run linting and auto-fix issues
	uv run ruff check --fix src/ tests/
	@echo "✓ Linting fixes applied"

format: ## Format code with ruff
	uv run ruff format src/ tests/
	@echo "✓ Code formatted"

format-check: ## Check code formatting without changing files
	uv run ruff format --check src/ tests/
	@echo "✓ Formatting check passed"

typecheck: ## Run type checking with pyright/mypy
	@if command -v pyright > /dev/null 2>&1; then \
		pyright src/; \
	elif command -v mypy > /dev/null 2>&1; then \
		mypy src/; \
	else \
		echo "No type checker found. Install pyright or mypy."; \
		exit 1; \
	fi

check: lint format-check ## Run all pre-commit checks (lint + format)
	@echo "✓ All checks passed"

# ============================================================
# CLI Commands
# ============================================================

.PHONY: cli cli-help cli-version cli-config cli-hardware cli-model cli-runtime tui

cli: ## Run polymind CLI (use ARGS="..." for arguments)
	uv run polymind $(ARGS)

cli-help: ## Show polymind help
	uv run polymind --help

cli-version: ## Show polymind version
	uv run polymind --version

cli-config: ## Show current configuration
	uv run polymind config show

cli-hardware: ## Run hardware scan
	uv run polymind hardware scan

cli-model-list: ## List installed models
	uv run polymind model list

cli-domain-list: ## List all domains
	uv run polymind domain list

cli-confidence: ## Show confidence scores
	uv run polymind confidence show

cli-pipeline-suggest: ## Suggest models for pipeline roles
	uv run polymind pipeline suggest

cli-pipeline-status: ## Show pipeline status
	uv run polymind pipeline status

cli-model-scan: ## Scan for models
	uv run polymind model scan

tui: ## Launch the Polymind TUI
	uv run polymind tui

# ============================================================
# Build & Publish
# ============================================================

build: ## Build distribution packages
	uv build
	@echo "✓ Build complete. Check dist/"

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned"

# ============================================================
# Development
# ============================================================

.PHONY: dev shell repl

dev: install-dev ## Start development environment
	@echo "Development environment ready."
	@echo "Run 'make test' to run tests."
	@echo "Run 'make cli ARGS=\"--help\"' to test CLI."

shell: ## Start a Python shell with polymind loaded
	uv run python -c "import polymind; print(f'polymind {polymind.__version__} loaded'); import code; code.interact()"

repl: shell ## Alias for shell

# ============================================================
# Documentation
# ============================================================

.PHONY: docs-serve

docs-serve: ## Serve documentation locally (if mkdocs is installed)
	@if command -v mkdocs > /dev/null 2>&1; then \
		mkdocs serve; \
	else \
		echo "mkdocs not installed. Install with: uv add --dev mkdocs"; \
		exit 1; \
	fi
