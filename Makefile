.DEFAULT_GOAL := help

# Configuration variables
UV ?= uv
BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 8501

.PHONY: help install backend frontend test test-unit test-e2e lint format seed docker-up docker-down docker-logs clean

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies including development tools
	$(UV) sync --extra dev

backend: ## Run FastAPI backend server
	$(UV) run uvicorn src.main:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT)

frontend: ## Run Streamlit frontend application
	$(UV) run streamlit run frontend/ui.py --server.port $(FRONTEND_PORT)

test: ## Run full test suite
	$(UV) run pytest

test-unit: ## Run unit tests (excluding E2E)
	$(UV) run pytest tests/ --ignore=tests/e2e

test-e2e: ## Run end-to-end tests
	$(UV) run pytest tests/e2e

lint: ## Check code formatting and linting
	$(UV) run flake8 src tests frontend
	$(UV) run black --check src tests frontend
	$(UV) run isort --check src tests frontend

format: ## Automatically format code
	$(UV) run black src tests frontend
	$(UV) run isort src tests frontend

seed: ## Seed database with test datasets
	$(UV) run python scripts/seed_data.py

docker-up: ## Build and start all services via Docker Compose
	docker compose up --build -d

docker-down: ## Stop all Docker Compose services
	docker compose down

docker-logs: ## Follow Docker Compose service logs
	docker compose logs -f

clean: ## Remove Python cache files and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf dist build *.egg-info
