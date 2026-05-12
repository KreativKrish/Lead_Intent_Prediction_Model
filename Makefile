.PHONY: help install install-dev test test-unit test-integration lint format clean docker-build docker-up docker-down train api

help:
	@echo "Lead Intent Prediction Model - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install              Install production dependencies"
	@echo "  make install-dev          Install dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make train                Train model locally"
	@echo "  make api                  Start API server"
	@echo "  make docker-up            Start all services (docker-compose)"
	@echo "  make docker-down          Stop all services"
	@echo ""
	@echo "Testing:"
	@echo "  make test                 Run all tests"
	@echo "  make test-unit            Run unit tests"
	@echo "  make test-integration     Run integration tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint                 Lint code (ruff, mypy)"
	@echo "  make format               Format code (black, isort)"
	@echo "  make clean                Remove cache files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python -m src.training.trainer

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

test: test-unit test-integration

test-unit:
	pytest tests/unit -v --cov=src --cov=api --cov-report=html

test-integration:
	pytest tests/integration -v

lint:
	ruff check .
	mypy src api --ignore-missing-imports

format:
	black src api tests
	isort src api tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	rm -rf build dist *.egg-info

requirements-check:
	pip list --outdated

freeze:
	pip freeze > requirements.lock
