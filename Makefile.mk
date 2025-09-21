.PHONY: help install install-dev test lint format clean docker-build docker-up docker-down run benchmark

help:
	@echo "LLM Serving Framework - Available Commands"
	@echo "==========================================="
	@echo "Setup:"
	@echo "  make install        - Install core dependencies"
	@echo "  make install-dev    - Install with development tools"
	@echo "  make install-full   - Install with vLLM and all features"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "Running:"
	@echo "  make run            - Run server locally"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start Docker containers"
	@echo "  make docker-down    - Stop Docker containers"
	@echo ""
	@echo "Testing:"
	@echo "  make test-install   - Test installation"
	@echo "  make test-simple    - Run simple tests"
	@echo "  make test-client    - Test API client"
	@echo "  make benchmark      - Run benchmarks"

install:
	@echo "Installing core dependencies..."
	pip install -r requirements-core.txt
	@echo "✓ Installation complete"

install-dev: install
	@echo "Installing development dependencies..."
	pip install pytest pytest-asyncio black flake8 mypy isort
	@echo "✓ Development installation complete"

install-full: install-dev
	@echo "Installing full dependencies (including vLLM)..."
	pip install -r requirements-full.txt
	@echo "Attempting vLLM installation..."
	pip install vllm || echo "⚠ vLLM installation failed (requires CUDA)"
	@echo "✓ Full installation complete"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing

test-install:
	@echo "Testing installation..."
	python scripts/test_installation.py

test-simple:
	@echo "Running simple tests..."
	python scripts/simple_test.py

test-client:
	@echo "Testing API client..."
	python tests/test_client.py

lint:
	@echo "Running linters..."
	flake8 src/ --max-line-length=100 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	black src/ tests/ benchmarks/
	isort src/ tests/ benchmarks/

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ logs/*.log
	@echo "✓ Cleaned"

run:
	@echo "Starting server..."
	python -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

docker-build:
	@echo "Building Docker images..."
	docker-compose -f docker-compose-simple.yml build

docker-up:
	@echo "Starting Docker containers..."
	docker-compose -f docker-compose-simple.yml up -d
	@echo "✓ Containers started"
	@echo "  Server: http://localhost:8000"
	@echo "  Metrics: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000"

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose -f docker-compose-simple.yml down

docker-logs:
	docker-compose -f docker-compose-simple.yml logs -f

benchmark:
	@echo "Running benchmarks..."
	@echo "Latency test..."
	python benchmarks/latency_test.py --num-requests 100 --concurrent-users 10
	@echo ""
	@echo "Throughput test..."
	python benchmarks/throughput_test.py --duration 60 --target-rps 10

setup:
	@echo "Running setup script..."
	chmod +x scripts/setup.sh scripts/install.sh
	./scripts/setup.sh

env:
	@if [ ! -f .env ]; then \
		echo "Creating .env file..."; \
		cp .env.example .env; \
		echo "✓ .env created - please edit with your settings"; \
	else \
		echo ".env file already exists"; \
	fi
