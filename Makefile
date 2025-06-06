# Makefile for Ray Serve Inference Menagerie

.PHONY: help setup lint test test-integration test-docker test-k8s run clean docker-build docker-push deploy

# Default target
help:
	@echo "Available targets:"
	@echo "  setup           - Install dependencies and set up development environment"
	@echo "  lint            - Run code linting and formatting"
	@echo "  lint-fix        - Run linting with automatic fixes"
	@echo "  test            - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-docker     - Test Docker builds"
	@echo "  test-k8s        - Test Kubernetes manifests"
	@echo "  run             - Start services locally"
	@echo "  clean           - Clean up temporary files and containers"
	@echo "  docker-build    - Build all Docker images"
	@echo "  docker-push     - Push Docker images to registry"
	@echo "  deploy          - Deploy to Kubernetes cluster"

# TODO: Implement all targets
setup:
	@echo "TODO: Run scripts/setup_dependencies.sh"
	./scripts/setup_dependencies.sh --dev

lint:
	@echo "TODO: Run scripts/lint.sh"
	./scripts/lint.sh

lint-fix:
	@echo "TODO: Run scripts/lint.sh --fix"
	./scripts/lint.sh --fix

test:
	@echo "TODO: Run scripts/test.sh"
	./scripts/test.sh

test-integration:
	@echo "TODO: Run scripts/test.sh --integration"
	./scripts/test.sh --integration

test-docker:
	@echo "TODO: Run Docker build tests"
	pytest tests/test_docker_builds.py -v

test-k8s:
	@echo "TODO: Run Kubernetes manifest tests"
	pytest tests/test_kubernetes_manifests.py -v

test-clip:
	@echo "Running CLIP service tests"
	./clip-service/tests/run_tests.sh

test-all-services:
	@echo "Running all service tests (comprehensive test suite)"
	@echo "Testing CLIP service..."
	./clip-service/tests/run_tests.sh
	@echo ""
	@echo "Testing other services..."
	pytest tests/test_grounding_dino_service.py -v
	pytest tests/test_gemma_service.py -v
	@echo ""
	@echo "All service tests completed!"

run:
	@echo "TODO: Run scripts/run_local.sh"
	./scripts/run_local.sh

clean:
	@echo "TODO: Clean up temporary files and containers"
	@echo "Would clean up:"
	@echo "  - Docker containers and images"
	@echo "  - Temporary files"
	@echo "  - Ray cluster processes"
	@echo "  - Test artifacts"

docker-build:
	@echo "TODO: Build all Docker images"
	@echo "Would build:"
	@echo "  - clip-service"
	@echo "  - gemma-service" 
	@echo "  - grounding-dino-service"

docker-push:
	@echo "TODO: Push Docker images to registry"
	@echo "Would push all service images to configured registry"

deploy:
	@echo "TODO: Deploy to Kubernetes"
	@echo "Would deploy all services using scripts/deploy_ray_service.sh"
