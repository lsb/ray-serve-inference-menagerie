#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Running Grounding DINO service tests..."

if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating Python virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: Virtual environment not found. Run ./scripts/setup_dependencies.sh first."
fi

echo "Checking Grounding DINO service health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Grounding DINO service is running"
else
    echo "✗ Grounding DINO service is not accessible at http://localhost:8000"
    echo "Make sure to deploy the service first:"
    echo "  ./scripts/setup_minikube.sh"
    echo "  ./grounding-dino-service/tests/deploy_test.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Running Grounding DINO-specific tests..."
pytest grounding-dino-service/tests/test_grounding_dino_service.py -v

echo ""
echo "All Grounding DINO tests completed successfully!"
