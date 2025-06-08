#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Running Moondream service tests..."

if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating Python virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: Virtual environment not found. Run ./scripts/setup_dependencies.sh first."
fi

echo "Checking Moondream service health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Moondream service is running"
else
    echo "✗ Moondream service is not accessible at http://localhost:8000"
    echo "Make sure to deploy the service first:"
    echo "  ./scripts/setup_minikube.sh"
    echo "  ./moondream-service/tests/deploy_test.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Running Moondream-specific tests..."
pytest moondream-service/tests/test_moondream_service.py -v

echo ""
echo "All Moondream tests completed successfully!"
