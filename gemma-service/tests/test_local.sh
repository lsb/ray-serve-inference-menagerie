#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Running Ray Serve Inference Menagerie Gemma tests..."

if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating Python virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: Virtual environment not found. Run ./scripts/setup_dependencies.sh first."
fi

echo "Checking Gemma service health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Gemma service is running"
else
    echo "✗ Gemma service is not accessible at http://localhost:8000"
    echo "Make sure to deploy the service first:"
    echo "  ./scripts/setup_minikube.sh"
    echo "  ./gemma-service/tests/deploy_test.sh"
    exit 1
fi

echo "Running pytest test suite..."
cd "$PROJECT_ROOT"

pytest gemma-service/tests/test_gemma_service.py -v --tb=short

echo ""
echo "Running Gemma validation tests..."
pytest gemma-service/tests/test_gemma_service.py::TestGemmaService::test_gemma_performance_benchmark -v

echo ""
echo "All tests completed successfully!"
echo ""
echo "Performance summary:"
curl -s http://localhost:8000/health | python3 -m json.tool
