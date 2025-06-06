#!/bin/bash

set -e

echo "Starting Moondream Service local tests..."
echo "========================================"

SERVICE_URL="http://localhost:8000"
TIMEOUT=30

wait_for_service() {
    local url=$1
    local timeout=$2
    local count=0
    
    echo "Waiting for service at $url to be ready..."
    
    while [ $count -lt $timeout ]; do
        if curl -s "$url/health" > /dev/null 2>&1; then
            echo "Service is ready!"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    echo "Service failed to start within $timeout seconds"
    return 1
}

if ! wait_for_service $SERVICE_URL $TIMEOUT; then
    echo "Error: Moondream service is not running at $SERVICE_URL"
    echo "Please start the service first with: python moondream-service/app.py"
    exit 1
fi

echo ""
echo "Running Moondream validation tests..."
pytest moondream-service/tests/test_moondream_service.py::TestMoondreamService::test_moondream_performance_benchmark -v
pytest moondream-service/tests/test_moondream_service.py::TestMoondreamService::test_cat_vs_dog_recognition -v
pytest moondream-service/tests/test_moondream_service.py::TestMoondreamService::test_inside_vs_outside_classification -v
pytest moondream-service/tests/test_moondream_service.py::TestMoondreamService::test_outdoor_scene_classification -v

echo ""
echo "All Moondream tests completed successfully!"
