import pytest
import os
import subprocess
import time
import requests
from typing import Generator

def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--run-integration", 
        action="store_true", 
        default=False, 
        help="Run integration tests that require services to be running"
    )
    parser.addoption(
        "--service-timeout",
        action="store",
        default=30,
        type=int,
        help="Timeout in seconds to wait for services to start"
    )

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring running services"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--run-integration"):
        return
    
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

@pytest.fixture(scope="session")
def service_timeout(request) -> int:
    """Get the service timeout from command line options."""
    return request.config.getoption("--service-timeout")

def wait_for_service(url: str, timeout: int = 30) -> bool:
    """Wait for a service to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False

@pytest.fixture(scope="session")
def clip_service_url() -> str:
    """URL for the CLIP service."""
    return os.environ.get("CLIP_SERVICE_URL", "http://localhost:8000")

@pytest.fixture(scope="session")
def gemma_service_url() -> str:
    """URL for the Gemma service."""
    return os.environ.get("GEMMA_SERVICE_URL", "http://localhost:8001")

@pytest.fixture(scope="session")
def grounding_dino_service_url() -> str:
    """URL for the Grounding DINO service."""
    return os.environ.get("GROUNDING_DINO_SERVICE_URL", "http://localhost:8002")

@pytest.fixture(scope="session")
def all_services_running(clip_service_url: str, gemma_service_url: str, 
                        grounding_dino_service_url: str, service_timeout: int) -> bool:
    """Check if all services are running and accessible."""
    services = {
        "CLIP": clip_service_url,
        "Gemma": gemma_service_url,
        "Grounding DINO": grounding_dino_service_url
    }
    
    running_services = []
    for name, url in services.items():
        if wait_for_service(url, timeout=service_timeout):
            running_services.append(name)
            print(f"✓ {name} service is running at {url}")
        else:
            print(f"✗ {name} service is not accessible at {url}")
    
    return len(running_services) == len(services)

class ServiceManager:
    """Helper class to manage service lifecycle during testing."""
    
    def __init__(self):
        self.processes = {}
    
    def start_service(self, name: str, command: list, cwd: str = None) -> subprocess.Popen:
        """Start a service process."""
        if name in self.processes:
            self.stop_service(name)
        
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.processes[name] = process
        return process
    
    def stop_service(self, name: str):
        """Stop a service process."""
        if name in self.processes:
            process = self.processes[name]
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            del self.processes[name]
    
    def stop_all_services(self):
        """Stop all managed service processes."""
        for name in list(self.processes.keys()):
            self.stop_service(name)

@pytest.fixture(scope="session")
def service_manager() -> Generator[ServiceManager, None, None]:
    """Provide a service manager for tests."""
    manager = ServiceManager()
    yield manager
    manager.stop_all_services()

@pytest.fixture(scope="session")
def test_images():
    """Common test images for all services."""
    return {
        "food": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
        "animals": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "objects": "https://images.unsplash.com/photo-1586953208448-b95a79798f07?w=400"
    }

@pytest.fixture(scope="session")
def test_labels():
    """Common test labels for classification tasks."""
    return {
        "general": ["food", "animal", "vehicle", "building", "person", "nature"],
        "food": ["pastry", "bread", "dessert", "cake", "cookie", "donut"],
        "animals": ["cat", "dog", "bird", "fish", "horse", "cow"]
    }
