import pytest
import requests
import json
import time
from typing import Dict, Any

class TestGemmaService:
    """Test suite for Gemma3 VLM (Vision-Language Model) service."""
    
    @pytest.fixture(scope="class")
    def service_url(self) -> str:
        """Base URL for the Gemma service."""
        return "http://localhost:8000"
    
    @pytest.fixture(scope="class")
    def sample_image_url(self) -> str:
        """Sample image URL for testing."""
        return "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    
    def test_service_health(self, service_url: str):
        """Test that the service is running and responsive."""
        try:
            response = requests.get(f"{service_url}/health", timeout=10)
            assert response.status_code == 200
            result = response.json()
            assert "status" in result
            assert result["status"] == "healthy"
            assert "service" in result
            assert result["service"] == "gemma"
            assert "device_info" in result
        except requests.exceptions.ConnectionError:
            pytest.skip("Gemma service is not running. Start the service to run this test.")
    
    def test_gemma_image_description(self, service_url: str, sample_image_url: str):
        """Test Gemma VLM with image description task."""
        payload = {
            "image_url": sample_image_url,
            "prompt": "Describe what you see in this image."
        }
        
        response = requests.post(service_url, json=payload, timeout=60)
        assert response.status_code == 200
        
        result = response.json()
        assert "answer" in result
        assert "performance" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        
        performance = result["performance"]
        assert "total_time_ms" in performance
        assert "inference_time_ms" in performance
        assert "device_info" in performance
    
    def test_gemma_image_only(self, service_url: str, sample_image_url: str):
        """Test Gemma VLM with image only (no prompt)."""
        payload = {
            "image_url": sample_image_url
        }
        
        response = requests.post(service_url, json=payload, timeout=60)
        assert response.status_code == 200
        
        result = response.json()
        assert "answer" in result
        assert "performance" in result
        assert isinstance(result["answer"], str)
    
    def test_gemma_missing_image_url(self, service_url: str):
        """Test Gemma VLM with missing image URL."""
        payload = {
            "prompt": "Describe the image."
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
        assert "image_url" in result["error"]
    
    def test_gemma_performance_benchmark(self, service_url: str, sample_image_url: str):
        """Benchmark the performance of Gemma VLM."""
        payload = {
            "image_url": sample_image_url,
            "prompt": "Describe this image briefly."
        }
        
        requests.post(service_url, json=payload, timeout=60)
        
        response_times = []
        for _ in range(3):
            start_time = time.time()
            response = requests.post(service_url, json=payload, timeout=60)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        print(f"Average response time: {avg_response_time:.2f} seconds")
        
        assert avg_response_time < 30.0
