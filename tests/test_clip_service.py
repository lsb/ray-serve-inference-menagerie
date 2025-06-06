import pytest
import requests
import json
import time
from typing import Dict, Any

class TestCLIPService:
    """Test suite for CLIP zero-shot image classification service."""
    
    @pytest.fixture(scope="class")
    def service_url(self) -> str:
        """Base URL for the CLIP service."""
        return "http://localhost:8000"
    
    @pytest.fixture(scope="class")
    def sample_image_url(self) -> str:
        """Sample image URL for testing."""
        return "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    
    @pytest.fixture(scope="class")
    def sample_labels(self) -> list:
        """Sample labels for zero-shot classification."""
        return ["food", "animal", "vehicle", "building", "person"]
    
    def test_service_health(self, service_url: str):
        """Test that the service is running and responsive."""
        try:
            response = requests.get(f"{service_url}/health", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("CLIP service is not running. Start the service to run this test.")
    
    def test_clip_classification_valid_request(self, service_url: str, sample_image_url: str, sample_labels: list):
        """Test CLIP classification with valid image URL and labels."""
        payload = {
            "image_url": sample_image_url,
            "labels": sample_labels
        }
        
        response = requests.post(service_url, json=payload, timeout=30)
        assert response.status_code == 200
        
        result = response.json()
        assert isinstance(result, list)
        assert len(result) > 0
        
        for item in result:
            assert "label" in item
            assert "score" in item
            assert isinstance(item["label"], str)
            assert isinstance(item["score"], (int, float))
            assert 0 <= item["score"] <= 1
        
        scores = [item["score"] for item in result]
        assert scores == sorted(scores, reverse=True)
    
    def test_clip_classification_missing_image_url(self, service_url: str, sample_labels: list):
        """Test CLIP classification with missing image URL."""
        payload = {
            "labels": sample_labels
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
        assert "image_url" in result["error"]
    
    def test_clip_classification_missing_labels(self, service_url: str, sample_image_url: str):
        """Test CLIP classification with missing labels."""
        payload = {
            "image_url": sample_image_url
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
        assert "labels" in result["error"]
    
    def test_clip_classification_empty_labels(self, service_url: str, sample_image_url: str):
        """Test CLIP classification with empty labels list."""
        payload = {
            "image_url": sample_image_url,
            "labels": []
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
    
    def test_clip_classification_invalid_image_url(self, service_url: str, sample_labels: list):
        """Test CLIP classification with invalid image URL."""
        payload = {
            "image_url": "https://invalid-url-that-does-not-exist.com/image.jpg",
            "labels": sample_labels
        }
        
        response = requests.post(service_url, json=payload, timeout=30)
        result = response.json()
        assert "error" in result or response.status_code != 200
    
    def test_clip_classification_single_label(self, service_url: str, sample_image_url: str):
        """Test CLIP classification with a single label."""
        payload = {
            "image_url": sample_image_url,
            "labels": ["food"]
        }
        
        response = requests.post(service_url, json=payload, timeout=30)
        assert response.status_code == 200
        
        result = response.json()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["label"] == "food"
    
    def test_clip_classification_many_labels(self, service_url: str, sample_image_url: str):
        """Test CLIP classification with many labels."""
        many_labels = [
            "food", "animal", "vehicle", "building", "person", "nature", "technology",
            "furniture", "clothing", "sports", "music", "art", "book", "tool", "toy"
        ]
        
        payload = {
            "image_url": sample_image_url,
            "labels": many_labels
        }
        
        response = requests.post(service_url, json=payload, timeout=30)
        assert response.status_code == 200
        
        result = response.json()
        assert isinstance(result, list)
        assert len(result) == len(many_labels)
        
        result_labels = {item["label"] for item in result}
        assert result_labels == set(many_labels)
    
    def test_clip_performance_benchmark(self, service_url: str, sample_image_url: str, sample_labels: list):
        """Benchmark the performance of CLIP classification."""
        payload = {
            "image_url": sample_image_url,
            "labels": sample_labels
        }
        
        requests.post(service_url, json=payload, timeout=30)
        
        response_times = []
        for _ in range(5):
            start_time = time.time()
            response = requests.post(service_url, json=payload, timeout=30)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        print(f"Average response time: {avg_response_time:.2f} seconds")
        
        assert avg_response_time < 10.0
