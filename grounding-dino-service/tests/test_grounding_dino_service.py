import pytest
import requests
import json
import time
import io
from PIL import Image
from typing import Dict, Any, List

class TestGroundingDinoService:
    """Test suite for Grounding DINO + SAM2 object detection and segmentation service."""
    
    @pytest.fixture(scope="class")
    def service_url(self) -> str:
        """Base URL for the Grounding DINO service."""
        return "http://localhost:8000"
    
    @pytest.fixture(scope="class")
    def sample_image_url(self) -> str:
        """Sample image URL for testing."""
        return "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    
    @pytest.fixture(scope="class")
    def sample_image_bytes(self, sample_image_url: str) -> bytes:
        """Download sample image as bytes for upload testing."""
        response = requests.get(sample_image_url)
        return response.content
    
    def test_service_health(self, service_url: str):
        """Test that the service is running and responsive."""
        try:
            response = requests.get(f"{service_url}/health", timeout=10)
            assert response.status_code == 200
            result = response.json()
            assert "status" in result
            assert result["status"] == "healthy"
            assert "service" in result
            assert result["service"] == "grounding-dino"
            assert "device_info" in result
        except requests.exceptions.ConnectionError:
            pytest.skip("Grounding DINO service is not running. Start the service to run this test.")
    
    def test_text_detect_valid_request(self, service_url: str, sample_image_bytes: bytes):
        """Test text-prompted detection with valid image and text."""
        files = {
            'image': ('test_image.jpg', sample_image_bytes, 'image/jpeg')
        }
        data = {
            'text': 'food'
        }
        
        response = requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=60)
        assert response.status_code == 200
        
        result = response.json()
        assert "boxes" in result
        assert "scores" in result
        assert "text" in result
        assert "performance" in result
        
        assert isinstance(result["boxes"], list)
        assert isinstance(result["scores"], list)
        assert result["text"] == "food"
        
        performance = result["performance"]
        assert "total_time_ms" in performance
        assert "inference_time_ms" in performance
        assert "device_info" in performance
        
        if len(result["boxes"]) > 0:
            assert len(result["boxes"]) == len(result["scores"])
            
            for box in result["boxes"]:
                assert isinstance(box, list)
                assert len(box) == 4  # [x1, y1, x2, y2]
                assert all(isinstance(coord, (int, float)) for coord in box)
                assert box[0] < box[2]  # x1 < x2
                assert box[1] < box[3]  # y1 < y2
            
            for score in result["scores"]:
                assert isinstance(score, (int, float))
                assert 0 <= score <= 1
    
    def test_text_detect_multiple_objects(self, service_url: str, sample_image_bytes: bytes):
        """Test text-prompted detection for multiple object types."""
        test_queries = ["food", "plate", "object", "item"]
        
        for query in test_queries:
            files = {
                'image': ('test_image.jpg', sample_image_bytes, 'image/jpeg')
            }
            data = {
                'text': query
            }
            
            response = requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=60)
            assert response.status_code == 200
            
            result = response.json()
            assert "boxes" in result
            assert "scores" in result
            assert "text" in result
            assert "performance" in result
            assert result["text"] == query
            
            performance = result["performance"]
            assert "total_time_ms" in performance
            assert "inference_time_ms" in performance
            assert "device_info" in performance
    
    def test_detect_generic(self, service_url: str, sample_image_bytes: bytes):
        """Test generic object detection without text prompts."""
        files = {
            'image': ('test_image.jpg', sample_image_bytes, 'image/jpeg')
        }
        
        response = requests.post(f"{service_url}/detect", files=files, timeout=60)
        assert response.status_code == 200
        
        result = response.json()
        assert "boxes" in result
        assert "performance" in result
        assert isinstance(result["boxes"], list)
        
        performance = result["performance"]
        assert "total_time_ms" in performance
        assert "inference_time_ms" in performance
        assert "device_info" in performance
        
        for box in result["boxes"]:
            assert isinstance(box, list)
            assert len(box) == 4  # [x1, y1, x2, y2]
            assert all(isinstance(coord, (int, float)) for coord in box)
            assert box[0] < box[2]  # x1 < x2
            assert box[1] < box[3]  # y1 < y2
    
    def test_text_detect_missing_image(self, service_url: str):
        """Test text-prompted detection with missing image."""
        data = {
            'text': 'food'
        }
        
        response = requests.post(f"{service_url}/text_detect", data=data, timeout=10)
        assert response.status_code == 422  # FastAPI validation error
    
    def test_text_detect_missing_text(self, service_url: str, sample_image_bytes: bytes):
        """Test text-prompted detection with missing text."""
        files = {
            'image': ('test_image.jpg', sample_image_bytes, 'image/jpeg')
        }
        
        response = requests.post(f"{service_url}/text_detect", files=files, timeout=10)
        assert response.status_code == 422  # FastAPI validation error
    
    def test_detect_missing_image(self, service_url: str):
        """Test generic detection with missing image."""
        response = requests.post(f"{service_url}/detect", timeout=10)
        assert response.status_code == 422  # FastAPI validation error
    
    def test_text_detect_invalid_image(self, service_url: str):
        """Test text-prompted detection with invalid image data."""
        files = {
            'image': ('test.txt', b'not an image', 'text/plain')
        }
        data = {
            'text': 'food'
        }
        
        response = requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=30)
        assert response.status_code != 200 or "error" in response.json()
    
    def test_text_detect_complex_query(self, service_url: str, sample_image_bytes: bytes):
        """Test text-prompted detection with complex text queries."""
        complex_queries = [
            "round food item",
            "golden brown pastry",
            "sweet dessert",
            "baked goods on plate"
        ]
        
        for query in complex_queries:
            files = {
                'image': ('test_image.jpg', sample_image_bytes, 'image/jpeg')
            }
            data = {
                'text': query
            }
            
            response = requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=60)
            assert response.status_code == 200
            
            result = response.json()
            assert "boxes" in result
            assert "scores" in result
            assert "text" in result
            assert "performance" in result
            assert result["text"] == query
            
            performance = result["performance"]
            assert "total_time_ms" in performance
            assert "inference_time_ms" in performance
            assert "device_info" in performance
    
    def test_performance_benchmark(self, service_url: str, sample_image_bytes: bytes):
        """Benchmark the performance of object detection."""
        files = {
            'image': ('test_image.jpg', sample_image_bytes, 'image/jpeg')
        }
        data = {
            'text': 'food'
        }
        
        requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=60)
        
        response_times = []
        for _ in range(3):  # Fewer iterations due to longer response times
            start_time = time.time()
            response = requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=60)
            end_time = time.time()
            
            assert response.status_code == 200
            result = response.json()
            assert "boxes" in result
            assert "performance" in result
            
            performance = result["performance"]
            assert "total_time_ms" in performance
            assert "inference_time_ms" in performance
            assert "device_info" in performance
            
            response_times.append(end_time - start_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        print(f"Average response time: {avg_response_time:.2f} seconds")
        
        assert avg_response_time < 30.0
    
    def test_batch_detection(self, service_url: str, sample_image_bytes: bytes):
        """Test multiple detection requests in sequence."""
        queries = ["food", "plate", "object"]
        
        for query in queries:
            files = {
                'image': ('test_image.jpg', sample_image_bytes, 'image/jpeg')
            }
            data = {
                'text': query
            }
            
            response = requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=60)
            assert response.status_code == 200
            
            result = response.json()
            assert "boxes" in result
            assert "scores" in result
            assert "text" in result
            assert "performance" in result
            
            performance = result["performance"]
            assert "total_time_ms" in performance
            assert "inference_time_ms" in performance
            assert "device_info" in performance
            
            print(f"Query '{query}' found {len(result['boxes'])} objects")
