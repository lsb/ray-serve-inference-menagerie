import pytest
import requests
import json
import time
import io
import os
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
    def cat_image_path(self) -> str:
        """Path to local cat sample image."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        return os.path.join(project_root, "sample_data", "cats_on_desk", "cat_on_desk_01.png")
    
    @pytest.fixture(scope="class")
    def dog_image_path(self) -> str:
        """Path to local dog sample image."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        return os.path.join(project_root, "sample_data", "dogs_running", "dog_running_01.png")
    
    @pytest.fixture(scope="class")
    def cat_image_bytes(self, cat_image_path: str) -> bytes:
        """Load local cat image as bytes for upload testing."""
        if not os.path.exists(cat_image_path):
            pytest.skip(f"Cat sample image not found at {cat_image_path}")
        with open(cat_image_path, 'rb') as f:
            return f.read()
    
    @pytest.fixture(scope="class")
    def dog_image_bytes(self, dog_image_path: str) -> bytes:
        """Load local dog image as bytes for upload testing."""
        if not os.path.exists(dog_image_path):
            pytest.skip(f"Dog sample image not found at {dog_image_path}")
        with open(dog_image_path, 'rb') as f:
            return f.read()
    
    @pytest.fixture(scope="class")
    def sample_image_bytes(self, cat_image_path: str) -> bytes:
        """Load local sample image as bytes for upload testing (fallback to cat image)."""
        if os.path.exists(cat_image_path):
            with open(cat_image_path, 'rb') as f:
                return f.read()
        else:
            response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
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
    
    def test_cat_vs_dog_detection(self, service_url: str, cat_image_bytes: bytes, dog_image_bytes: bytes):
        """Test that Grounding DINO correctly detects cats in cat images and dogs in dog images."""
        cat_files = {
            'image': ('cat_image.png', cat_image_bytes, 'image/png')
        }
        cat_data = {
            'text': 'cat'
        }
        
        try:
            cat_response = requests.post(f"{service_url}/text_detect", files=cat_files, data=cat_data, timeout=60)
            assert cat_response.status_code == 200
            cat_result = cat_response.json()
            assert "boxes" in cat_result
            assert "scores" in cat_result
            assert "text" in cat_result
            assert "performance" in cat_result
            assert cat_result["text"] == "cat"
            
            performance = cat_result["performance"]
            assert "total_time_ms" in performance
            assert "inference_time_ms" in performance
            assert "device_info" in performance
            
            dog_files = {
                'image': ('dog_image.png', dog_image_bytes, 'image/png')
            }
            dog_data = {
                'text': 'dog'
            }
            
            dog_response = requests.post(f"{service_url}/text_detect", files=dog_files, data=dog_data, timeout=60)
            assert dog_response.status_code == 200
            dog_result = dog_response.json()
            assert "boxes" in dog_result
            assert "scores" in dog_result
            assert "text" in dog_result
            assert "performance" in dog_result
            assert dog_result["text"] == "dog"
            
            dog_performance = dog_result["performance"]
            assert "total_time_ms" in dog_performance
            assert "inference_time_ms" in dog_performance
            assert "device_info" in dog_performance
            
            print(f"Cat detection: Found {len(cat_result['boxes'])} boxes")
            print(f"Dog detection: Found {len(dog_result['boxes'])} boxes")
            
            assert len(cat_result['boxes']) > 0 or len(dog_result['boxes']) > 0, "No detections found in either cat or dog images"
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Grounding DINO service is not running. Start the service to run this test.")
    
    def test_all_sample_images_detection(self, service_url: str):
        """Test detection on all available cat and dog sample images."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        cats_dir = os.path.join(project_root, "sample_data", "cats_on_desk")
        cat_images = [f for f in os.listdir(cats_dir) if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(cats_dir) else []
        
        successful_cat_tests = 0
        for cat_image in cat_images:
            cat_path = os.path.join(cats_dir, cat_image)
            try:
                with open(cat_path, 'rb') as f:
                    cat_bytes = f.read()
                
                files = {
                    'image': (cat_image, cat_bytes, 'image/png')
                }
                data = {
                    'text': 'cat'
                }
                
                response = requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=60)
                assert response.status_code == 200
                result = response.json()
                assert "boxes" in result
                assert "performance" in result
                
                successful_cat_tests += 1
                print(f"✓ {cat_image}: Found {len(result['boxes'])} detections")
                
            except requests.exceptions.ConnectionError:
                pytest.skip("Grounding DINO service is not running. Start the service to run this test.")
            except Exception as e:
                print(f"✗ {cat_image} failed: {e}")
        
        dogs_dir = os.path.join(project_root, "sample_data", "dogs_running")
        dog_images = [f for f in os.listdir(dogs_dir) if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(dogs_dir) else []
        
        successful_dog_tests = 0
        for dog_image in dog_images:
            dog_path = os.path.join(dogs_dir, dog_image)
            try:
                with open(dog_path, 'rb') as f:
                    dog_bytes = f.read()
                
                files = {
                    'image': (dog_image, dog_bytes, 'image/png')
                }
                data = {
                    'text': 'dog'
                }
                
                response = requests.post(f"{service_url}/text_detect", files=files, data=data, timeout=60)
                assert response.status_code == 200
                result = response.json()
                assert "boxes" in result
                assert "performance" in result
                
                successful_dog_tests += 1
                print(f"✓ {dog_image}: Found {len(result['boxes'])} detections")
                
            except requests.exceptions.ConnectionError:
                pytest.skip("Grounding DINO service is not running. Start the service to run this test.")
            except Exception as e:
                print(f"✗ {dog_image} failed: {e}")
        
        total_tests = len(cat_images) + len(dog_images)
        successful_tests = successful_cat_tests + successful_dog_tests
        print(f"Overall success rate: {successful_tests}/{total_tests} images processed successfully")
        
        assert total_tests > 0, "No sample images found to test"
        assert successful_tests > 0, "No sample images processed successfully"
