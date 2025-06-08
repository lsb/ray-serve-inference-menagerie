import pytest
import requests
import json
import time
import io
import os
from typing import Dict, Any, Callable
from PIL import Image

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
    
    @pytest.fixture(scope="class")
    def sample_image_bytes(self) -> bytes:
        """Sample image bytes for testing."""
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
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
        return os.path.join(project_root, "sample_data", "dogs_running", "dog_running_05.png")
    
    @pytest.fixture(scope="class")
    def cat_vs_dog_prompts(self) -> list:
        """Prompts for cat vs dog classification test."""
        return ["Is this a cat or a dog?", "What animal is this?"]
    
    @pytest.fixture(scope="class")
    def inside_vs_outside_prompts(self) -> list:
        """Prompts for inside vs outside classification test."""
        return ["Is this an indoor or outdoor scene?", "Where is this photo taken - inside or outside?"]
    
    @pytest.fixture(scope="class")
    def image_to_bytes(self) -> Callable[[str], bytes]:
        """Helper function to convert image files to bytes."""
        def convert(image_path: str) -> bytes:
            with open(image_path, 'rb') as f:
                return f.read()
        return convert
    
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
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"Gemma service is not accessible at {service_url}: {str(e)}")
    
    def test_gemma_image_description(self, service_url: str, sample_image_url: str):
        """Test Gemma VLM with image description task."""
        img_response = requests.get(sample_image_url)
        img_response.raise_for_status()
        
        files = {
            'image': ('test_image.png', img_response.content, 'image/png')
        }
        data = {
            'prompt': "Describe what you see in this image."
        }
        
        response = requests.post(service_url, files=files, data=data, timeout=60)
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
        img_response = requests.get(sample_image_url)
        img_response.raise_for_status()
        
        files = {
            'image': ('test_image.png', img_response.content, 'image/png')
        }
        
        response = requests.post(service_url, files=files, timeout=60)
        assert response.status_code == 200
        
        result = response.json()
        assert "answer" in result
        assert "performance" in result
        assert isinstance(result["answer"], str)
    
    def test_gemma_image_bytes(self, service_url: str, sample_image_bytes: bytes):
        """Test Gemma VLM with image bytes."""
        files = {
            'image': ('test_image.png', sample_image_bytes, 'image/png')
        }
        data = {
            'prompt': "Describe what you see in this image."
        }
        
        response = requests.post(service_url, files=files, data=data, timeout=60)
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
    
    def test_gemma_missing_image_data(self, service_url: str):
        """Test Gemma VLM with missing image data."""
        data = {
            'prompt': "Describe the image."
        }
        
        response = requests.post(service_url, data=data, timeout=10)
        assert response.status_code == 422
    
    def test_gemma_invalid_image_data(self, service_url: str):
        """Test Gemma VLM with invalid image data."""
        files = {
            'image': ('test_image.png', b'invalid_image_data', 'image/png')
        }
        data = {
            'prompt': "Describe the image."
        }
        
        response = requests.post(service_url, files=files, data=data, timeout=10)
        result = response.json()
        
        assert "error" in result
    
    def test_gemma_performance_benchmark(self, service_url: str, sample_image_url: str):
        """Benchmark the performance of Gemma VLM."""
        img_response = requests.get(sample_image_url)
        img_response.raise_for_status()
        
        files = {
            'image': ('test_image.png', img_response.content, 'image/png')
        }
        data = {
            'prompt': "Describe this image briefly."
        }
        
        requests.post(service_url, files=files, data=data, timeout=60)
        
        response_times = []
        for _ in range(3):
            start_time = time.time()
            response = requests.post(service_url, files=files, data=data, timeout=60)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        print(f"Average response time: {avg_response_time:.2f} seconds")
        
        assert avg_response_time < 30.0
    
    def test_cat_vs_dog_recognition(self, service_url: str, cat_image_path: str, dog_image_path: str, cat_vs_dog_prompts: list, image_to_bytes: Callable[[str], bytes]):
        """Test that Gemma correctly identifies cat images as cats rather than dogs using local sample images."""
        assert os.path.exists(cat_image_path), f"Cat sample image not found at {cat_image_path}"
        assert os.path.exists(dog_image_path), f"Dog sample image not found at {dog_image_path}"
        
        for prompt in cat_vs_dog_prompts:
            cat_bytes = image_to_bytes(cat_image_path)
            cat_files = {
                'image': ('cat_image.png', cat_bytes, 'image/png')
            }
            cat_data = {
                'prompt': prompt
            }
            
            try:
                cat_response = requests.post(service_url, files=cat_files, data=cat_data, timeout=60)
                assert cat_response.status_code == 200
                cat_result = cat_response.json()
                assert "answer" in cat_result
                
                cat_answer = cat_result["answer"].lower()
                assert "cat" in cat_answer, f"Cat image should mention 'cat' in response. Got: {cat_result['answer']}"
                
                dog_bytes = image_to_bytes(dog_image_path)
                dog_files = {
                    'image': ('dog_image.png', dog_bytes, 'image/png')
                }
                dog_data = {
                    'prompt': prompt
                }
                
                dog_response = requests.post(service_url, files=dog_files, data=dog_data, timeout=60)
                assert dog_response.status_code == 200
                dog_result = dog_response.json()
                assert "answer" in dog_result
                
                dog_answer = dog_result["answer"].lower()
                assert "dog" in dog_answer, f"Dog image should be identified as dog, got: {dog_result['answer']}"
                
                print(f"Cat vs Dog test passed:")
                print(f"  Cat image response: {cat_result['answer']}")
                print(f"  Dog image response: {dog_result['answer']}")
                
                if "performance" in cat_result:
                    performance = cat_result["performance"]
                    print(f"Performance: {performance.get('total_time_ms', 'N/A')}ms total")
                    print(f"Device: {performance.get('device_info', {}).get('device_type', 'unknown')}")
                    
            except requests.exceptions.ConnectionError as e:
                pytest.fail(f"Gemma service is not accessible at {service_url}: {str(e)}")
    
    def test_inside_vs_outside_classification(self, service_url: str, cat_image_path: str, inside_vs_outside_prompts: list, image_to_bytes: Callable[[str], bytes]):
        """Test that Gemma correctly identifies cat desk images as inside rather than outside."""
        assert os.path.exists(cat_image_path), f"Cat sample image not found at {cat_image_path}"
        
        for prompt in inside_vs_outside_prompts:
            cat_bytes = image_to_bytes(cat_image_path)
            files = {
                'image': ('cat_image.png', cat_bytes, 'image/png')
            }
            data = {
                'prompt': prompt
            }
            
            try:
                response = requests.post(service_url, files=files, data=data, timeout=60)
                assert response.status_code == 200
                
                result = response.json()
                assert "answer" in result
                assert "performance" in result
                
                answer = result["answer"].lower()
                assert "indoor" in answer or "inside" in answer, f"Cat on desk should be identified as indoor/inside. Got: {result['answer']}"
                
                print(f"Inside vs Outside classification for prompt '{prompt}': {result['answer'][:100]}...")
                
            except requests.exceptions.ConnectionError as e:
                pytest.fail(f"Gemma service is not accessible at {service_url}: {str(e)}")
