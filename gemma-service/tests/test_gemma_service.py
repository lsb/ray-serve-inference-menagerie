import pytest
import requests
import json
import time
import base64
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
    def sample_image_base64(self) -> str:
        """Sample base64-encoded image for testing."""
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
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
    def image_to_base64(self) -> Callable[[str], str]:
        """Helper function to convert image files to base64."""
        def convert(image_path: str) -> str:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')
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
    
    def test_gemma_base64_image(self, service_url: str, sample_image_base64: str):
        """Test Gemma VLM with base64-encoded image."""
        payload = {
            "image": sample_image_base64,
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
    
    def test_gemma_missing_image_data(self, service_url: str):
        """Test Gemma VLM with missing image data."""
        payload = {
            "prompt": "Describe the image."
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
        assert "image" in result["error"] or "image_url" in result["error"]
    
    def test_gemma_invalid_base64(self, service_url: str):
        """Test Gemma VLM with invalid base64 image data."""
        payload = {
            "image": "invalid_base64_data",
            "prompt": "Describe the image."
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
        assert "Invalid base64" in result["error"]
    
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
    
    def test_cat_vs_dog_recognition(self, service_url: str, cat_image_path: str, dog_image_path: str, cat_vs_dog_prompts: list, image_to_base64: Callable[[str], str]):
        """Test that Gemma correctly identifies cat images as cats rather than dogs using local sample images."""
        assert os.path.exists(cat_image_path), f"Cat sample image not found at {cat_image_path}"
        assert os.path.exists(dog_image_path), f"Dog sample image not found at {dog_image_path}"
        
        cat_base64 = image_to_base64(cat_image_path)
        cat_payload = {
            "image": cat_base64,
            "prompt": cat_vs_dog_prompts[0]
        }
        
        try:
            cat_response = requests.post(service_url, json=cat_payload, timeout=60)
            assert cat_response.status_code == 200
            cat_result = cat_response.json()
            assert "answer" in cat_result
            
            cat_answer = cat_result["answer"].lower()
            assert "cat" in cat_answer, f"Cat image should be identified as cat, got: {cat_result['answer']}"
            
            dog_base64 = image_to_base64(dog_image_path)
            dog_payload = {
                "image": dog_base64,
                "prompt": cat_vs_dog_prompts[0]
            }
            
            dog_response = requests.post(service_url, json=dog_payload, timeout=60)
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
    
    def test_inside_vs_outside_classification(self, service_url: str, cat_image_path: str, inside_vs_outside_prompts: list, image_to_base64: Callable[[str], str]):
        """Test that Gemma correctly identifies cat desk images as inside rather than outside."""
        assert os.path.exists(cat_image_path), f"Cat sample image not found at {cat_image_path}"
        
        cat_base64 = image_to_base64(cat_image_path)
        payload = {
            "image": cat_base64,
            "prompt": inside_vs_outside_prompts[0]
        }
        
        try:
            response = requests.post(service_url, json=payload, timeout=60)
            assert response.status_code == 200
            
            result = response.json()
            assert "answer" in result
            assert "performance" in result
            
            answer = result["answer"].lower()
            assert "indoor" in answer or "inside" in answer, f"Cat on desk should be identified as indoor/inside, got: {result['answer']}"
            
            print(f"Indoor vs Outdoor classification: {result['answer']}")
            
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"Gemma service is not accessible at {service_url}: {str(e)}")
    
    def test_outdoor_scene_classification(self, service_url: str, dog_image_path: str, inside_vs_outside_prompts: list, image_to_base64: Callable[[str], str]):
        """Test that Gemma correctly identifies dog running images as outside rather than inside."""
        assert os.path.exists(dog_image_path), f"Dog sample image not found at {dog_image_path}"
        
        dog_base64 = image_to_base64(dog_image_path)
        payload = {
            "image": dog_base64,
            "prompt": inside_vs_outside_prompts[0]
        }
        
        try:
            response = requests.post(service_url, json=payload, timeout=60)
            assert response.status_code == 200
            
            result = response.json()
            assert "answer" in result
            assert "performance" in result
            
            answer = result["answer"].lower()
            assert "outdoor" in answer or "outside" in answer, f"Dog running in field should be identified as outdoor/outside, got: {result['answer']}"
            
            print(f"Outdoor scene classification: {result['answer']}")
            
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"Gemma service is not accessible at {service_url}: {str(e)}")
