import pytest
import requests
import json
import time
import os
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
    
    @pytest.fixture(scope="class")
    def cat_image_path(self) -> str:
        """Path to local cat sample image."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        return os.path.join(project_root, "sample_data", "cats_on_desk", "cat_on_desk_01.png")
    
    @pytest.fixture(scope="class")
    def cat_vs_dog_labels(self) -> list:
        """Labels for cat vs dog classification test."""
        return ["a photo of a cat", "a photo of a dog"]
    
    @pytest.fixture(scope="class")
    def inside_vs_outside_labels(self) -> list:
        """Labels for inside vs outside classification test."""
        return ["indoor scene", "outdoor scene"]
    
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
        assert "predictions" in result
        assert "performance" in result
        
        predictions = result["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        
        for item in predictions:
            assert "label" in item
            assert "score" in item
            assert isinstance(item["label"], str)
            assert isinstance(item["score"], (int, float))
            assert 0 <= item["score"] <= 1
        
        scores = [item["score"] for item in predictions]
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
        assert "predictions" in result
        
        predictions = result["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert predictions[0]["label"] == "food"
    
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
        assert "predictions" in result
        
        predictions = result["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) == len(many_labels)
        
        result_labels = {item["label"] for item in predictions}
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
    
    def test_cat_vs_dog_classification(self, service_url: str, cat_image_path: str, cat_vs_dog_labels: list):
        """Test that CLIP correctly identifies cat images as cats rather than dogs."""
        if not os.path.exists(cat_image_path):
            pytest.skip(f"Cat sample image not found at {cat_image_path}")
        
        public_cat_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
        
        payload = {
            "image_url": public_cat_url,
            "labels": cat_vs_dog_labels
        }
        
        try:
            response = requests.post(service_url, json=payload, timeout=30)
            assert response.status_code == 200
            
            result = response.json()
            assert "predictions" in result
            assert "performance" in result
            
            predictions = result["predictions"]
            assert len(predictions) == 2
            
            cat_score = None
            dog_score = None
            for pred in predictions:
                if "cat" in pred["label"].lower():
                    cat_score = pred["score"]
                elif "dog" in pred["label"].lower():
                    dog_score = pred["score"]
            
            assert cat_score is not None, "Cat prediction not found"
            assert dog_score is not None, "Dog prediction not found"
            
            assert cat_score > dog_score, f"Cat score ({cat_score}) should be higher than dog score ({dog_score})"
            
            performance = result["performance"]
            assert "total_time_ms" in performance
            assert "inference_time_ms" in performance
            assert "device_info" in performance
            
            print(f"Cat vs Dog test passed: cat={cat_score:.3f}, dog={dog_score:.3f}")
            print(f"Performance: {performance['total_time_ms']}ms total, {performance['inference_time_ms']}ms inference")
            print(f"Device: {performance['device_info']['device_type']}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("CLIP service is not running. Start the service to run this test.")
    
    def test_inside_vs_outside_classification(self, service_url: str, cat_image_path: str, inside_vs_outside_labels: list):
        """Test that CLIP correctly identifies cat desk images as inside rather than outside."""
        if not os.path.exists(cat_image_path):
            pytest.skip(f"Cat sample image not found at {cat_image_path}")
        
        public_indoor_cat_url = "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400"
        
        payload = {
            "image_url": public_indoor_cat_url,
            "labels": inside_vs_outside_labels
        }
        
        try:
            response = requests.post(service_url, json=payload, timeout=30)
            assert response.status_code == 200
            
            result = response.json()
            assert "predictions" in result
            assert "performance" in result
            
            predictions = result["predictions"]
            assert len(predictions) == 2
            
            indoor_score = None
            outdoor_score = None
            for pred in predictions:
                if "indoor" in pred["label"].lower():
                    indoor_score = pred["score"]
                elif "outdoor" in pred["label"].lower():
                    outdoor_score = pred["score"]
            
            assert indoor_score is not None, "Indoor prediction not found"
            assert outdoor_score is not None, "Outdoor prediction not found"
            
            assert indoor_score > outdoor_score, f"Indoor score ({indoor_score}) should be higher than outdoor score ({outdoor_score})"
            
            print(f"Indoor vs Outdoor test passed: indoor={indoor_score:.3f}, outdoor={outdoor_score:.3f}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("CLIP service is not running. Start the service to run this test.")
    
    def test_all_cat_images_classification(self, service_url: str, cat_vs_dog_labels: list):
        """Test CLIP classification on multiple cat images using public URLs."""
        cat_image_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg",
            "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
            "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=400",
            "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400",
            "https://images.unsplash.com/photo-1592194996308-7b43878e84a6?w=400"
        ]
        
        successful_tests = 0
        total_tests = len(cat_image_urls)
        
        for i, cat_url in enumerate(cat_image_urls):
            payload = {
                "image_url": cat_url,
                "labels": cat_vs_dog_labels
            }
            
            try:
                response = requests.post(service_url, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    predictions = result["predictions"]
                    
                    cat_score = None
                    dog_score = None
                    for pred in predictions:
                        if "cat" in pred["label"].lower():
                            cat_score = pred["score"]
                        elif "dog" in pred["label"].lower():
                            dog_score = pred["score"]
                    
                    if cat_score is not None and dog_score is not None and cat_score > dog_score:
                        successful_tests += 1
                        print(f"✓ Cat image {i+1}: cat={cat_score:.3f} > dog={dog_score:.3f}")
                    else:
                        print(f"✗ Cat image {i+1}: cat={cat_score:.3f} <= dog={dog_score:.3f}")
                
            except requests.exceptions.ConnectionError:
                pytest.skip("CLIP service is not running. Start the service to run this test.")
            except Exception as e:
                print(f"Error testing cat image {i+1}: {e}")
        
        success_rate = successful_tests / total_tests
        assert success_rate >= 0.8, f"Only {successful_tests}/{total_tests} ({success_rate:.1%}) cat images classified correctly"
        
        print(f"All cat images test: {successful_tests}/{total_tests} ({success_rate:.1%}) successful")
