import pytest
import requests
import json
import time
import os
from typing import Dict, Any, Callable

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
        project_root = os.path.dirname(os.path.dirname(script_dir))
        return os.path.join(project_root, "sample_data", "cats_on_desk", "cat_on_desk_01.png")
    
    @pytest.fixture(scope="class")
    def dog_image_path(self) -> str:
        """Path to local dog sample image."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        return os.path.join(project_root, "sample_data", "dogs_running", "dog_running_05.png")
    
    @pytest.fixture(scope="class")
    def cat_vs_dog_labels(self) -> list:
        """Labels for cat vs dog classification test."""
        return ["a photo of a cat", "a photo of a dog"]
    
    @pytest.fixture(scope="class")
    def inside_vs_outside_labels(self) -> list:
        """Labels for inside vs outside classification test."""
        return ["indoor scene", "outdoor scene"]
    
    @pytest.fixture(scope="class")
    def image_to_base64(self) -> Callable[[str], str]:
        """Helper function to convert image files to base64."""
        def convert(image_path: str) -> str:
            import base64
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')
        return convert
    
    def test_service_health(self, service_url: str):
        """Test that the service is running and responsive."""
        try:
            response = requests.get(f"{service_url}/health", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("CLIP service is not running. Start the service to run this test.")
    
    def test_clip_classification_valid_request(self, service_url: str, sample_image_url: str, sample_labels: list, image_to_base64: Callable[[str], str]):
        """Test CLIP classification with valid base64-encoded image and labels."""
        import requests as req_lib
        try:
            img_response = req_lib.get(sample_image_url)
            img_response.raise_for_status()
            import base64
            sample_base64 = base64.b64encode(img_response.content).decode('utf-8')
        except Exception:
            import base64
            from PIL import Image
            import io
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            sample_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        payload = {
            "image": sample_base64,
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
        """Test CLIP classification with missing image data."""
        payload = {
            "labels": sample_labels
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
        assert "image" in result["error"]
    
    def test_clip_classification_missing_labels(self, service_url: str, sample_image_url: str):
        """Test CLIP classification with missing labels."""
        import requests as req_lib
        try:
            img_response = req_lib.get(sample_image_url)
            img_response.raise_for_status()
            import base64
            sample_base64 = base64.b64encode(img_response.content).decode('utf-8')
        except Exception:
            import base64
            from PIL import Image
            import io
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            sample_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        payload = {
            "image": sample_base64
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
        assert "labels" in result["error"]
    
    def test_clip_classification_empty_labels(self, service_url: str, sample_image_url: str):
        """Test CLIP classification with empty labels list."""
        import requests as req_lib
        try:
            img_response = req_lib.get(sample_image_url)
            img_response.raise_for_status()
            import base64
            sample_base64 = base64.b64encode(img_response.content).decode('utf-8')
        except Exception:
            import base64
            from PIL import Image
            import io
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            sample_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        payload = {
            "image": sample_base64,
            "labels": []
        }
        
        response = requests.post(service_url, json=payload, timeout=10)
        result = response.json()
        
        assert "error" in result
    
    def test_clip_classification_invalid_image_data(self, service_url: str, sample_labels: list):
        """Test CLIP classification with invalid base64 image data."""
        payload = {
            "image": "invalid_base64_data_that_cannot_be_decoded",
            "labels": sample_labels
        }
        
        response = requests.post(service_url, json=payload, timeout=30)
        result = response.json()
        assert "error" in result or response.status_code != 200
    
    def test_clip_classification_single_label(self, service_url: str, sample_image_url: str):
        """Test CLIP classification with a single label."""
        import requests as req_lib
        try:
            img_response = req_lib.get(sample_image_url)
            img_response.raise_for_status()
            import base64
            sample_base64 = base64.b64encode(img_response.content).decode('utf-8')
        except Exception:
            import base64
            from PIL import Image
            import io
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            sample_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        payload = {
            "image": sample_base64,
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
        
        import requests as req_lib
        try:
            img_response = req_lib.get(sample_image_url)
            img_response.raise_for_status()
            import base64
            sample_base64 = base64.b64encode(img_response.content).decode('utf-8')
        except Exception:
            import base64
            from PIL import Image
            import io
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            sample_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        payload = {
            "image": sample_base64,
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
        import requests as req_lib
        try:
            img_response = req_lib.get(sample_image_url)
            img_response.raise_for_status()
            import base64
            sample_base64 = base64.b64encode(img_response.content).decode('utf-8')
        except Exception:
            import base64
            from PIL import Image
            import io
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            sample_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        payload = {
            "image": sample_base64,
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
    
    def test_cat_vs_dog_classification(self, service_url: str, cat_image_path: str, dog_image_path: str, cat_vs_dog_labels: list, image_to_base64: Callable[[str], str]):
        """Test that CLIP correctly identifies cat images as cats rather than dogs using local sample images."""
        if not os.path.exists(cat_image_path):
            pytest.skip(f"Cat sample image not found at {cat_image_path}")
        if not os.path.exists(dog_image_path):
            pytest.skip(f"Dog sample image not found at {dog_image_path}")
        
        cat_base64 = image_to_base64(cat_image_path)
        cat_payload = {
            "image": cat_base64,
            "labels": cat_vs_dog_labels
        }
        
        try:
            cat_response = requests.post(service_url, json=cat_payload, timeout=30)
            assert cat_response.status_code == 200
            cat_result = cat_response.json()
            assert "predictions" in cat_result
            
            cat_predictions = cat_result["predictions"]
            assert len(cat_predictions) == 2
            
            cat_score = None
            dog_score = None
            for pred in cat_predictions:
                if "cat" in pred["label"].lower():
                    cat_score = pred["score"]
                elif "dog" in pred["label"].lower():
                    dog_score = pred["score"]
            
            assert cat_score is not None, "Cat prediction not found"
            assert dog_score is not None, "Dog prediction not found"
            assert cat_score > dog_score, f"Cat image: cat score ({cat_score}) should be higher than dog score ({dog_score})"
            
            dog_base64 = image_to_base64(dog_image_path)
            dog_payload = {
                "image": dog_base64,
                "labels": cat_vs_dog_labels
            }
            
            dog_response = requests.post(service_url, json=dog_payload, timeout=30)
            assert dog_response.status_code == 200
            dog_result = dog_response.json()
            assert "predictions" in dog_result
            
            dog_predictions = dog_result["predictions"]
            assert len(dog_predictions) == 2
            
            dog_cat_score = None
            dog_dog_score = None
            for pred in dog_predictions:
                if "cat" in pred["label"].lower():
                    dog_cat_score = pred["score"]
                elif "dog" in pred["label"].lower():
                    dog_dog_score = pred["score"]
            
            assert dog_cat_score is not None, "Cat prediction not found for dog image"
            assert dog_dog_score is not None, "Dog prediction not found for dog image"
            assert dog_dog_score > dog_cat_score, f"Dog image: dog score ({dog_dog_score}) should be higher than cat score ({dog_cat_score})"
            
            print(f"Cat vs Dog test passed:")
            print(f"  Cat image: cat={cat_score:.3f} > dog={dog_score:.3f}")
            print(f"  Dog image: dog={dog_dog_score:.3f} > cat={dog_cat_score:.3f}")
            
            if "performance" in cat_result:
                performance = cat_result["performance"]
                print(f"Performance: {performance.get('total_time_ms', 'N/A')}ms total")
                print(f"Device: {performance.get('device_info', {}).get('device_type', 'unknown')}")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("CLIP service is not running. Start the service to run this test.")
    
    def test_inside_vs_outside_classification(self, service_url: str, cat_image_path: str, inside_vs_outside_labels: list):
        """Test that CLIP correctly identifies cat desk images as inside rather than outside."""
        if not os.path.exists(cat_image_path):
            pytest.skip(f"Cat sample image not found at {cat_image_path}")
        
        public_indoor_cat_url = "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400"
        
        import requests as req_lib
        try:
            img_response = req_lib.get(public_indoor_cat_url)
            img_response.raise_for_status()
            import base64
            sample_base64 = base64.b64encode(img_response.content).decode('utf-8')
        except Exception:
            import base64
            from PIL import Image
            import io
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            sample_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        payload = {
            "image": sample_base64,
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
            
            assert indoor_score + outdoor_score > 0.8, f"Combined scores too low: indoor={indoor_score}, outdoor={outdoor_score}"
            print(f"Indoor vs Outdoor classification: indoor={indoor_score:.3f}, outdoor={outdoor_score:.3f}")
            
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
            import requests as req_lib
            try:
                img_response = req_lib.get(cat_url)
                img_response.raise_for_status()
                import base64
                cat_base64 = base64.b64encode(img_response.content).decode('utf-8')
            except Exception:
                import base64
                from PIL import Image
                import io
                test_image = Image.new('RGB', (100, 100), color='red')
                img_buffer = io.BytesIO()
                test_image.save(img_buffer, format='JPEG')
                cat_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            payload = {
                "image": cat_base64,
                "labels": cat_vs_dog_labels
            }
            
            try:
                response = requests.post(service_url, json=payload, timeout=30)
                assert response.status_code == 200
                
                result = response.json()
                assert "predictions" in result
                
                predictions = result["predictions"]
                assert len(predictions) == 2
                
                cat_score = None
                dog_score = None
                for pred in predictions:
                    if "cat" in pred["label"].lower():
                        cat_score = pred["score"]
                    elif "dog" in pred["label"].lower():
                        dog_score = pred["score"]
                
                assert cat_score is not None, f"Cat prediction not found for image {i+1}"
                assert dog_score is not None, f"Dog prediction not found for image {i+1}"
                
                if cat_score > dog_score:
                    successful_tests += 1
                    print(f"✓ Image {i+1}: cat={cat_score:.3f} > dog={dog_score:.3f}")
                else:
                    print(f"✗ Image {i+1}: cat={cat_score:.3f} < dog={dog_score:.3f}")
                    
            except requests.exceptions.ConnectionError:
                pytest.skip("CLIP service is not running. Start the service to run this test.")
            except Exception as e:
                print(f"✗ Image {i+1} failed: {e}")
        
        success_rate = successful_tests / total_tests
        print(f"Overall success rate: {successful_tests}/{total_tests} ({success_rate:.1%})")
        
        assert success_rate >= 0.6, f"Success rate {success_rate:.1%} is below 60% threshold"
