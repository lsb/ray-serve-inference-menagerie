import pytest
import subprocess
import os
from pathlib import Path

class TestDockerBuilds:
    """Test suite for Docker image builds."""
    
    @pytest.fixture(scope="class")
    def repo_root(self) -> Path:
        """Get the repository root directory."""
        return Path(__file__).parent.parent
    
    def test_clip_service_docker_build(self, repo_root: Path):
        """Test that CLIP service Docker image builds successfully."""
        clip_dir = repo_root / "clip-service"
        
        dockerfile = clip_dir / "Dockerfile"
        assert dockerfile.exists(), "CLIP service Dockerfile not found"
        
        app_file = clip_dir / "app.py"
        assert app_file.exists(), "CLIP service app.py not found"
        
        result = subprocess.run([
            "docker", "build", "-t", "clip-service:test", str(clip_dir)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Docker build stdout: {result.stdout}")
            print(f"Docker build stderr: {result.stderr}")
        
        assert result.returncode == 0, f"CLIP service Docker build failed: {result.stderr}"
    
    def test_gemma_service_docker_build(self, repo_root: Path):
        """Test that Gemma service Docker image builds successfully."""
        gemma_dir = repo_root / "gemma-service"
        
        dockerfile = gemma_dir / "Dockerfile"
        assert dockerfile.exists(), "Gemma service Dockerfile not found"
        
        app_file = gemma_dir / "app.py"
        assert app_file.exists(), "Gemma service app.py not found"
        
        result = subprocess.run([
            "docker", "build", "-t", "gemma-service:test", str(gemma_dir)
        ], capture_output=True, text=True, timeout=600)  # Longer timeout for larger model
        
        if result.returncode != 0:
            print(f"Docker build stdout: {result.stdout}")
            print(f"Docker build stderr: {result.stderr}")
        
        assert result.returncode == 0, f"Gemma service Docker build failed: {result.stderr}"
    
    def test_grounding_dino_service_docker_build(self, repo_root: Path):
        """Test that Grounding DINO service Docker image builds successfully."""
        grounding_dino_dir = repo_root / "grounding-dino-service"
        
        dockerfile = grounding_dino_dir / "Dockerfile"
        assert dockerfile.exists(), "Grounding DINO service Dockerfile not found"
        
        app_file = grounding_dino_dir / "app.py"
        assert app_file.exists(), "Grounding DINO service app.py not found"
        
        result = subprocess.run([
            "docker", "build", "-t", "grounding-dino-service:test", str(grounding_dino_dir)
        ], capture_output=True, text=True, timeout=900)  # Longer timeout for complex dependencies
        
        if result.returncode != 0:
            print(f"Docker build stdout: {result.stdout}")
            print(f"Docker build stderr: {result.stderr}")
        
        assert result.returncode == 0, f"Grounding DINO service Docker build failed: {result.stderr}"
    
    def test_docker_images_exist(self):
        """Test that built Docker images exist and can be listed."""
        result = subprocess.run([
            "docker", "images", "--format", "table {{.Repository}}:{{.Tag}}"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, "Failed to list Docker images"
        
        images_output = result.stdout
        expected_images = [
            "clip-service:test",
            "gemma-service:test", 
            "grounding-dino-service:test"
        ]
        
        for image in expected_images:
            assert image in images_output, f"Docker image {image} not found in image list"
    
    @pytest.mark.slow
    def test_docker_image_sizes(self):
        """Test that Docker images are within reasonable size limits."""
        size_limits = {
            "clip-service:test": 5.0,  # GB
            "gemma-service:test": 10.0,  # GB
            "grounding-dino-service:test": 15.0  # GB
        }
        
        for image, max_size_gb in size_limits.items():
            result = subprocess.run([
                "docker", "images", image, "--format", "{{.Size}}"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"Failed to get size for image {image}"
            
            size_str = result.stdout.strip()
            if size_str:
                if "GB" in size_str:
                    size_gb = float(size_str.replace("GB", ""))
                elif "MB" in size_str:
                    size_gb = float(size_str.replace("MB", "")) / 1000
                else:
                    continue
                
                assert size_gb <= max_size_gb, f"Image {image} size {size_gb}GB exceeds limit {max_size_gb}GB"
    
    def test_dockerfile_best_practices(self, repo_root: Path):
        """Test that Dockerfiles follow best practices."""
        services = ["clip-service", "gemma-service", "grounding-dino-service"]
        
        for service in services:
            dockerfile_path = repo_root / service / "Dockerfile"
            assert dockerfile_path.exists(), f"{service} Dockerfile not found"
            
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
            
            assert "FROM" in dockerfile_content, f"{service} Dockerfile missing FROM instruction"
            assert "COPY" in dockerfile_content or "ADD" in dockerfile_content, f"{service} Dockerfile missing COPY/ADD instruction"
            assert "CMD" in dockerfile_content or "ENTRYPOINT" in dockerfile_content, f"{service} Dockerfile missing CMD/ENTRYPOINT"
            
            lines = dockerfile_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('RUN apt-get update'):
                    assert 'rm -rf /var/lib/apt/lists/*' in dockerfile_content, f"{service} Dockerfile should clean apt cache"
