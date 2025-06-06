import pytest
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any

class TestKubernetesManifests:
    """Test suite for Kubernetes manifest validation."""
    
    @pytest.fixture(scope="class")
    def repo_root(self) -> Path:
        """Get the repository root directory."""
        return Path(__file__).parent.parent
    
    @pytest.fixture(scope="class")
    def k8s_manifests_dir(self, repo_root: Path) -> Path:
        """Get the k8s-manifests directory."""
        return repo_root / "k8s-manifests"
    
    def test_manifest_files_exist(self, k8s_manifests_dir: Path):
        """Test that all required manifest files exist."""
        expected_files = [
            "clip-deployment.yaml",
            "clip-service.yaml",
            "gemma-deployment.yaml", 
            "gemma-service.yaml",
            "grounding-dino-deployment.yaml",
            "grounding-dino-service.yaml",
            "ray_serve_deployment.yaml"
        ]
        
        for filename in expected_files:
            manifest_path = k8s_manifests_dir / filename
            assert manifest_path.exists(), f"Manifest file {filename} not found"
    
    def test_yaml_syntax_valid(self, k8s_manifests_dir: Path):
        """Test that all YAML manifest files have valid syntax."""
        yaml_files = list(k8s_manifests_dir.glob("*.yaml"))
        assert len(yaml_files) > 0, "No YAML files found in k8s-manifests directory"
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                try:
                    documents = list(yaml.safe_load_all(f))
                    assert len(documents) > 0, f"{yaml_file.name} contains no valid YAML documents"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML syntax in {yaml_file.name}: {e}")
    
    def test_deployment_manifests_structure(self, k8s_manifests_dir: Path):
        """Test that deployment manifests have correct structure."""
        deployment_files = [
            "clip-deployment.yaml",
            "gemma-deployment.yaml"
        ]
        
        for filename in deployment_files:
            manifest_path = k8s_manifests_dir / filename
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            assert manifest["apiVersion"] == "apps/v1", f"{filename} should use apps/v1 API version"
            assert manifest["kind"] == "Deployment", f"{filename} should be a Deployment"
            assert "metadata" in manifest, f"{filename} missing metadata"
            assert "spec" in manifest, f"{filename} missing spec"
            
            metadata = manifest["metadata"]
            assert "name" in metadata, f"{filename} missing metadata.name"
            assert "labels" in metadata, f"{filename} missing metadata.labels"
            
            spec = manifest["spec"]
            assert "replicas" in spec, f"{filename} missing spec.replicas"
            assert "selector" in spec, f"{filename} missing spec.selector"
            assert "template" in spec, f"{filename} missing spec.template"
            
            template = spec["template"]
            assert "metadata" in template, f"{filename} missing spec.template.metadata"
            assert "spec" in template, f"{filename} missing spec.template.spec"
            
            pod_spec = template["spec"]
            assert "containers" in pod_spec, f"{filename} missing containers"
            assert len(pod_spec["containers"]) > 0, f"{filename} should have at least one container"
            
            container = pod_spec["containers"][0]
            assert "name" in container, f"{filename} container missing name"
            assert "image" in container, f"{filename} container missing image"
            assert "ports" in container, f"{filename} container missing ports"
    
    def test_service_manifests_structure(self, k8s_manifests_dir: Path):
        """Test that service manifests have correct structure."""
        service_files = [
            "clip-service.yaml",
            "gemma-service.yaml",
            "grounding-dino-service.yaml"
        ]
        
        for filename in service_files:
            manifest_path = k8s_manifests_dir / filename
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            assert manifest["apiVersion"] == "v1", f"{filename} should use v1 API version"
            assert manifest["kind"] == "Service", f"{filename} should be a Service"
            assert "metadata" in manifest, f"{filename} missing metadata"
            assert "spec" in manifest, f"{filename} missing spec"
            
            spec = manifest["spec"]
            assert "selector" in spec, f"{filename} missing spec.selector"
            assert "ports" in spec, f"{filename} missing spec.ports"
            assert len(spec["ports"]) > 0, f"{filename} should have at least one port"
            
            port = spec["ports"][0]
            assert "port" in port, f"{filename} port missing port number"
            assert "targetPort" in port, f"{filename} port missing targetPort"
    
    def test_resource_requirements(self, k8s_manifests_dir: Path):
        """Test that deployments specify resource requirements."""
        deployment_files = [
            "clip-deployment.yaml",
            "gemma-deployment.yaml"
        ]
        
        for filename in deployment_files:
            manifest_path = k8s_manifests_dir / filename
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            container = manifest["spec"]["template"]["spec"]["containers"][0]
            assert "resources" in container, f"{filename} container missing resources"
            
            resources = container["resources"]
            assert "limits" in resources or "requests" in resources, f"{filename} should specify resource limits or requests"
            
            if "limits" in resources:
                limits = resources["limits"]
                if "nvidia.com/gpu" in limits:
                    assert limits["nvidia.com/gpu"] >= 1, f"{filename} should request at least 1 GPU"
    
    def test_node_selector_configuration(self, k8s_manifests_dir: Path):
        """Test that GPU services have node selector configuration."""
        gpu_deployment_files = [
            "clip-deployment.yaml",
            "gemma-deployment.yaml"
        ]
        
        for filename in gpu_deployment_files:
            manifest_path = k8s_manifests_dir / filename
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            pod_spec = manifest["spec"]["template"]["spec"]
            assert "nodeSelector" in pod_spec, f"{filename} should have nodeSelector for GPU scheduling"
    
    def test_label_consistency(self, k8s_manifests_dir: Path):
        """Test that labels are consistent between deployments and services."""
        service_pairs = [
            ("clip-deployment.yaml", "clip-service.yaml"),
            ("gemma-deployment.yaml", "gemma-service.yaml")
        ]
        
        for deployment_file, service_file in service_pairs:
            with open(k8s_manifests_dir / deployment_file, 'r') as f:
                deployment = yaml.safe_load(f)
            
            with open(k8s_manifests_dir / service_file, 'r') as f:
                service = yaml.safe_load(f)
            
            deployment_labels = deployment["spec"]["template"]["metadata"]["labels"]
            
            service_selector = service["spec"]["selector"]
            
            for key, value in service_selector.items():
                assert key in deployment_labels, f"{service_file} selector key '{key}' not found in deployment labels"
                assert deployment_labels[key] == value, f"{service_file} selector value mismatch for key '{key}'"
