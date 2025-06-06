# Ray Serve Inference Menagerie

A production-grade machine learning model serving platform built on Ray Serve and Kubernetes. Deploy and manage a diverse collection ("menagerie") of ML inference models as scalable HTTP services.

## Overview

The Ray Serve Inference Menagerie provides enterprise-ready ML model serving with:

- **Advanced Object Detection**: Production-ready GroundingDinoService combining Grounding DINO and SAM2 models
- **Foundation Model Services**: Template implementations for CLIP and Gemma3 models  
- **Kubernetes-Native Deployment**: CLI-driven deployment with GPU scheduling and multi-user isolation
- **Enterprise Observability**: Full Datadog integration for metrics, logging, and tracing

## Services

### 🎯 GroundingDinoService (Production Ready)
Advanced AI service combining Grounding DINO and SAM2 models for object detection and segmentation.

**Endpoints:**
- `POST /grounding-dino/text_detect` - Text-prompted object detection with bounding boxes
- `POST /grounding-dino/detect` - Universal object detection using SAM2

**Example Usage:**
```bash
# Text-prompted detection
curl -X POST "http://localhost:8000/grounding-dino/text_detect" \
  -F "image=@image.jpg" \
  -F "text=person walking dog"

# Universal detection
curl -X POST "http://localhost:8000/grounding-dino/detect" \
  -F "image=@image.jpg"
```

### 🖼️ CLIPService (Template)
Zero-shot image classification using OpenAI CLIP model.

**Endpoint:** `POST /clip`
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
         "labels": ["a photo of a cat", "a photo of a dog", "a photo of a car"]}' \
    http://localhost:8080/
```

### 💬 GemmaService (Template)  
Vision-language generation using Google Gemma3 model.

**Endpoint:** `POST /gemma`
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
         "prompt": "What animal is on the candy?"}' \
    http://localhost:8081/
```

## Quick Start

### Prerequisites
- Kubernetes cluster with GPU nodes
- Docker and kubectl installed
- Container registry access
- Python 3.8+ for local development

### 1. Setup Dependencies
```bash
# Install dependencies
make setup
# or
./scripts/setup_dependencies.sh --dev
```

### 2. Build Docker Images
```bash
# Build all images
make docker-build

# Build specific service
docker build -t your-registry/clip-service:latest ./clip-service
docker build -t your-registry/gemma-service:latest ./gemma-service
docker build -t your-registry/grounding-dino-service:latest ./grounding-dino-service
```

### 3. Deploy to Kubernetes
```bash
# Deploy GroundingDinoService with GPU scheduling
./scripts/deploy_ray_service.sh grounding-dino \
  --node-selector accelerator=nvidia-tesla-t4 \
  --namespace ml-services

# Deploy CLIP service
./scripts/deploy_ray_service.sh clip \
  --username alice \
  --replicas 2

# Deploy Gemma service
./scripts/deploy_ray_service.sh gemma \
  --username bob \
  --node-selector gpu=true
```

### 4. Access Services
```bash
# Port forward to access locally
kubectl port-forward svc/alice-clip-service 8080:80
kubectl port-forward svc/bob-gemma-service 8081:80
kubectl port-forward svc/grounding-dino-service 8082:80

# Test CLIP service
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
         "labels": ["a photo of a cat", "a photo of a dog", "a photo of a car"]}' \
    http://localhost:8080/

# Test Gemma service
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
         "prompt": "What animal is on the candy?"}' \
    http://localhost:8081/
```

## Development

### Run Locally
```bash
# Start all services locally
make run
# or
./scripts/run_local.sh

# Start specific service
./scripts/run_local.sh grounding-dino --port 8000
```

### Testing
```bash
# Run all tests
make test

# Run integration tests (requires services running)
make test-integration

# Test Docker builds
make test-docker

# Test Kubernetes manifests
make test-k8s

# Run specific service tests
pytest tests/test_clip_service.py -v
pytest tests/test_grounding_dino_service.py -v
```

### Linting
```bash
# Check code style
make lint

# Auto-fix issues
make lint-fix
```

## Architecture

The platform is built around three main components:

1. **Service Implementations** (`*-service/`) - Ray Serve applications with Docker containers
2. **Kubernetes Manifests** (`k8s-manifests/`) - Deployment configurations with GPU support
3. **Deployment Tooling** (`scripts/`) - CLI automation for multi-user deployments

### Directory Structure
```
ray-serve-inference-menagerie/
├── clip-service/           # CLIP zero-shot classification
│   ├── app.py             # Ray Serve implementation
│   └── Dockerfile         # Container definition
├── gemma-service/          # Gemma3 vision-language model
│   ├── app.py
│   └── Dockerfile
├── grounding-dino-service/ # Production object detection
│   ├── app.py
│   └── Dockerfile
├── k8s-manifests/          # Kubernetes deployments
│   ├── *-deployment.yaml  # Service deployments
│   ├── *-service.yaml     # Service definitions
│   └── ray_serve_deployment.yaml # Generic template
├── scripts/                # Automation tools
│   ├── deploy_ray_service.sh # CLI deployment
│   ├── setup_dependencies.sh # Environment setup
│   ├── lint.sh            # Code quality
│   ├── test.sh            # Test runner
│   └── run_local.sh       # Local development
└── tests/                  # Test suite
    ├── test_*_service.py  # Service integration tests
    ├── test_docker_builds.py # Container tests
    └── test_kubernetes_manifests.py # K8s validation
```

## Configuration

### Environment Variables
- `CLIP_SERVICE_URL` - CLIP service endpoint (default: http://localhost:8000)
- `GEMMA_SERVICE_URL` - Gemma service endpoint (default: http://localhost:8001)  
- `GROUNDING_DINO_SERVICE_URL` - Grounding DINO endpoint (default: http://localhost:8002)

### GPU Node Scheduling
Services automatically schedule on GPU nodes using node selectors:
```yaml
nodeSelector:
  accelerator: "nvidia-tesla-t4"  # Configure for your cluster
```

### Multi-User Isolation
All deployments are prefixed with username for isolation:
```bash
# Creates resources like: alice-grounding-dino-service
./scripts/deploy_ray_service.sh grounding-dino --username alice
```

## Monitoring

### Datadog Integration
Services include built-in Datadog annotations for:
- Prometheus metrics scraping (port 8080)
- Log collection with service tagging
- Distributed tracing support

### Health Checks
All services expose health endpoints:
```bash
curl http://localhost:8000/health
```

## Customization and Next Steps

### Adding New Services
1. Create service directory with `app.py` and `Dockerfile`
2. Add Kubernetes manifests in `k8s-manifests/`
3. Update deployment script to support new service
4. Add comprehensive tests in `tests/`

### Scaling and Production
- Configure horizontal pod autoscaling based on metrics
- Set up ingress controllers for external access
- Implement proper secret management for model weights
- Configure persistent volumes for model caching

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-service`
3. Make changes and add tests
4. Run the test suite: `make test`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Architecture Documentation

For detailed implementation specifications, see:
- [ARCHITECTURE_ROADMAP_1.md](ARCHITECTURE_ROADMAP_1.md) - Foundation service templates
- [ARCHITECTURE_ROADMAP_2.md](ARCHITECTURE_ROADMAP_2.md) - Deployment and observability  
- [ARCHITECTURE_ROADMAP_3.md](ARCHITECTURE_ROADMAP_3.md) - Production GroundingDinoService
