The goal is to deploy a variety of ML models for vision and language inference using Ray Serve on Kubernetes.

This is part of a compilation of sketches of the architecture, before starting to build.

---


# Ray Serve Model Deployment (CLIP & Gemma3) Skeleton Repository

This repository provides a skeleton for deploying two machine learning models using Ray Serve to Kubernetes (compatible with both GKE and on-prem clusters). It includes a Hugging Face CLIP model service for zero-shot image classification and an open vision-language model (VLM) service using a Gemma 3 model. Both services are served via HTTP endpoints using Ray Serve with GPU support. Kubernetes manifests are provided with nodeSelectors and annotations for scheduling on GPU nodes (with other configurations left to defaults). Docker is used for containerizing the services. Note: This is a starting point (a "shell" repository) meant for customization. It omits CI/CD and advanced Ray cluster setup for simplicity.

```
Repository Structure

ray-serve-ml-deployment/
├── clip-service/
│   ├── Dockerfile           # Dockerfile for CLIP service container
│   └── app.py               # Python entrypoint using Ray Serve for CLIP model
├── gemma-service/
│   ├── Dockerfile           # Dockerfile for Gemma3 VLM service container
│   └── app.py               # Python entrypoint using Ray Serve for VLM model
├── k8s-manifests/
│   ├── clip-deployment.yaml # Kubernetes Deployment for CLIP service
│   ├── clip-service.yaml    # Kubernetes Service for CLIP service
│   ├── gemma-deployment.yaml# Kubernetes Deployment for Gemma3 service
│   ├── gemma-service.yaml   # Kubernetes Service for Gemma3 service
└── README.md                # Instructions to build, deploy, and use the services
```

## Dockerfile for CLIP Service

The CLIP service container uses a Python base image with CUDA support, installs Ray Serve, Hugging Face Transformers, and PyTorch with GPU support. It then launches the Ray Serve app on container start.

```
# ./clip-service/Dockerfile
FROM python:3.10-slim

# Install system packages (if needed, e.g., for PIL image processing). 
# (Slim base may require libgl1, libglib2.0-0 etc., but PIL wheels often suffice.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies: PyTorch (with CUDA support), Transformers, Ray[Serve], etc.
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
      -f https://download.pytorch.org/whl/cu118/torch_stable.html && \
    pip install --no-cache-dir transformers==4.50.0 pillow==9.5.0 requests==2.31.0 \
      "ray[serve]==2.5.1"

# Copy the service code
COPY app.py /app.py

# Run the Ray Serve application (starts the server on port 8000)
ENV PORT 8000
CMD ["python", "/app.py"]
```

Notes:
We pin specific versions for reproducibility (e.g., Transformers 4.50.0, which supports Gemma 3 models, and a PyTorch version with CUDA 11.8). Adjust versions as needed.
The base image and installed packages ensure GPU support (the PyTorch install includes CUDA binaries). The container should be run with an NVIDIA-compatible runtime on GPU nodes. Ray Serve is included via ray[serve].
The CMD launches the app.py which will initialize Ray Serve and host the model.

## Dockerfile for Gemma3 VLM Service

The Gemma3 service is similar, installing the Gemma model's requirements. It can share much of the base environment with the CLIP service. In this skeleton we use a similar setup and rely on the Hugging Face Transformers pipeline to download the Gemma 3 model at runtime.

```
# ./gemma-service/Dockerfile
FROM python:3.10-slim

# (Similar base setup as CLIP Dockerfile; could be optimized by using a common base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
      -f https://download.pytorch.org/whl/cu118/torch_stable.html && \
    pip install --no-cache-dir transformers==4.50.0 pillow==9.5.0 requests==2.31.0 \
      "ray[serve]==2.5.1"

COPY app.py /app.py

ENV PORT 8000
CMD ["python", "/app.py"]
```

Both Dockerfiles use the same base dependencies for simplicity. In a real setup, you might tailor the image (e.g., exclude unneeded packages or include model files to avoid downloading at runtime).

## CLIP Service Ray Serve Application (clip-service/app.py)

This Python entrypoint uses Ray Serve to create a HTTP endpoint for the CLIP model. It utilizes Hugging Face's CLIP via the zero-shot-image-classification pipeline to classify images into given text labels. The Ray Serve deployment is configured to use 1 GPU if available.

```
# ./clip-service/app.py
import os, time, signal, asyncio
import ray
from ray import serve
import torch
from transformers import pipeline

# Initialize Ray and Ray Serve
ray.init()  # Starts a local Ray runtime (or connects to one if specified by environment)
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))})

# Load the CLIP model pipeline (zero-shot image classification)
# The CLIP model compares image features and text features in a shared space:contentReference[oaicite:2]{index=2}.
clip_pipeline = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    device=0 if torch.cuda.is_available() else -1,  # use GPU if available
    # torch_dtype=torch.bfloat16 could be set for efficiency, requires torch>=1.10
)

@serve.deployment(route_prefix="/", ray_actor_options={"num_gpus": 1})
class CLIPService:
    def __init__(self):
        self.pipeline = clip_pipeline
    async def __call__(self, request):
        # Expect JSON with an image URL and a list of candidate labels
        data = await request.json()
        image_url = data.get("image_url")
        candidate_labels = data.get("labels")
        if not image_url or not candidate_labels:
            return {"error": "Provide 'image_url' and 'labels' in request JSON."}
        # Run the CLIP pipeline to get classification results
        result = self.pipeline(image_url, candidate_labels=candidate_labels)
        return result  # e.g., list of {"label": ..., "score": ...} dicts

# Deploy the service (1 replica by default)
CLIPService.deploy()

# Block indefinitely to keep the service running (Ray Serve runs in the background)
signal.signal(signal.SIGTERM, lambda sig, frame: exit(0))
print("CLIP service is running on Ray Serve (port {}), awaiting requests...".format(os.environ.get("PORT", 8000)))
while True:
    time.sleep(60)
```

Key points:
We use serve.start(detached=True, http_options={"host": "0.0.0.0", ...}) to ensure the HTTP server listens on the container's interface (allowing external requests) on the given port.
The CLIP pipeline is loaded once in the global scope (before the class) to avoid reloading in each replica. It uses the OpenAI CLIP ViT-B/32 model via Hugging Face Transformers. This model computes an image and text embedding and scores their similarity.
The CLIPService deployment is defined with route_prefix="/" (so it serves at the root path of this service) and ray_actor_options={"num_gpus": 1} to allocate one GPU to the Ray actor running the model. Ray Serve ensures the model runs on the GPU resource.
The __call__ method is async to accommodate asynchronous request handling. It expects a JSON payload with an "image_url" (pointing to an image file accessible by the service) and a list of "labels" (candidate text descriptions). The CLIP pipeline then returns a classification result (the most likely label and score, or a list of scores for each label).
After deploying the service with CLIPService.deploy(), we keep the process alive (using an infinite loop and handling SIGTERM for graceful shutdown). This ensures the Ray Serve HTTP server stays up to handle incoming requests.

## Gemma3 VLM Service Ray Serve Application (gemma-service/app.py)

This entrypoint serves a multimodal vision-language model (Gemma 3) via an HTTP endpoint. We use Hugging Face's pipeline for "image-text-to-text" tasks, which generates text (e.g. answers or descriptions) given an image and optional text prompt. The Gemma 3 model is an open multimodal model by Google that accepts image+text input and produces text output.

```
# ./gemma-service/app.py
import os, time, signal, asyncio
import ray
from ray import serve
import torch
from transformers import pipeline

ray.init()
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))})

# Initialize the Gemma 3 pipeline for image-to-text generation
vlm_pipeline = pipeline(
    "image-text-to-text",
    model="google/gemma-3-1b-it",  # 1B parameter Gemma3 model (instruction-tuned variant)
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.bfloat16  # use bfloat16 for efficiency on modern GPUs
)

@serve.deployment(route_prefix="/", ray_actor_options={"num_gpus": 1})
class GemmaService:
    def __init__(self):
        self.pipeline = vlm_pipeline
    async def __call__(self, request):
        # Expect JSON with an image URL and an optional text prompt
        data = await request.json()
        image_url = data.get("image_url")
        user_prompt = data.get("prompt", "") or ""
        if not image_url:
            return {"error": "Provide 'image_url' (and optionally 'prompt') in request JSON."}
        # Construct the input in Gemma's chat format (system + user message with image)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        # Run the pipeline to generate a response
        output = self.pipeline(text=messages, max_new_tokens=200)
        # Extract the assistant's answer from the output structure
        try:
            generated_messages = output[0]["generated_text"]
            # Assuming the pipeline returns a list of message dicts, get the last message content
            answer = generated_messages[-1]["content"] if isinstance(generated_messages, list) else generated_messages
        except Exception as e:
            answer = None
        return {"answer": answer}

# Deploy the service
GemmaService.deploy()

# Keep the service running
signal.signal(signal.SIGTERM, lambda sig, frame: exit(0))
print("Gemma3 VLM service is running on Ray Serve (port {}), awaiting requests...".format(os.environ.get("PORT", 8000)))
while True:
    time.sleep(60)
```

Key points:

We use the image-text-to-text pipeline with the google/gemma-3-1b-it model (a 1B-parameter Gemma 3 variant). This pipeline loads the vision encoder and language model to allow image-conditioned text generation. (Larger variants like 4B, 12B, 27B are available, but 1B is used here for a lighter example.)
The service expects an "image_url" and an optional "prompt" in the JSON request. We format the input as a chat history: a system message to establish assistant behavior, and a user message containing the image and prompt. This follows the usage guidelines for instruction-tuned multimodal models.
The pipeline generates a list of messages as output (the model’s response may be formatted as a series of messages). We extract the content of the last message as the answer. The response JSON contains this "answer" text.
The Ray Serve deployment is again configured with num_gpus: 1 for the actor to use a GPU. Ensure that the model is permitted (some open models require accepting a license on Hugging Face or providing an access token).
Similar to the CLIP service, we keep the process alive after deploying to keep serving requests.

Kubernetes Manifests

In the k8s-manifests/ directory, we provide sample Kubernetes manifests to deploy each service on a Kubernetes cluster. These include Deployments (to run the pods) and Services (to expose the HTTP endpoint within the cluster, and optionally to external clients). The manifests are structured to work on both Google Kubernetes Engine (GKE) and generic on-prem clusters, with placeholders for GPU scheduling labels. Note: These manifests assume you have GPU nodes in your cluster with the NVIDIA Kubernetes device plugin installed. We use nodeSelector to target GPU nodes and request a GPU resource in the container spec. You may need to adjust labels and annotations to match your cluster setup.

## Deployment for CLIP Service

```
# ./k8s-manifests/clip-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: clip-service
  labels:
    app: clip-service
spec:
  replicas: 1  # start with a single replica; scale out as needed
  selector:
    matchLabels:
      app: clip-service
  template:
    metadata:
      labels:
        app: clip-service
      annotations:
        # Example annotation (e.g., to disable sidecar injection, if using service mesh/Istio)
        sidecar.istio.io/inject: "false"
    spec:
      nodeSelector:
        # Ensure the pod runs on a GPU node. For example, on GKE you might label GPU nodes with:
        # "cloud.google.com/gke-accelerator": "nvidia-tesla-t4"
        # On on-prem, you might use a custom label like "node-role.kubernetes.io/gpu: true".
        # Update the key/value below to match your cluster's GPU node label.
        accelerator: "nvidia-tesla-t4"
      containers:
      - name: clip-service
        image: YOUR_REGISTRY/clip-service:latest  # use the built image name/tag
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8000   # Ray Serve HTTP port inside container
        resources:
          limits:
            nvidia.com/gpu: 1     # request one GPU
            memory: 4Gi
            cpu: 1
          requests:
            nvidia.com/gpu: 1
            memory: 4Gi
            cpu: 0.5
        # You might include environment variables or volume mounts here if needed (e.g., for config or model storage).
```

A few things to note in this Deployment spec:
nodeSelector: This is set to schedule the pod on a node with a GPU. In the example, we use a generic label accelerator: nvidia-tesla-t4, assuming the cluster admin labeled GPU nodes accordingly. Replace this with appropriate label keys/values for your cluster. (On GKE, a common label is cloud.google.com/gke-accelerator with the GPU type as value. On on-prem, you might have a label like gpu: true or similar.)
annotations: We included an example annotation to disable Istio sidecar injection. You can include any needed annotations (for example, for monitoring, networking, or GPU drivers) here. By default, no special annotations are required for basic operation.
resources: The container requests and limits 1 GPU (nvidia.com/gpu: 1). It also specifies some CPU and memory requests/limits. Adjust these based on the model’s needs and your cluster’s scheduling requirements. This ensures Kubernetes knows the pod requires a GPU and reserves it (the NVIDIA device plugin advertises nvidia.com/gpu resources).

## Service for CLIP Service

```
# ./k8s-manifests/clip-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: clip-service
  labels:
    app: clip-service
spec:
  selector:
    app: clip-service   # targets the pods with label app: clip-service
  ports:
    - name: http
      port: 80
      targetPort: 8000  # container's port
      protocol: TCP
  type: ClusterIP
```

This defines a ClusterIP service (internal Kubernetes service) on port 80 forwarding to the pod's port 8000 (where Ray Serve is running). You can change the type to LoadBalancer for external access (on GKE this would provision a cloud load balancer). For on-prem, you might use NodePort or an Ingress for external exposure. By default, we keep it internal and you can use port-forwarding or ingress as needed.

## Deployment for Gemma3 Service

```
# ./k8s-manifests/gemma-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gemma-service
  labels:
    app: gemma-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gemma-service
  template:
    metadata:
      labels:
        app: gemma-service
      annotations:
        sidecar.istio.io/inject: "false"
    spec:
      nodeSelector:
        accelerator: "nvidia-tesla-t4"   # update to match your GPU node label
      containers:
      - name: gemma-service
        image: YOUR_REGISTRY/gemma-service:latest
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 2
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 1
```

The Gemma3 deployment is similar to the CLIP deployment, but we increased the resource requests assuming the model is larger (for example, 1B+ parameter model might need more memory and CPU). Adjust resources as appropriate for the specific model variant you use. The nodeSelector again targets a GPU node.

## Service for Gemma3 Service

```
# ./k8s-manifests/gemma-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gemma-service
  labels:
    app: gemma-service
spec:
  selector:
    app: gemma-service
  ports:
    - name: http
      port: 80
      targetPort: 8000
      protocol: TCP
  type: ClusterIP
```

This is analogous to the CLIP Service. It exposes the Gemma service internally on port 80. Change to LoadBalancer or add an Ingress if you need external access to this endpoint.

## Usage Guide (Building, Deploying, and Calling the Services)

Below is a brief guide, as would be included in the README.md, explaining how to build the Docker images, deploy them to Kubernetes, and interact with the running services.

1. Prerequisites
Kubernetes cluster with GPU nodes available (e.g., GKE with GPU node pool, or an on-prem cluster with NVIDIA GPUs). Make sure the NVIDIA device plugin is installed so that nvidia.com/gpu resources are available on nodes.
Docker installed locally to build images (or use a CI/build service).
(Optional) A container registry (like Docker Hub, GCR, ECR) to push the images for the cluster to pull.
Ensure you have kubectl access to your cluster and the context is set correctly. If on GKE, also ensure you have gcloud configured and credentials up to date for the cluster.

2. Building the Docker Images
First, clone this repository and navigate to its root. Then build the images:

```
# Build the CLIP service image
docker build -t YOUR_REGISTRY/clip-service:latest ./clip-service

# Build the Gemma3 VLM service image
docker build -t YOUR_REGISTRY/gemma-service:latest ./gemma-service
```

Replace YOUR_REGISTRY/... with the image name (and registry prefix) you want to use. For example, gcr.io/your-project/clip-service:v0.1 if using Google Container Registry, or your-dockerhub-user/clip-service:latest for Docker Hub. If you prefer to not use a registry and your cluster supports local image loading (like kind or minikube), you can skip pushing and load images directly.

After building, push the images to your registry (if using):

```
docker push YOUR_REGISTRY/clip-service:latest
docker push YOUR_REGISTRY/gemma-service:latest
```

Make sure the Kubernetes cluster nodes have access to this registry (for GKE, GCR is accessible if same project; for others ensure proper imagePullSecrets if needed).

3. Deploying to Kubernetes

Update the Kubernetes manifest files (clip-deployment.yaml and gemma-deployment.yaml) to use the correct image references (replace YOUR_REGISTRY/... with the actual image names you built). Also adjust the nodeSelector label values if necessary to match your cluster's GPU node labels (and any other annotations or resource requests as needed). Then apply the manifests:

```
kubectl apply -f k8s-manifests/clip-deployment.yaml
kubectl apply -f k8s-manifests/clip-service.yaml
kubectl apply -f k8s-manifests/gemma-deployment.yaml
kubectl apply -f k8s-manifests/gemma-service.yaml
```

This will create two deployments and services. You can verify the pods are running and the services are created:

```
kubectl get pods -l app=clip-service
kubectl get pods -l app=gemma-service
kubectl get svc clip-service
kubectl get svc gemma-service
```

Ensure the pods reach a Running state. It may take some time on first start as the models will download from Hugging Face. You can check logs to see progress:

```
kubectl logs -l app=clip-service -f  # logs for CLIP service
kubectl logs -l app=gemma-service -f # logs for Gemma service
```

You should see output indicating Ray initialization, model loading, and the print statements confirming the service is running (from our app.py). For example, "CLIP service is running on Ray Serve (port 8000), awaiting requests..."【app.py log】.

4. Accessing the Services (HTTP Endpoints)

By default, we used ClusterIP services, which are accessible from within the cluster or via kubectl port-forward or an ingress. To quickly test the endpoints, you can use port-forward in a development environment:

CLIP Service:

```
kubectl port-forward service/clip-service 8080:80
```

This forwards your local port 8080 to the CLIP service. Now you can send an HTTP request to the CLIP endpoint. For example, let's classify an image using CLIP:

```
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
         "labels": ["a photo of a cat", "a photo of a dog", "a photo of a car"]}' \
    http://localhost:8080/
```

In this example, we use an image URL from COCO dataset and ask the model whether it's a cat, dog, or car. CLIP will return a JSON response with scores. For example, you might get a response indicating one of the labels with the highest score:

```
[
  {"label": "a photo of a dog", "score": 0.98},
  {"label": "a photo of a cat", "score": 0.01},
  {"label": "a photo of a car", "score": 0.01}
]
```

This would mean the image was identified as a dog with high confidence (as expected in this test).

Gemma3 VLM Service:

In a separate terminal, port-forward the Gemma service (if not using an external LB or ingress):

`kubectl port-forward service/gemma-service 8081:80`

Now send a request to generate text from an image. For example, if you have an image URL and want a description:

```
curl -X POST -H "Content-Type: application/json" \
    -d '{"image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
         "prompt": "What animal is on the candy?"}' \
    http://localhost:8081/
```

This corresponds to the example from the Gemma 3 model card. The service should return a JSON with an "answer" field. For the above, the answer might be:
`{ "answer": "Based on the image, the animal on the candy is a turtle." }`

(This was the expected answer in the Gemma 3 example.) The model has analyzed the image and responded with a description.

5. Customization and Next Steps

Model Variants: You can swap out the models used. For CLIP, any Hugging Face zero-shot-image-classification pipeline model can be used. For the VLM, you can use larger Gemma 3 variants or other open multimodal models (ensure they are supported by Hugging Face pipelines or write custom inference code). Make sure to adjust resource requests for larger models (e.g., more memory or multiple GPUs if using model parallelism).
Scaling: This skeleton uses one replica for each Deployment. To scale, you could increase the K8s Deployment replicas (e.g., replicas: 2 or more). Since each replica is a separate Ray Serve instance, the Kubernetes Service will load-balance requests between them. (Alternatively, for more advanced use, you could run a single Ray cluster with multiple actors for the model, but that requires using Ray's KubeRay operator and is beyond this initial setup.)
Node Scheduling: We demonstrated nodeSelector for GPU scheduling. You could also use tolerations and taints if your GPU nodes are tainted (e.g., nvidia.com/gpu taint). Add corresponding tolerations in the pod spec if needed. You may also use topology spread constraints or affinities for more control. By default, the provided manifests stick to the essential scheduling for GPU.
Annotations: If deploying on GKE, you might not need any special annotations for GPU usage beyond scheduling to the right node and requesting the resource. On some on-prem setups, you may want to add an annotation to use NVIDIA runtime (depending on your Kubernetes configuration, e.g., nvidia.com/gpu.deploy.batch=...). However, typically if the device plugin is installed, just requesting the GPU resource is enough and the NVIDIA Container Runtime is configured cluster-wide. We've kept an Istio annotation as an example; you can add or remove annotations for your environment as needed.
Logging/Monitoring: Ray Serve logs requests and you can integrate with Prometheus or other monitoring by exporting metrics from Ray or adding sidecar agents. For brevity, this is not included here.
Security: The services as configured have no authentication and expect to be used in a secure network or behind an API gateway. In production, consider adding auth (e.g., using an ingress with auth, or implementing auth in the app). Also, container images should be scanned for vulnerabilities, and you may want to run them with a non-root user, etc. (In this skeleton, we did not set a specific user in Dockerfile, but you can add one for security best practices.)

## Conclusion

This repository structure provides an idiomatic starting point for deploying ML model serving microservices with Ray Serve on Kubernetes. By using Ray Serve, we easily expose models as HTTP endpoints and can leverage Ray's scalability in the future. The included Dockerfiles and manifests are meant to be adjusted to your specific models and cluster setup. With CLIP, we can do zero-shot image classification (matching images with text labels), and with Gemma 3 (or similar VLMs) we can handle multimodal queries (image + text to text answers). By following the README steps, you can build the containers, deploy to a GPU-enabled Kubernetes cluster, and test the HTTP APIs. From here, you can expand the setup, integrate into CI/CD, add monitoring, or switch to using the Ray operator for a multi-node Ray cluster if needed. This provides a production-oriented baseline for serving advanced vision-language models in a scalable manner.
