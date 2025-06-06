The goal is to deploy a variety of ML models for vision and language inference using Ray Serve on Kubernetes.

This is part of a compilation of sketches of the architecture, before starting to build.

---

## Ray Serve Deployment: Grounding DINO with SAM2 Integration

### Overview and Objectives

We need to deploy a Grounding DINO model integrated with SAM 2 (Segment Anything Model v2) on a GPU within an existing Ray Serve cluster. This deployment will expose two HTTP inference endpoints: one for text-prompted region proposals (accepts an image and text query, returns bounding boxes of objects matching the text) and another for generic object detection (accepts an image alone, returns bounding box proposals for all detected regions). The solution must download model weights at runtime from open repositories, leverage GPU scheduling (with Kubernetes nodeSelector for GPU nodes), and include robust logging and metrics for production monitoring. We will integrate this new service into the same Ray Serve cluster as other models (e.g. CLIP, Gemma) to allow seamless scaling and management. Below, we present the Ray Serve application code, Dockerfile, and Kubernetes manifest needed to achieve this, along with explanations of design decisions and best practices.

### Model Integration: Grounding DINO + SAM 2

Grounding DINO is an open-set object detector that uses text prompts to find arbitrary objects in an image. Given an input phrase (e.g. "a red car"), Grounding DINO will output bounding boxes for regions in the image that likely correspond to that description. It marries a transformer detector (DINO) with language embeddings, enabling zero-shot detection of objects by name. SAM 2 (Segment Anything Model 2) is a next-generation segmentation model from Meta that can segment any object in images or videos, even without predefined categories. SAM2 is promptable (via points or boxes) but can also “segment everything” in an image when no prompt is given. By combining Grounding DINO with SAM2, we get a powerful pipeline that can detect and segment any object based on text input. In this deployment, we use Grounding DINO to generate bounding boxes (region proposals) and leverage SAM2 for broad image segmentation when no text is provided (to find all objects).

Both models will be loaded at runtime from public repositories. We use Hugging Face Transformers to download Grounding DINO, and the official SAM2 library (via Hugging Face Hub) for the SAM2 model. This means the container does not ship with large model weights baked in – instead, it pulls them on startup, ensuring we get the latest versions and abide by licensing. For Grounding DINO, we use the pre-trained checkpoint from the IDEA-Research repository (e.g. grounding-dino-base or grounding-dino-1.5). For SAM2, we use Meta’s published checkpoint (e.g. facebook/sam2.1-hiera-large on Hugging Face, which corresponds to SAM 2.1 Large).

### Ray Serve Deployment Implementation (app.py)

We create a Ray Serve deployment that loads both models into GPU memory and defines two HTTP endpoints. We use FastAPI integration for convenience of routing and payload parsing. The deployment is defined as a Python class with the @serve.deployment decorator (to register it with Ray Serve) and @serve.ingress(app) to attach FastAPI routes. We specify ray_actor_options={"num_gpus": 1} on the deployment to ensure Ray allocates a GPU and schedules the replica on a GPU node. This guarantees the heavy model computations run on the GPU. Model downloading and initialization happens in __init__, so it occurs once per replica (at startup). Grounding DINO is loaded via the Hugging Face Transformers API (AutoProcessor and AutoModel), and moved to CUDA. SAM2 is loaded via its predictor class from the official library, using the Hugging Face Hub checkpoint (this requires the sam2 Python package from Meta).

We define two endpoints using FastAPI decorators: /text_detect for text-prompted detection and /detect for pure image detection. The FastAPI UploadFile is used to receive images in requests, and text is a query or form field. The logic is:
* /text_detect: Accepts an image file and a text string. We run the Grounding DINO model on the image with the given text prompt to obtain bounding boxes for objects matching the description. This involves preprocessing the image and text (using the AutoProcessor), running the model forward pass, and then applying the model’s post-processing to get boxes and confidence scores. We can use processor.post_process_grounded_object_detection(...) to get boxes filtered by the text prompt and a confidence threshold. The result returned is a list of bounding box coordinates (e.g. in pixel values relative to the image) along with the prompt or score. (For simplicity, our example code returns just the coordinates and scores).
* /detect: Accepts only an image. Because Grounding DINO requires a text prompt to detect specific objects, for the “no text” case we instead leverage SAM2 to find all prominent objects. We call the SAM2 predictor with no prompts, which triggers its “segment everything” mode to generate masks for all objects in the image. From the set of masks, we derive bounding boxes by computing the tight rectangle around each mask. These boxes represent generic object proposals in the image. (Alternatively, one could call Grounding DINO with a very generic prompt like "." or a list of common object names, but using SAM2’s automatic segmentation ensures we capture any arbitrary object). The endpoint returns the list of bounding box coordinates for all detected regions.

Each inference request is logged for monitoring. We use Python’s logging within the deployment (via the Ray Serve logger "ray.serve") to print details like request type and number of boxes found. Ray will output these logs to the container’s stderr (and to a log file on disk) by default, so they can be captured by kubectl logs or a logging aggregator. Ray Serve also tracks request counts and latencies for this deployment automatically, which we can scrape via Prometheus/Grafana as with the other models in the cluster.

Below is the app.py implementation of the deployment:

```
from ray import serve
from fastapi import FastAPI, File, UploadFile, Form
import torch, logging
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# Import SAM2 predictor (assuming `sam2` package is installed)
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI()
logger = logging.getLogger("ray.serve")

@serve.deployment(ray_actor_options={"num_gpus": 1})  # Allocate one GPU to this actor
@serve.ingress(app)
class GroundingDinoService:
    def __init__(self):
        # Load Grounding DINO model from Hugging Face (download at runtime)
        model_id = "IDEA-Research/grounding-dino-base"  # or "grounding-dino-1.5" if available
        logger.info(f"Downloading GroundingDINO model '{model_id}'...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")
        self.dino_model.eval()
        # Load SAM2 model (from Hugging Face Hub via SAM2 library)
        sam_checkpoint = "facebook/sam2.1-hiera-large"
        logger.info(f"Downloading SAM2 model '{sam_checkpoint}'...")
        self.sam_predictor = SAM2ImagePredictor.from_pretrained(sam_checkpoint)
        # Note: SAM2ImagePredictor will internally load model to GPU (uses torch.autocast).
        # Ensure image predictor uses GPU:
        self.sam_predictor.model.to("cuda")
        logger.info("Models loaded successfully.")

    @app.post("/text_detect")
    async def text_detect(self, 
                           image: UploadFile = File(...), 
                           text: str = Form(...)) -> dict:
        """Handle text-prompted detection: returns bounding boxes for objects matching the text."""
        # Read image bytes and open with PIL
        image_data = await image.read()
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        # Prepare inputs for Grounding DINO
        inputs = self.processor(images=pil_img, text=[[text]], return_tensors="pt")
        inputs = inputs.to("cuda")
        # Run inference
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        # Post-process to get boxes and scores (using text threshold and box threshold)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.3, text_threshold=0.25, 
            target_sizes=[pil_img.size[::-1]]  # (height, width)
        )
        # Format results
        det = results[0]  # first (single) image
        boxes = det["boxes"].tolist()  # list of [x1, y1, x2, y2] in pixels
        scores = det["scores"].tolist()
        logger.info(f"text_detect: Found {len(boxes)} boxes for query '{text}'.")
        return {"boxes": boxes, "scores": scores, "text": text}

    @app.post("/detect")
    async def detect(self, image: UploadFile = File(...)) -> dict:
        """Handle generic detection: returns bounding boxes for all detected regions."""
        image_data = await image.read()
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        # Use SAM2 to get masks for all objects (no prompts given)
        self.sam_predictor.set_image(pil_img)
        masks, iou_scores, _ = self.sam_predictor.predict(prompts=None)  
        # 'masks' is a list of binary mask arrays for each object
        boxes = []
        for mask in masks:
            # Compute tight bounding box of the mask
            coords = torch.nonzero(torch.from_numpy(mask))  # indices where mask=1
            y_coords = coords[:,0]; x_coords = coords[:,1]
            y1, y2 = int(torch.min(y_coords)), int(torch.max(y_coords))
            x1, x2 = int(torch.min(x_coords)), int(torch.max(x_coords))
            boxes.append([x1, y1, x2, y2])
        logger.info(f"detect: Found {len(boxes)} objects (no prompt).")
        return {"boxes": boxes}
```

(The code above assumes necessary imports and that the sam2 package is installed. It logs each request’s outcome and returns JSON responses containing bounding box coordinates. In a real implementation, you might add error handling (e.g., if image is not readable) and possibly limit the number of boxes or mask size for performance.)

A few Ray Serve best practices are observed in this implementation: we load the models once in the actor’s __init__ (so we don’t re-download on every request), and the model objects are stored as class attributes so they persist for all queries. We also set self.dino_model.eval() and wrap inference in torch.no_grad() to avoid unnecessary grad computations. The use of ray_actor_options={"num_gpus": 1} ensures the actor has exclusive access to one GPU. If needed, we could further tune num_cpus or use batch inference for throughput, but here we keep one request at a time per replica. Ray Serve will also allow scaling out to multiple replicas if needed (just increase num_replicas in the decorator or config).

### Dockerfile for the Service

We create a Docker image that contains all dependencies (Ray, models, and our app code). This image will be used for both the Ray head and worker nodes that run the GroundingDINO+SAM2 service. Below is an example Dockerfile:

```
# Start from Ray's official base image with Python 3.10 and GPU support
FROM rayproject/ray:2.6.0-py310-cu118  # (ensure this matches the Ray version used in cluster)
# Install model dependencies
RUN pip install --no-cache-dir transformers==4.31.0 huggingface_hub==0.16.4 \
                "torch>=2.5.1" torchvision --upgrade 
# Install Segment Anything Model 2 (SAM2) from source
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e . && \
    cd .. && rm -rf sam2
# (Optional: The above compiles SAM2. If any CUDA extension fails to build, SAM2 can still run:contentReference[oaicite:15]{index=15}.)
# Install any additional utilities
RUN pip install pillow fastapi uvicorn[standard]
# Copy the Ray Serve app code
COPY app.py /app/app.py
WORKDIR /app
```

Explanation: We base on a Ray image that already includes Ray Serve and common ML libraries. We then pip install Hugging Face Transformers and Huggingface Hub (to load Grounding DINO) and ensure PyTorch is at the required version (SAM2 needs torch >= 2.5.1 as noted by Meta). We install the SAM2 library by cloning the official GitHub and using `pip install -e .` . (This will also download the model config files. The actual model weights are not downloaded here – those will download at runtime when SAM2ImagePredictor.from_pretrained is called.) We also include Pillow for image processing and FastAPI/Uvicorn in case they aren’t already in the Ray image. Finally, we copy our app.py into the image. When building this image, ensure it’s pushed to a registry accessible by your K8s cluster (e.g., ECR, Docker Hub, etc.). The image size may be large due to PyTorch and models; you might use a slim base and install only needed components to minimize it.

### Kubernetes Deployment Manifest

To run this in Kubernetes, we leverage KubeRay and the Ray Serve operator. Since we want this service to run in the existing Ray cluster alongside other models, we don’t create a separate cluster. Instead, we can update the Ray cluster’s config to include a worker group for this deployment or use a RayService to deploy the new application. The key is to ensure a GPU node is available for the model and the Ray actor is scheduled there. Below is an example RayService manifest snippet that ensures a GPU-backed worker is present and deploys our GroundingDINO+SAM2 Serve application. This assumes the cluster uses a label (e.g., node_type: gpu) to identify GPU nodes. We use nodeSelector to pin the worker to a GPU node and request a GPU resource in the pod spec. The RayService’s serveDeploymentGraphSpec references our app’s import path to launch the Serve deployment on cluster start.

```
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
  name: grounding-dino-sam-service
spec:
  rayClusterConfig:
    rayVersion: "2.6.0"
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            image: myregistry/grounding-dino-sam:latest   # Image built from Dockerfile
            resources:
              requests:
                cpu: 1
                memory: 4Gi
    workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 1
      minReplicas: 1
      maxReplicas: 1
      rayStartParams:
        num-gpus: "1"    # Advertise 1 GPU to Ray on this node
      template:
        spec:
          nodeSelector:
            node_type: gpu               # schedule on GPU node (label must exist on GPU nodes)
          tolerations:
            - key: "nvidia.com/gpu"      # (example toleration if GPU nodes are tainted)
              operator: "Exists"
          containers:
          - name: ray-worker
            image: myregistry/grounding-dino-sam:latest   # Same image with models
            resources:
              limits:
                nvidia.com/gpu: 1        # request one GPU
                cpu: 4
                memory: 16Gi
              requests:
                cpu: 2
                memory: 16Gi
  serveDeploymentGraphSpec:
    importPath: "app:GroundingDinoService"   # Module and class for Serve app
    serveConfigSpecs:
      - name: GroundingDinoService           # Serve deployment name
        routePrefix: "/grounding-dino"       # Base URL path for this service
```

In this manifest, the Ray cluster is configured with one GPU worker pod. The nodeSelector and tolerations ensure this pod only runs on a node with a GPU (in our example, nodes with label node_type: gpu). We set the Ray start parameter num-gpus: "1" so that Ray is aware the worker has a GPU resource. The container resource limits include nvidia.com/gpu: 1 to actually allocate the GPU from Kubernetes. The head doesn’t need a GPU, so it’s scheduled normally (we could also scale other CPU-only workers for other models as needed).

The serveDeploymentGraphSpec tells the Ray Serve controller to deploy our application on this cluster. We use importPath: "app:GroundingDinoService" which means “in the container’s app.py, find the GroundingDinoService deployment”. The operator will invoke serve.run(GroundingDinoService.bind()) under the hood. We also specify a routePrefix of "/grounding-dino" for this service’s endpoints – combined with our FastAPI routes, the final paths will be /grounding-dino/text_detect and /grounding-dino/detect. This setup keeps it organized alongside other models (e.g., other deployments might use /clip or /gemma prefixes).

After applying this manifest (kubectl apply -f rayservice.yaml), KubeRay will ensure the Ray cluster is running and then deploy the Serve application. The GroundingDinoService actor will start on the GPU node, load the models (downloading weights on first startup), and then begin listening on the HTTP endpoints.

### Logging and Metrics in Production

For monitoring, our service leverages Ray Serve’s built-in logging and metrics. We log inference events using the logging module (with the "ray.serve" logger) so that logs are integrated with Ray’s logging system. By default, Ray writes Serve logs to both the pod’s local log file and stderr. In a Kubernetes environment, these will appear in the pod’s logs (accessible via kubectl logs) as well as in Ray’s dashboard. You can adjust logging levels or formats via Ray’s logging configuration if needed, but simply using logger.info() as shown will print to stdout/err which satisfies the requirement to log to console. Each request will produce a log like “text_detect: Found 2 boxes for query 'cat'.”, which can be monitored in real time.

Ray Serve also exposes metrics for each deployment that can be scraped by Prometheus (when integrated via KubeRay’s Prometheus support). Out of the box, you get counters such as ray_serve_deployment_request_counter_total (number of requests processed) and ray_serve_deployment_error_counter_total (number of exceptions), broken down by deployment name and route. Latency distributions (*_latency_ms) and replica health metrics are also provided. These metrics will include our GroundingDinoService deployment, just like the existing CLIP or Gemma deployments, so you can view how many requests it’s serving, its error rate, and latency in dashboards. If using KubeRay with Prometheus, ensure the Ray pods have the proper annotations or ServiceMonitor so that Prometheus picks up the metrics endpoint. We did not need to add any custom metrics in code, but one could use the Ray Serve metric API to log custom application-level metrics if desired.

### Example Usage

Once deployed, clients can call the endpoints to get bounding box predictions. Below are example requests to the service (assuming you port-forward or expose the Ray Serve HTTP address, and using the route prefix /grounding-dino as configured):

* Text-prompted detection example: Suppose we want to find all cats in an image. We send a POST request to /grounding-dino/text_detect with the image file and the text prompt. For instance, using curl: 
```
curl -X POST "http://<ray-head-service>/grounding-dino/text_detect" \
     -F "image=@/path/to/example.jpg" \
     -F "text=cat"
```
The response will be a JSON object containing coordinates of bounding boxes for cats found in the image. For example:
```
{ 
  "boxes": [[100, 50, 200, 180], [400, 45, 550, 170]],
  "scores": [0.85, 0.80],
  "text": "cat"
}
```

This indicates two boxes (with their top-left and bottom-right pixel coordinates) and confidence scores for the prompt "cat". In a real scenario, you might draw these boxes on the image or use the coordinates in downstream processing.

* Generic detection example: To detect all objects without specifying a class, we send a POST to /grounding-dino/detect with just the image:

```
curl -X POST "http://<ray-head-service>/grounding-dino/detect" \
     -F "image=@/path/to/example.jpg"
```
The service will respond with a JSON listing boxes for all distinct objects it found. For example:
```
{ 
  "boxes": [[30, 40, 85, 120], [100, 50, 200, 180], [400, 45, 550, 170]] 
}
```
Here three region proposals were returned. These correspond to whatever objects SAM2 identified in the scene (for instance, perhaps a chair, a cat, and a dog, though no labels are given in this generic mode). If needed, an upstream system could run an image classifier or tagger on those regions to assign labels, or the client can simply treat them as unidentified object locations.

These examples demonstrate that the deployment meets the requirements: it can handle requests with or without text prompts and returns bounding box predictions accordingly. The models are loaded dynamically from Hugging Face (visible in logs the first time as downloads) and run on the GPU. The Ray Serve cluster manages the scaling and routing, so this service can be invoked alongside other model endpoints.

### Conclusion

We have developed a Ray Serve deployment for Grounding DINO with SAM2 integration that is suitable for production. It supports both text-driven object detection and general proposal generation, running on GPU for accelerated inference. The configuration ensures the deployment is scheduled to a GPU node in Kubernetes and leverages Ray’s abstractions for scalability. Logging is directed to stdout (and Ray’s log files) for easy monitoring, and Ray’s metrics are available for integration into observability systems. By deploying this service into the existing Ray Serve cluster (next to models like CLIP and Gemma), we maintain a unified serving platform where each model is a modular endpoint. This design adheres to Ray Serve best practices for model loading, resource allocation, and application structure, providing a robust and extensible foundation for advanced computer vision inference in production.
