The goal is to deploy a variety of ML models for vision and language inference using Ray Serve on Kubernetes.

This is part of a compilation of sketches of the architecture, before starting to build.

---

## Ray Serve Deployment: Grounding DINO with SAM2 Integration

### Overview and Objectives

We need to deploy a Grounding DINO model integrated with SAM 2 (Segment Anything Model v2) on a GPU within an existing Ray Serve cluster. This deployment will expose two HTTP inference endpoints: one for text-prompted region proposals (accepts an image and text query, returns bounding boxes of objects matching the text) and another for generic object detection (accepts an image alone, returns bounding box proposals for all detected regions). The solution must download model weights at runtime from open repositories, leverage GPU scheduling (with Kubernetes nodeSelector for GPU nodes), and include robust logging and metrics for production monitoring. We will integrate this new service into the same Ray Serve cluster as other models (e.g. CLIP, Gemma) to allow seamless scaling and management. Below, we present the Ray Serve application code, Dockerfile, and Kubernetes manifest needed to achieve this, along with explanations of design decisions and best practices.

### Model Integration: Grounding DINO + SAM 2

Grounding DINO is an open-set object detector that uses text prompts to find arbitrary objects in an image. Given an input phrase (e.g. "a red car"), Grounding DINO will output bounding boxes for regions in the image that likely correspond to that description. It marries a transformer detector (DINO) with language embeddings, enabling zero-shot detection of objects by name. SAM 2 (Segment Anything Model 2) is a next-generation segmentation model from Meta that can segment any object in images or videos, even without predefined categories. SAM2 is promptable (via points or boxes) but can also “segment everything” in an image when no prompt is given. By combining Grounding DINO with SAM2, we get a powerful pipeline that can detect and segment any object based on text input. In this deployment, we use Grounding DINO to generate bounding boxes (region proposals) and leverage SAM2 for broad image segmentation when no text is provided (to find all objects).

Both models will be loaded at runtime from public repositories. We use Hugging Face Transformers to download Grounding DINO, and the official SAM2 library (via Hugging Face Hub) for the SAM2 model. This means the container does not ship with large model weights baked in – instead, it pulls them on startup, ensuring we get the latest versions and abide by licensing. For Grounding DINO, we use the pre-trained checkpoint from the IDEA-Research repository (e.g. grounding-dino-base or grounding-dino-1.5). For SAM2, we use Meta’s published checkpoint (e.g. facebook/sam2.1-hiera-large on Hugging Face, which corresponds to SAM 2.1 Large).
