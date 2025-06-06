import os, time, signal, asyncio
import ray
from ray import serve
import torch
from transformers import pipeline

ray.init()  # Starts a local Ray runtime (or connects to one if specified by environment)
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))})

clip_pipeline = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    device=0 if torch.cuda.is_available() else -1,  # use GPU if available
)

@serve.deployment(route_prefix="/", ray_actor_options={"num_gpus": 1})
class CLIPService:
    def __init__(self):
        self.pipeline = clip_pipeline
    async def __call__(self, request):
        data = await request.json()
        image_url = data.get("image_url")
        candidate_labels = data.get("labels")
        if not image_url or not candidate_labels:
            return {"error": "Provide 'image_url' and 'labels' in request JSON."}
        result = self.pipeline(image_url, candidate_labels=candidate_labels)
        return result  # e.g., list of {"label": ..., "score": ...} dicts

CLIPService.deploy()

signal.signal(signal.SIGTERM, lambda sig, frame: exit(0))
print("CLIP service is running on Ray Serve (port {}), awaiting requests...".format(os.environ.get("PORT", 8000)))
while True:
    time.sleep(60)
