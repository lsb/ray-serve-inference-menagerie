import os, time, signal, asyncio
import ray
from ray import serve
import torch
from transformers import pipeline

ray.init()
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))})

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
        data = await request.json()
        image_url = data.get("image_url")
        user_prompt = data.get("prompt", "") or ""
        if not image_url:
            return {"error": "Provide 'image_url' (and optionally 'prompt') in request JSON."}
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        output = self.pipeline(text=messages, max_new_tokens=200)
        try:
            generated_messages = output[0]["generated_text"]
            answer = generated_messages[-1]["content"] if isinstance(generated_messages, list) else generated_messages
        except Exception as e:
            answer = None
        return {"answer": answer}

GemmaService.deploy()

signal.signal(signal.SIGTERM, lambda sig, frame: exit(0))
print("Gemma3 VLM service is running on Ray Serve (port {}), awaiting requests...".format(os.environ.get("PORT", 8000)))
while True:
    time.sleep(60)
