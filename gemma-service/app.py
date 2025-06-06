import os
import time
import signal
import asyncio
from typing import Dict, Any
import ray
from ray import serve
import torch
from transformers import pipeline
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Gemma Service", description="Vision-language generation using Gemma3")

@serve.deployment(
    ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0}
)
@serve.ingress(app)
class GemmaService:
    def __init__(self):
        self.device_info = self._get_device_info()
        self.pipeline = self._initialize_pipeline()
        print(f"Gemma Service initialized with device: {self.device_info}")
    
    def _get_device_info(self) -> Dict[str, Any]:
        device_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": None,
            "device_type": "cpu"
        }
        
        if torch.cuda.is_available():
            device_info["device_name"] = torch.cuda.get_device_name(0)
            device_info["device_type"] = "gpu"
        
        return device_info
    
    def _initialize_pipeline(self):
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "image-text-to-text",
            model="google/gemma-3-1b-it",
            device=device,
            torch_dtype=torch.bfloat16
        )
    
    @app.get("/health")
    async def health_check(self):
        return {
            "status": "healthy",
            "service": "gemma",
            "device_info": self.device_info
        }
    
    @app.post("/")
    async def generate_text(self, request_data: Dict[str, Any]) -> JSONResponse:
        start_time = time.time()
        
        try:
            image_url = request_data.get("image_url")
            user_prompt = request_data.get("prompt", "") or ""
            
            if not image_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide 'image_url' (and optionally 'prompt') in request JSON."}
                )
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": user_prompt}
                ]}
            ]
            
            inference_start = time.time()
            output = self.pipeline(text=messages, max_new_tokens=200)
            inference_time = time.time() - inference_start
            
            try:
                generated_messages = output[0]["generated_text"]
                answer = generated_messages[-1]["content"] if isinstance(generated_messages, list) else generated_messages
            except Exception as e:
                answer = f"Error processing response: {str(e)}"
            
            total_time = time.time() - start_time
            
            response = {
                "answer": answer,
                "performance": {
                    "total_time_ms": round(total_time * 1000, 2),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "device_info": self.device_info
                }
            }
            
            return JSONResponse(content=response)
            
        except Exception as e:
            error_time = time.time() - start_time
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Generation failed: {str(e)}",
                    "performance": {
                        "error_time_ms": round(error_time * 1000, 2),
                        "device_info": self.device_info
                    }
                }
            )

if __name__ == "__main__":
    import signal
    import time
    
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))})
    serve.run(GemmaService.bind())
    
    def signal_handler(sig, frame):
        serve.shutdown()
        ray.shutdown()
        exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Gemma service is running on port {os.environ.get('PORT', 8000)}")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        signal_handler(None, None)
