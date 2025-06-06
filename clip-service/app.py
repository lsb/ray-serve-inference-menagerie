import os
import time
import signal
import asyncio
from typing import Dict, Any, List
import ray
from ray import serve
import torch
from transformers import pipeline
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="CLIP Service", description="Zero-shot image classification using CLIP")

@serve.deployment(
    ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0}
)
@serve.ingress(app)
class CLIPService:
    def __init__(self):
        self.device_info = self._get_device_info()
        self.pipeline = self._initialize_pipeline()
        print(f"CLIP Service initialized with device: {self.device_info}")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information for performance debugging."""
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
        """Initialize CLIP pipeline with appropriate device."""
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=device
        )
    
    @app.get("/health")
    async def health_check(self):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "clip",
            "device_info": self.device_info
        }
    
    @app.post("/")
    async def classify_image(self, request_data: Dict[str, Any]) -> JSONResponse:
        """Main classification endpoint."""
        start_time = time.time()
        
        try:
            image_url = request_data.get("image_url")
            candidate_labels = request_data.get("labels")
            
            if not image_url or not candidate_labels:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide 'image_url' and 'labels' in request JSON."}
                )
            
            if not isinstance(candidate_labels, list) or len(candidate_labels) == 0:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Labels must be a non-empty list."}
                )
            
            processed_image_url = image_url
            if image_url.startswith("file://"):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Local file URLs (file://) are not supported in containerized deployment. Please use HTTP/HTTPS URLs or base64 encoded images.",
                        "performance": {
                            "error_time_ms": round((time.time() - start_time) * 1000, 2),
                            "device_info": self.device_info
                        }
                    }
                )
            
            inference_start = time.time()
            result = self.pipeline(processed_image_url, candidate_labels=candidate_labels)
            inference_time = time.time() - inference_start
            
            total_time = time.time() - start_time
            
            response = {
                "predictions": result,
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
                    "error": f"Classification failed: {str(e)}",
                    "performance": {
                        "error_time_ms": round(error_time * 1000, 2),
                        "device_info": self.device_info
                    }
                }
            )

if __name__ == "__main__":
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))})
    serve.run(CLIPService.bind())
    
    import signal
    import time
    
    def signal_handler(sig, frame):
        serve.shutdown()
        ray.shutdown()
        exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"CLIP service is running on port {os.environ.get('PORT', 8000)}")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        signal_handler(None, None)
