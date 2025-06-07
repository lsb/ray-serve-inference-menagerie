import os
import time
import signal
import asyncio

import io
from typing import Dict, Any
import ray
from ray import serve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="Moondream Service", description="Vision-language model using Moondream2")

@serve.deployment(
    ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0}
)
@serve.ingress(app)
class MoondreamService:
    def __init__(self):
        self.device_info = self._get_device_info()
        self.model = self._initialize_model()
        print(f"Moondream Service initialized with device: {self.device_info}")
    
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
    
    def _initialize_model(self):
        device_map = {"": "cuda"} if torch.cuda.is_available() else None
        return AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-04-14",
            trust_remote_code=True,
            device_map=device_map
        )
    
    async def _process_image(self, image: UploadFile) -> Image.Image:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        return pil_image
    
    @app.get("/health")
    async def health_check(self):
        return {
            "status": "healthy",
            "service": "moondream",
            "device_info": self.device_info
        }
    
    @app.post("/caption")
    async def caption_image(self, 
                          image: UploadFile = File(...), 
                          length: str = Form("normal")) -> JSONResponse:
        start_time = time.time()
        
        try:
            pil_image = await self._process_image(image)
            
            inference_start = time.time()
            result = self.model.caption(pil_image, length=length)
            inference_time = time.time() - inference_start
            
            caption = result["caption"]
            
            total_time = time.time() - start_time
            
            response = {
                "caption": caption,
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
                    "error": f"Captioning failed: {str(e)}",
                    "performance": {
                        "error_time_ms": round(error_time * 1000, 2),
                        "device_info": self.device_info
                    }
                }
            )
    
    @app.post("/query")
    async def query_image(self, 
                        image: UploadFile = File(...), 
                        question: str = Form(...)) -> JSONResponse:
        start_time = time.time()
        
        try:
            pil_image = await self._process_image(image)
            
            if not question:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Provide 'question' in form data.",
                        "performance": {
                            "error_time_ms": round((time.time() - start_time) * 1000, 2),
                            "device_info": self.device_info
                        }
                    }
                )
            
            inference_start = time.time()
            result = self.model.query(pil_image, question)
            inference_time = time.time() - inference_start
            
            answer = result["answer"]
            
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
                    "error": f"Query failed: {str(e)}",
                    "performance": {
                        "error_time_ms": round(error_time * 1000, 2),
                        "device_info": self.device_info
                    }
                }
            )
    
    @app.post("/detect")
    async def detect_objects(self, 
                           image: UploadFile = File(...), 
                           object: str = Form(...)) -> JSONResponse:
        start_time = time.time()
        
        try:
            pil_image = await self._process_image(image)
            object_name = object
            
            if not object_name:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Provide 'object' name in form data.",
                        "performance": {
                            "error_time_ms": round((time.time() - start_time) * 1000, 2),
                            "device_info": self.device_info
                        }
                    }
                )
            
            inference_start = time.time()
            result = self.model.detect(pil_image, object_name)
            inference_time = time.time() - inference_start
            
            objects = result["objects"]
            
            total_time = time.time() - start_time
            
            response = {
                "objects": objects,
                "count": len(objects),
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
                    "error": f"Detection failed: {str(e)}",
                    "performance": {
                        "error_time_ms": round(error_time * 1000, 2),
                        "device_info": self.device_info
                    }
                }
            )
    
    @app.post("/point")
    async def point_objects(self, 
                          image: UploadFile = File(...), 
                          object: str = Form(...)) -> JSONResponse:
        start_time = time.time()
        
        try:
            pil_image = await self._process_image(image)
            object_name = object
            
            if not object_name:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Provide 'object' name in form data.",
                        "performance": {
                            "error_time_ms": round((time.time() - start_time) * 1000, 2),
                            "device_info": self.device_info
                        }
                    }
                )
            
            inference_start = time.time()
            result = self.model.point(pil_image, object_name)
            inference_time = time.time() - inference_start
            
            points = result["points"]
            
            total_time = time.time() - start_time
            
            response = {
                "points": points,
                "count": len(points),
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
                    "error": f"Pointing failed: {str(e)}",
                    "performance": {
                        "error_time_ms": round(error_time * 1000, 2),
                        "device_info": self.device_info
                    }
                }
            )

def signal_handler(sig, frame):
    serve.shutdown()
    ray.shutdown()
    exit(0)

moondream_service = MoondreamService.bind()

if __name__ == "__main__":
    import signal
    import time
    
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))})
    
    serve.run(moondream_service)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Moondream service is running on port {os.environ.get('PORT', 8000)}")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        signal_handler(None, None)
