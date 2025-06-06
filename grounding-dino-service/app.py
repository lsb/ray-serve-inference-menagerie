from ray import serve
from fastapi import FastAPI, File, UploadFile, Form
import torch, logging, io, time
from typing import Dict, Any
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI(title="Grounding DINO Service", description="Object detection using Grounding DINO and SAM2")
logger = logging.getLogger("ray.serve")

@serve.deployment(ray_actor_options={"num_gpus": 1})  # Allocate one GPU to this actor
@serve.ingress(app)
class GroundingDinoService:
    def __init__(self):
        self.device_info = self._get_device_info()
        model_id = "IDEA-Research/grounding-dino-base"  # or "grounding-dino-1.5" if available
        logger.info(f"Downloading GroundingDINO model '{model_id}'...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")
        self.dino_model.eval()
        sam_checkpoint = "facebook/sam2.1-hiera-large"
        logger.info(f"Downloading SAM2 model '{sam_checkpoint}'...")
        self.sam_predictor = SAM2ImagePredictor.from_pretrained(sam_checkpoint)
        self.sam_predictor.model.to("cuda")
        logger.info("Models loaded successfully.")
        print(f"Grounding DINO Service initialized with device: {self.device_info}")

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

    @app.get("/health")
    async def health_check(self):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "grounding-dino",
            "device_info": self.device_info
        }

    @app.post("/text_detect")
    async def text_detect(self, 
                           image: UploadFile = File(...), 
                           text: str = Form(...)) -> dict:
        """Handle text-prompted detection: returns bounding boxes for objects matching the text."""
        start_time = time.time()
        
        try:
            image_data = await image.read()
            from PIL import Image
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            inference_start = time.time()
            inputs = self.processor(images=pil_img, text=[[text]], return_tensors="pt")
            inputs = inputs.to("cuda")
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=0.3, text_threshold=0.25, 
                target_sizes=[pil_img.size[::-1]]  # (height, width)
            )
            inference_time = time.time() - inference_start
            
            det = results[0]  # first (single) image
            boxes = det["boxes"].tolist()  # list of [x1, y1, x2, y2] in pixels
            scores = det["scores"].tolist()
            logger.info(f"text_detect: Found {len(boxes)} boxes for query '{text}'.")
            
            total_time = time.time() - start_time
            return {
                "boxes": boxes, 
                "scores": scores, 
                "text": text,
                "performance": {
                    "total_time_ms": round(total_time * 1000, 2),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "device_info": self.device_info
                }
            }
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"text_detect failed: {str(e)}")
            return {
                "error": f"Detection failed: {str(e)}",
                "performance": {
                    "error_time_ms": round(error_time * 1000, 2),
                    "device_info": self.device_info
                }
            }

    @app.post("/detect")
    async def detect(self, image: UploadFile = File(...)) -> dict:
        """Handle generic detection: returns bounding boxes for all detected regions."""
        start_time = time.time()
        
        try:
            image_data = await image.read()
            from PIL import Image
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            inference_start = time.time()
            self.sam_predictor.set_image(pil_img)
            masks, iou_scores, _ = self.sam_predictor.predict(prompts=None)  
            boxes = []
            for mask in masks:
                coords = torch.nonzero(torch.from_numpy(mask))  # indices where mask=1
                y_coords = coords[:,0]; x_coords = coords[:,1]
                y1, y2 = int(torch.min(y_coords)), int(torch.max(y_coords))
                x1, x2 = int(torch.min(x_coords)), int(torch.max(x_coords))
                boxes.append([x1, y1, x2, y2])
            inference_time = time.time() - inference_start
            
            logger.info(f"detect: Found {len(boxes)} objects (no prompt).")
            
            total_time = time.time() - start_time
            return {
                "boxes": boxes,
                "performance": {
                    "total_time_ms": round(total_time * 1000, 2),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "device_info": self.device_info
                }
            }
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"detect failed: {str(e)}")
            return {
                "error": f"Detection failed: {str(e)}",
                "performance": {
                    "error_time_ms": round(error_time * 1000, 2),
                    "device_info": self.device_info
                }
            }

grounding_dino_service = GroundingDinoService.bind()

if __name__ == "__main__":
    import signal
    import os
    import ray
    
    def signal_handler(sig, frame):
        serve.shutdown()
        ray.shutdown()
        exit(0)
    
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))})
    serve.run(grounding_dino_service)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Grounding DINO service is running on port {os.environ.get('PORT', 8000)}")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        signal_handler(None, None)
