from ray import serve
from fastapi import FastAPI, File, UploadFile, Form
import torch, logging, io
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI()
logger = logging.getLogger("ray.serve")

@serve.deployment(ray_actor_options={"num_gpus": 1})  # Allocate one GPU to this actor
@serve.ingress(app)
class GroundingDinoService:
    def __init__(self):
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

    @app.post("/text_detect")
    async def text_detect(self, 
                           image: UploadFile = File(...), 
                           text: str = Form(...)) -> dict:
        """Handle text-prompted detection: returns bounding boxes for objects matching the text."""
        image_data = await image.read()
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        inputs = self.processor(images=pil_img, text=[[text]], return_tensors="pt")
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.3, text_threshold=0.25, 
            target_sizes=[pil_img.size[::-1]]  # (height, width)
        )
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
        self.sam_predictor.set_image(pil_img)
        masks, iou_scores, _ = self.sam_predictor.predict(prompts=None)  
        boxes = []
        for mask in masks:
            coords = torch.nonzero(torch.from_numpy(mask))  # indices where mask=1
            y_coords = coords[:,0]; x_coords = coords[:,1]
            y1, y2 = int(torch.min(y_coords)), int(torch.max(y_coords))
            x1, x2 = int(torch.min(x_coords)), int(torch.max(x_coords))
            boxes.append([x1, y1, x2, y2])
        logger.info(f"detect: Found {len(boxes)} objects (no prompt).")
        return {"boxes": boxes}
