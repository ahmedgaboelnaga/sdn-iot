from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import uvicorn
import io
import sys
import cv2
import numpy as np

# Try to import YOLO, handle if not installed
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[!] 'ultralytics' library not found. Please install it: pip install ultralytics")

app = FastAPI()

class DetectionResult(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    scores: List[float]
    message: str

class TumorDetector:
    def __init__(self, model_path="best.pt"):
        self.model = None
        if HAS_YOLO:
            try:
                print(f"[*] Loading YOLO model from {model_path}...")
                self.model = YOLO(model_path)
                print("[*] Model loaded successfully.")
            except Exception as e:
                print(f"[!] Failed to load model: {e}")
                print(f"[!] Please ensure '{model_path}' is in the current directory.")
        else:
            print("[!] Running in MOCK mode (missing library).")

    def predict(self, image_bytes) -> DetectionResult:
        if self.model:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image (BGR) - OpenCV format
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_np is None:
                print("[!] Failed to decode image")
                return DetectionResult(boxes=[], labels=[], scores=[], message="Failed to decode image")

            # Run inference
            results = self.model.predict(source=img_np, conf=0.01)
            
            result = results[0]
            det_count = len(result.boxes)
            print(f"[*] Detections found: {det_count}")
            
            # Extract data
            boxes = result.boxes.xyxy.tolist()
            scores = result.boxes.conf.tolist()
            # Map class IDs to names
            labels = [result.names[int(cls)] for cls in result.boxes.cls.tolist()]
            
            message = "Tumor detected" if det_count > 0 else "No tumor detected"
            
            return DetectionResult(boxes=boxes, labels=labels, scores=scores, message=message)
        else:
            # Fallback Mock logic
            print("[!] Using MOCK prediction (Model not active)")
            return DetectionResult(
                boxes=[[50.0, 50.0, 200.0, 200.0]], 
                labels=["tumor_mock"], 
                scores=[0.95], 
                message="MOCK RESULT"
            )

# Initialize model
detector = TumorDetector(model_path="best.pt")

@app.post("/detect", response_model=DetectionResult)
async def detect_tumor(file: UploadFile = File(...)):
    print(f"[*] Received file: {file.filename}")
    content = await file.read()
    
    result = detector.predict(content)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
