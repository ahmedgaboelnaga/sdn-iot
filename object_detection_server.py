from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
import uvicorn
import io

app = FastAPI()

class DetectionResult(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    scores: List[float]

# Load the YOLOv8 model
# This will automatically download yolov8n.pt if not present
print("[*] Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
print("[*] Model loaded.")

@app.post("/detect", response_model=DetectionResult)
async def detect_objects(file: UploadFile = File(...)):
    print(f"[*] Received file: {file.filename}")
    
    # Read and decode the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return DetectionResult(boxes=[], labels=[], scores=[])

    # Run YOLOv8 inference
    results = model.predict(img)

    # Extract predictions
    predictions = results[0]
    
    # Get boxes, scores, and class IDs
    boxes = predictions.boxes.xyxy.cpu().numpy().tolist()
    scores = predictions.boxes.conf.cpu().numpy().tolist()
    class_ids = predictions.boxes.cls.cpu().numpy()
    
    # Map class IDs to names
    labels = [model.names[int(cls)] for cls in class_ids]

    print(f"[*] Detected {len(boxes)} objects.")

    # Return detection results
    return DetectionResult(boxes=boxes, labels=labels, scores=scores)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
