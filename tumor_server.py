from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
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

    def predict(self, image_bytes):
        if self.model:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image (BGR) - OpenCV format
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_np is None:
                print("[!] Failed to decode image")
                return image_bytes

            # Run inference
            # Using model.predict() explicitly as requested
            # source=img_np passes the loaded image directly
            results = self.model.predict(source=img_np, conf=0.01)
            
            det_count = len(results[0].boxes)
            print(f"[*] Detections found: {det_count}")
            
            if det_count == 0:
                print("[!] Warning: No tumors detected in this image.")

            # Get plotted image (numpy array BGR)
            # This generates the image with boxes, similar to how save=True works but in memory
            im_array = results[0].plot(conf=True, labels=True, boxes=True)
            
            # Encode back to jpg
            success, encoded_img = cv2.imencode('.jpg', im_array)
            
            if success:
                return encoded_img.tobytes()
            else:
                print("[!] Failed to encode result image")
                return image_bytes
        else:
            # Fallback Mock logic if model failed to load
            print("[!] Using MOCK prediction (Model not active)")
            return image_bytes

# Initialize model
# IMPORTANT: Place your 'best.pt' in the same directory as this script
detector = TumorDetector(model_path="best.pt")

@app.post("/detect")
async def detect_tumor(file: UploadFile = File(...)):
    print(f"[*] Received file: {file.filename}")
    content = await file.read()
    
    image_bytes = detector.predict(content)
    
    return Response(content=image_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
