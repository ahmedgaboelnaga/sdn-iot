from fastapi import FastAPI, File, UploadFile
import uvicorn
import io
from PIL import Image
import sys

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
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Run inference
            results = self.model(image)
            
            # Process results (simplified for demo)
            # Assuming class 0 is 'tumor'
            has_tumor = False
            confidence = 0.0
            
            if results and len(results) > 0:
                # Check boxes
                for box in results[0].boxes:
                    # You might need to check box.cls if you have multiple classes
                    # For now, we assume any detection is a tumor
                    has_tumor = True
                    confidence = float(box.conf[0])
                    break
            
            return {"has_tumor": has_tumor, "confidence": round(confidence, 2)}
        else:
            # Fallback Mock logic if model failed to load
            import random
            print("[!] Using MOCK prediction (Model not active)")
            return {
                "has_tumor": random.choice([True, False]), 
                "confidence": 0.85, 
                "note": "MOCK RESULT - Model not loaded"
            }

# Initialize model
# IMPORTANT: Place your 'best.pt' in the same directory as this script
detector = TumorDetector(model_path="best.pt")

@app.post("/detect")
async def detect_tumor(file: UploadFile = File(...)):
    print(f"[*] Received file: {file.filename}")
    content = await file.read()
    
    result = detector.predict(content)
    
    return {
        "filename": file.filename,
        "prediction": result,
        "message": "Tumor detected" if result["has_tumor"] else "No tumor detected"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
