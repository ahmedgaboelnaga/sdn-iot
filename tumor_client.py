import requests
import sys
import os
import cv2
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python tumor_client.py <server_ip> <image_path>")
        # Default to localhost for testing if no args
        server_ip = "127.0.0.1"
        image_path = "test_mri.jpg"
        print(f"[*] No args provided. Using default: {server_ip} {image_path}")
    else:
        server_ip = sys.argv[1]
        image_path = sys.argv[2]

    url = f"http://{server_ip}:8001/detect"
    
    # Create a dummy file if it doesn't exist for testing
    if not os.path.exists(image_path):
        with open(image_path, "wb") as f:
            f.write(b"fake_image_data_for_testing")
        print(f"[*] Created dummy image file: {image_path}")

    try:
        print(f"[*] Sending {image_path} to {url}...")
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("[*] Response received:")
            result = response.json()
            print(result)
            
            # Draw boxes on the image
            img = cv2.imread(image_path)
            if img is None:
                print("[!] Could not read original image to draw boxes.")
                return

            boxes = result.get("boxes", [])
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label
                text = f"{label} {score:.2f}"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            output_filename = "tumor_detection_result.jpg"
            cv2.imwrite(output_filename, img)
            print(f"[*] Saved annotated image to {output_filename}")
            
        else:
            print(f"[!] Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"[!] Connection failed: {e}")

if __name__ == "__main__":
    main()
