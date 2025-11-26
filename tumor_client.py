import requests
import sys
import os

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
            print(response.json())
        else:
            print(f"[!] Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"[!] Connection failed: {e}")

if __name__ == "__main__":
    main()
