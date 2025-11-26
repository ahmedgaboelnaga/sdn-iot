import requests
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python sentiment_client.py <server_ip> <text_to_analyze>")
        server_ip = "127.0.0.1"
        text = "This project is going really well and I am happy!"
        print(f"[*] No args provided. Using default: {server_ip} '{text}'")
    else:
        server_ip = sys.argv[1]
        # Join all remaining arguments as the text
        text = " ".join(sys.argv[2:])

    url = f"http://{server_ip}:8002/analyze"
    
    try:
        print(f"[*] Sending text to {url}...")
        response = requests.post(url, json={"text": text})
        
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
