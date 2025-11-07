import sys
import json
import requests

if len(sys.argv) < 2:
    print("Usage: python client_example.py <path_to_image>")
    sys.exit(1)

img_path = sys.argv[1]
url = "http://127.0.0.1:8000/predict?top_k=3"

with open(img_path, "rb") as f:
    files = {"file": (img_path, f, "application/octet-stream")}
    r = requests.post(url, files=files, timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))
