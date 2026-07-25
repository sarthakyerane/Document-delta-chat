import os
import urllib.request
import json
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        for model in data.get("models", []):
            print(model.get("name"))
except Exception as e:
    print("Error:", e)
