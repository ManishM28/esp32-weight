from flask import Flask, request, jsonify
import requests
import base64
import re

app = Flask(__name__)

API_KEY = "AIzaSyCmlIsG6Y-BL35u-K-fEzdZq_LZJxN7rwY"

@app.route("/")
def home():
    return "ESP32 OCR Server Running"

@app.route("/detect", methods=["POST"])
def detect():

    image = request.data
    img_base64 = base64.b64encode(image).decode()

    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

    payload = {
        "requests": [
            {
                "image": {"content": img_base64},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }

    r = requests.post(url, json=payload)
    result = r.json()

    text = ""

    try:
        text = result["responses"][0]["fullTextAnnotation"]["text"]
    except:
        pass

    weight = 0
    match = re.search(r"\d+\.\d+|\d+", text)

    if match:
        weight = match.group()

    return jsonify({"weight": weight})