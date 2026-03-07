from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def home():
    return "ESP32 weight server running"

@app.route("/detect", methods=["POST"])
def detect():

    img_bytes = request.data

    if not img_bytes:
        return jsonify({"weight":"0"})

    npimg = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"weight":"0"})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)[1]

    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    digits = []

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        if h > 40:
            digits.append("1")  # simple test detection

    weight = "".join(digits)

    return jsonify({"weight":weight})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)