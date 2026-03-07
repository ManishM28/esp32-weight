from flask import Flask, request, jsonify
import numpy as np
import cv2
import pytesseract

app = Flask(__name__)

@app.route("/")
def home():
    return "ESP32 OCR Server Running"


@app.route("/detect", methods=["POST"])
def detect():

    img_bytes = request.data

    if not img_bytes:
        return jsonify({"weight":"0"})

    npimg = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"weight":"0"})

    h, w, _ = img.shape

    # Crop display area
    img = img[int(h*0.35):int(h*0.55), int(w*0.25):int(w*0.75)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=0123456789."
    )

    text = text.strip()

    return jsonify({"weight":text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)