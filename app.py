from flask import Flask, request, jsonify
import numpy as np
import cv2
import pytesseract

app = Flask(__name__)

def read_weight(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=0123456789."
    )

    weight = ''.join(c for c in text if c.isdigit() or c=='.')

    return weight


@app.route("/")
def home():
    return "ESP32 OCR Server Running"


@app.route("/detect", methods=["POST"])
def detect():

    img_bytes = request.data

    if len(img_bytes) == 0:
        return jsonify({"weight":"0"})

    npimg = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    weight = read_weight(img)

    print("Detected weight:", weight)

    return jsonify({"weight":weight})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)