from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def home():
    return "ESP32 Weight Detection Server Running"


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

    # Crop only display area
    img = img[int(h*0.35):int(h*0.55), int(w*0.25):int(w*0.75)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digits = []

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        if h > 25 and w > 10:

            roi = thresh[y:y+h, x:x+w]

            white = cv2.countNonZero(roi)

            if white > 50:
                digits.append((x,"1"))

    digits = sorted(digits,key=lambda x:x[0])

    weight = ""

    for d in digits:
        weight += d[1]

    if weight == "":
        weight = "0"

    return jsonify({"weight":weight})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)