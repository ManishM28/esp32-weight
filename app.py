from flask import Flask, request, jsonify
import numpy as np
import cv2
import pytesseract

app = Flask(__name__)

def read_weight(image):

    # convert to grayscale (works for any display color)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # increase contrast
    gray = cv2.equalizeHist(gray)

    # blur to remove noise
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # adaptive threshold works for different lighting/colors
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # OCR tuned for numbers only
    text = pytesseract.image_to_string(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=0123456789."
    )

    # extract only numbers and decimal
    weight = ''.join(c for c in text if c.isdigit() or c == '.')

    return weight


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"error":"no image received"})

    file = request.files["image"]

    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    weight = read_weight(img)

    print("Detected Weight:", weight)

    return jsonify({"weight": weight})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)