code from flask import Flask, request, jsonify import numpy as np import cv2
app = Flask(name)
DIGITS_LOOKUP = { (1,1,1,0,1,1,1):0, (0,0,1,0,0,1,0):1, (1,0,1,1,1,0,1):2, (1,0,1,1,0,1,1):3, (0,1,1,1,0,1,0):4, (1,1,0,1,0,1,1):5, (1,1,0,1,1,1,1):6, (1,0,1,0,0,1,0):7, (1,1,1,1,1,1,1):8, (1,1,1,1,0,1,1):9 }
def detect_weight(image):
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray,0,255,
cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

contours,_ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digits = []

for c in contours:

    x,y,w,h = cv2.boundingRect(c)

    if h < 30:
        continue

    roi = thresh[y:y+h, x:x+w]

    segments = [
    roi[0:int(h*0.2), :],
    roi[int(h*0.2):int(h*0.4), 0:int(w*0.3)],
    roi[int(h*0.2):int(h*0.4), int(w*0.7):w],
    roi[int(h*0.4):int(h*0.6), :],
    roi[int(h*0.6):int(h*0.8), 0:int(w*0.3)],
    roi[int(h*0.6):int(h*0.8), int(w*0.7):w],
    roi[int(h*0.8):h, :]
    ]

    on = []

    for seg in segments:
        area = cv2.countNonZero(seg)
        if area > (seg.size * 0.3):
            on.append(1)
        else:
            on.append(0)

    digit = DIGITS_LOOKUP.get(tuple(on),None)

    if digit is not None:
        digits.append(str(digit))

return "".join(digits)
@app.route("/") def home(): return "Weight detection server running"
@app.route("/detect", methods=["POST"]) def detect():
img_bytes = request.data

if not img_bytes:
    return jsonify({"weight":"0"})

npimg = np.frombuffer(img_bytes, np.uint8)

img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

weight = detect_weight(img)

return jsonify({"weight":weight})
if name == "main": app.run(host="0.0.0.0", port=10000)