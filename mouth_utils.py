import cv2
import numpy as np
from config import IMG_SIZE

MOUTH_IDX = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291, 308, 78
]

def preprocess_mouth(img):
    if img is None or img.size == 0:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray_img = gray_img / 255.0
    return gray_img.reshape(1, IMG_SIZE, IMG_SIZE, -1)

def get_mouth_roi(frame, landmarks, w, h, pad=35):
    xs = [int(landmarks[i].x * w) for i in MOUTH_IDX]
    ys = [int(landmarks[i].y * h) for i in MOUTH_IDX]

    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)
