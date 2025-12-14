import cv2
import numpy as np

def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BAYER_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    return edges