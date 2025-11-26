import cv2

def apply_threshold(mask):
    _, th = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    return th
