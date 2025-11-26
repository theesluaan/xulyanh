import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)

def clean_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask
