import cv2

def create_subtractor():
    # MOG2 cho che khuất tốt
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40)
