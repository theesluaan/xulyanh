import cv2
import numpy as np

# Kernel tròn tốt hơn kernel vuông (ít làm méo hình)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# hoặc kernel chữ nhật nhỏ hơn cũng ổn: np.ones((3,3), np.uint8)

def clean_mask(mask):
    # 1. Opening: loại nhiễu + tách các vùng dính nhẹ
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. Closing: vá lỗ thủng bên trong người (rất quan trọng!)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Dilate nhẹ để nối các phần bị đứt (chỉ 1 lần là đủ)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask