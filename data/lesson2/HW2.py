import cv2
import numpy as np
import os
import sys

# Додаємо шлях до кореневої папки, де лежить utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils import trackbar_decorator

# Читання зображення
img = cv2.imread("data/lesson2/darken.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Вирівнювання гістограми
v_eq = cv2.equalizeHist(v)
hsv_eq = cv2.merge([h, s, v_eq])
bgr_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
cv2.imshow("Histogram Equalized", bgr_eq)

# Підвищення яскравості з можливістю керування слайдером
@trackbar_decorator(Brightness=(50, 150))  # 50%–150%
def enhance_value(Brightness):
    factor = Brightness / 100.0
    v_scaled = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    hsv_scaled = cv2.merge([h, s, v_scaled])
    bgr_scaled = cv2.cvtColor(hsv_scaled, cv2.COLOR_HSV2BGR)
    return bgr_scaled

enhance_value()
