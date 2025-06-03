import cv2
import numpy as np 

# task 1
img = cv2.imread("data/lesson3/sonet.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Original", img)

# 2. Розмиття (для зменшення шуму)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 3. Адаптивна бінаризація
binary = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY,
    7,  
    2   
)

# 4. Очищення шумів морфологією
kernel = np.ones((3, 3), np.uint8)
denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 5. Виведення
cv2.imshow("Original", img)
cv2.imshow("Binary", binary)
cv2.imshow("Denoised", denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()