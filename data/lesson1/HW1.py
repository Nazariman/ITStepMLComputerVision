import numpy as np
import cv2 

#1
img = cv2.imread("data/lesson1/Lenna.png", cv2.IMREAD_GRAYSCALE)

mask = img > 128 

output = np.zeros_like(img)

output[mask] = 255

cv2.imshow('Binary Mask', output)


#2
img = cv2.imread("data/lesson1/baboo.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow('1',img[20:50])

cv2.waitKey(0)
cv2.destroyAllWindows()
