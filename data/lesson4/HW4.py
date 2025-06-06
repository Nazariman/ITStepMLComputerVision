import cv2
import numpy as np

def process_image(image_path):
    # Завантаження зображення у відтінках сірого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Гаусове розмиття
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Виявлення країв за допомогою оператора Canny
    edges = cv2.Canny(blurred, 100, 200)

    # Морфологічне закриття для очищення контурів
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Відображення результатів
    cv2.imshow("Origin", image)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edges)
    cv2.imshow("Morphed", morphed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Обробка зображень
process_image("data/lesson4/apple.png")
#process_image("data/lesson4/apple_noised.png")
#process_image("data/lesson4/apple_salt_pepper.png")
