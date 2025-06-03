import cv2
import numpy as np

def preprocess_image(
    image_path,
    blur_kernel_size=(5, 5),
    adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshold_type=cv2.THRESH_BINARY,
    block_size=7,
    C=2,
    morph_kernel_size=(3, 3),
    morph_operation=cv2.MORPH_OPEN,
    window_prefix="Result"
):
    # 1. Завантаження зображення у відтінках сірого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Помилка] Зображення не знайдено: {image_path}")
        return

    # 2. Розмиття
    blurred = cv2.GaussianBlur(img, blur_kernel_size, 0)

    # 3. Адаптивна бінарізація
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        adaptive_method,
        threshold_type,
        block_size,
        C
    )

    # 4. Морфологічне очищення шумів
    kernel = np.ones(morph_kernel_size, np.uint8)
    denoised = cv2.morphologyEx(binary, morph_operation, kernel)

    # 5. Виведення
    cv2.imshow(f"{window_prefix} - Original", img)
    cv2.imshow(f"{window_prefix} - Binary", binary)
    cv2.imshow(f"{window_prefix} - Denoised", denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


preprocess_image("data/lesson3/sonet.png")
preprocess_image("data/lesson3/sonet_noised.png")