from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Завантаження моделі сегментації
model = YOLO('data/lesson_seg/brain-tumor-seg.pt')

# Завантаження зображення
img_path = 'data/lesson_seg/tumor1.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Сегментація зображення
results = model(img_rgb)

# Витягування маски (беремо перший об'єкт на зображенні)
mask = results[0].masks.data[0].cpu().numpy()

# Приводимо маску до цілих значень (0 або 1)
mask_bin = (mask > 0.5).astype(np.uint8)

# Розрахунок площі пухлини в пікселях
tumor_pixels = np.sum(mask_bin)
tumor_area_mm2 = tumor_pixels * 0.0025

# Визначення типу пухлини
if tumor_area_mm2 < 10:
    tumor_type = "small"
elif tumor_area_mm2 <= 25:
    tumor_type = "middle"
else:
    tumor_type = "large"

# Створення маскованого зображення (залишаємо тільки пікселі пухлини)
tumor_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_bin)

# Збереження зображення з назвою типу
save_path = f"{tumor_type}.jpg"
cv2.imwrite(save_path, cv2.cvtColor(tumor_masked, cv2.COLOR_RGB2BGR))

# Виведення результатів
print(f"Площа пухлини (пікселів): {tumor_pixels}")
print(f"Площа пухлини (мм²): {tumor_area_mm2:.2f}")
print(f"Тип пухлини: {tumor_type}")

# Показ зображення
plt.imshow(tumor_masked)
plt.title(f"Tumor type: {tumor_type}")
plt.axis('off')
plt.show()
