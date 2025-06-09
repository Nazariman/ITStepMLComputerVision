"""
Відкрийте відео з файлу data\lesson8\meetings.mp4
Застосуйте детекцію та виведіть результат, підберіть
параметри
Можете змінити розмір кадру для кращої візуалізації
cv2.resize()
"""
from ultralytics import YOLO
import cv2

# Завантаження моделі (YOLOv8n - найлегша, можна замінити на 'yolov8s' або 'yolov8m')
model = YOLO('yolov8n.pt')

# Відкриття відео
video_path = 'data/lesson8/meetings.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Зменшуємо розмір для швидкості
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

    results = model(frame, verbose=False)[0]

    # Малюємо знайдені об'єкти
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = box.conf[0]

        # беремо тільки координати "person" прямокутника
        if label == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Показ результату
    cv2.imshow("YOLOv8: Об'єкти на відео", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
