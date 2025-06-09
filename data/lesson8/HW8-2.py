from ultralytics import YOLO
import cv2

# Завантаження моделі YOLOv8n
model = YOLO('yolov8n.pt')

video_path = 'data/lesson8/meetings.mp4'
cap = cv2.VideoCapture(video_path)

start_showing = False  # прапорець для запуску показу

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Зменшення масштабу для швидшої обробки
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

    # Детекція
    results = model(frame, verbose=False)[0]

    # Підрахунок кількості людей
    person_count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label == 'person':
            person_count += 1

    # Якщо ще не почали показ, перевіряємо, чи є 5+ людей
    if not start_showing:
        if person_count >= 5:
            start_showing = True
        else:
            continue  # пропускаємо кадр, поки людей менше 5

    # Малюємо прямокутники для людей
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = box.conf[0]

        if label == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Відображення кадру
    cv2.imshow("YOLOv8: Люди на відео", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
