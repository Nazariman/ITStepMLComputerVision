import cv2
import numpy as np
from ultralytics import YOLO

# Використовуємо твою функцію
def get_angle(x1, y1, x2, y2, x3, y3):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])

    ab = a - b
    cb = c - b

    dot = ab @ cb
    norm_ab = (ab @ ab) ** 0.5
    norm_cb = (cb @ cb) ** 0.5
    angle = np.arccos(dot / (norm_ab * norm_cb + 1e-6))
    angle = angle / np.pi * 180

    return angle


# Завантаження моделі
model = YOLO("yolov8n-pose.pt")  # або yolov8s-pose.pt

# Відкриття відео
cap = cv2.VideoCapture("data/lesson_pose/sitting.mp4")

# Початкові значення
counter = 0
stage = None
ANGLE_DOWN = 70
ANGLE_UP = 160

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    if len(results.keypoints) > 0:
        # Вибрати лише першу людину (kp = results.keypoints[0])
        keypoints = results.keypoints[0].xy[0].cpu().numpy()
        # Права нога (24-26-28): стегно–коліно–щиколотка
        try:
            hip = keypoints[24]
            knee = keypoints[26]
            ankle = keypoints[28]

            x1, y1 = hip
            x2, y2 = knee
            x3, y3 = ankle

            angle = get_angle(x1, y1, x2, y2, x3, y3)

            # Логіка присідання
            if angle < ANGLE_DOWN:
                stage = "down"
            if angle > ANGLE_UP and stage == "down":
                stage = "up"
                counter += 1

            # Вивід кута та кількості
            cv2.putText(frame, f"Angle: {int(angle)}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Squats: {counter}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Малювання ключових точок
            for pt in [hip, knee, ankle]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)

        except:
            continue
        
    frame = cv2.resize(frame, (960, 540))  # або (1280, 720) — як зручно

    cv2.imshow("YOLOv8 Pose Squat Counter", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
