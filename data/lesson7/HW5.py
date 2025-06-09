import cv2 

# Шлях до файлів
input_path = 'data/lesson7/meter.mp4'
output_path = 'data/lesson7/meter_binary.mp4'

# Відкриваємо відео
cap = cv2.VideoCapture(input_path)

# Параметри відео 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # роблю ч/б
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # зберігаю результат
    out.write(binary)

# Звільняємо ресурси
cap.release()
out.release()
print("Відео збережено:", output_path)