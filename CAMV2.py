import cv2
import numpy as np
import time
import json
import os
from ultralytics import YOLO
from collections import defaultdict, deque
import logging

# Настройка логгера
logging.basicConfig(filename='detections.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Инициализация обычной модели YOLO
model = YOLO('./object.onnx')  # Например, yolo8n.onnx

# Параметры отображения
screen_width = 720
screen_height = 576
cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# История треков (не используется)
track_history = defaultdict(list)

# Буфер последних 20 смещений
offset_buffer = deque(maxlen=20)

def save_offset(avg_x, avg_y):
    data = {'x': avg_x, 'y': avg_y}
    tmp_filename = 'offsets_tmp.json'
    final_filename = 'offsets.json'
    try:
        with open(tmp_filename, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_filename, final_filename)
    except Exception as e:
        logging.error(f"Error saving offsets: {e}")

def crop_frame(frame, center_x, center_y, size):
    crop_x1 = max(center_x - size // 2, 0)
    crop_x2 = min(center_x + size // 2, frame.shape[1])
    crop_y1 = max(center_y - size // 2, 0)
    crop_y2 = min(center_y + size // 2, frame.shape[0])
    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

def display_frame(frame_resized, cropped_frame, fps, center_x, center_y, avg_x, avg_y):
    detection_preview_size = 200
    cropped_resized = cv2.resize(cropped_frame, (detection_preview_size, detection_preview_size))
    y_offset = screen_height - detection_preview_size
    x_offset = screen_width - detection_preview_size
    frame_resized[y_offset:y_offset + detection_preview_size, x_offset:x_offset + detection_preview_size] = cropped_resized

    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.circle(frame_resized, (center_x, center_y), 5, (0, 0, 255), -1)

    detection_point_x = center_x + avg_x
    detection_point_y = center_y + avg_y
    cv2.circle(frame_resized, (detection_point_x, detection_point_y), 5, (0, 255, 0), -1)

    cv2.imshow("Detection", frame_resized)

def main(model):
    cap = cv2.VideoCapture(0)
    size = 320
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame from camera.")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        cropped_frame = crop_frame(frame, center_x, center_y, size)

        results = model(cropped_frame, imgsz=320, conf=0.6)
        offset_x, offset_y = 0, 0

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xywh.cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, obj_class, confidence in zip(boxes, classes, confidences):
                x, y, w, h = map(int, box)

                cropped_center_x = cropped_frame.shape[1] // 2
                cropped_center_y = cropped_frame.shape[0] // 2
                offset_x = x - cropped_center_x
                offset_y = y - cropped_center_y

                cv2.rectangle(cropped_frame, 
                              (int(x - w / 2), int(y - h / 2)), 
                              (int(x + w / 2), int(y + h / 2)), 
                              (0, 255, 0), 2)
                break  # Обрабатываем только первый объект

        # Обновление буфера и усреднение
        offset_buffer.append((offset_x, offset_y))
        avg_x = int(np.mean([x for x, _ in offset_buffer]))
        avg_y = int(np.mean([y for _, y in offset_buffer]))
        save_offset(avg_x, avg_y)

        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        display_frame(frame_resized, cropped_frame, fps, center_x, center_y, avg_x, avg_y)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting program.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(model)
