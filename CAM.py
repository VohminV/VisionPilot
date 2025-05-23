import cv2
import numpy as np
import time
import json
import os
from ultralytics import YOLO
from collections import deque
import logging
import math

# Настройка логгера
logging.basicConfig(filename='detections.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Инициализация модели YOLO сегментации
model = YOLO('Y.onnx', task='segment')

# Параметры отображения
screen_width = 720
screen_height = 576
cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

offset_buffer = deque(maxlen=20)
angle_buffer = deque(maxlen=20)  # буфер для углов

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

def display_frame(frame_resized, isolated_obj, fps, center_x, center_y, avg_x, avg_y, avg_angle):
    detection_preview_size = 200
    cropped_resized = cv2.resize(isolated_obj, (detection_preview_size, detection_preview_size))
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
    cap.set(cv2.CAP_PROP_FPS, 60)
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

        results = model(cropped_frame, imgsz=320, conf=0.8)

        offset_x, offset_y = 0, 0
        angle = 0

        isolated_obj = cropped_frame.copy()  # Рабочий кадр для отображения с нарисованной стрелкой

        if results and results[0].masks is not None:
            classes = results[0].boxes.cls
            classes_np = classes.cpu().numpy()
            indices = np.where(classes_np == 0)[0]

            if len(indices) > 0:
                # Получаем координаты маски объекта (без применения маски к изображению)
                polygon_points_list = results[0].masks.xy[indices[0]]

                ys = [pt[1] for pt in polygon_points_list]
                xs = [pt[0] for pt in polygon_points_list]

                if xs and ys:
                    cX = int(np.mean(xs))
                    cY = int(np.mean(ys))

                    offset_x = cX - size // 2
                    offset_y = cY - size // 2

                    points = np.column_stack((xs, ys))
                    rect = cv2.minAreaRect(points)
                    (rect_cx, rect_cy), (w, h), angle = rect
                    if w < h:
                        angle = 90 + angle

                    # Рисуем стрелку и центр объекта на cropped_frame
                    length = 50
                    angle_rad = math.radians(angle)
                    start_point = (cX, cY)
                    end_point = (
                        int(start_point[0] + length * math.cos(angle_rad)),
                        int(start_point[1] + length * math.sin(angle_rad)),
                    )
                    cv2.arrowedLine(isolated_obj, start_point, end_point, (255, 0, 0), 3)
                    cv2.circle(isolated_obj, start_point, 5, (0, 0, 255), -1)

                    logging.info(f"Angle: {angle:.2f} degrees")

        offset_buffer.append((offset_x, offset_y))
        angle_buffer.append(angle)

        avg_x = int(np.mean([x for x, _ in offset_buffer]))
        avg_y = int(np.mean([y for _, y in offset_buffer]))

        # Усреднение угла с учётом цикличности
        sin_sum = np.mean([math.sin(math.radians(a)) for a in angle_buffer])
        cos_sum = np.mean([math.cos(math.radians(a)) for a in angle_buffer])
        avg_angle = math.degrees(math.atan2(sin_sum, cos_sum))
        if avg_angle < 0:
            avg_angle += 360

        save_offset(avg_x, avg_y)

        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        display_frame(frame_resized, isolated_obj, fps, center_x, center_y, avg_x, avg_y, avg_angle)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting program.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(model)
