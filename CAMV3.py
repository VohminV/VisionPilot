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

# Инициализация модели YOLO сегментации (укажи путь к своей модели)
model = YOLO('Y.pt')

# Параметры отображения
screen_width = 720
screen_height = 576
cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Буферы последних 20 смещений центра и углов объекта
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

        results = model(cropped_frame, imgsz=320, conf=0.75)

        offset_x, offset_y = 0, 0
        avg_angle = 0

        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            target_class_id = 0  # укажи здесь нужный класс (если у тебя 1 класс — 0)
            # Найти индекс маски с наибольшим confidence для нужного класса
            best_mask_idx = -1
            best_conf = 0
            for i, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
                if cls_id == target_class_id and conf > best_conf:
                    best_conf = conf
                    best_mask_idx = i

            if best_mask_idx >= 0:
                mask_tensor = results[0].masks.data[best_mask_idx]
                mask = mask_tensor.cpu().numpy().astype(np.uint8)

                # Контуры
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(cropped_frame, contours, -1, (0, 255, 255), 2)

                    cnt = contours[0]
                    rect = cv2.minAreaRect(cnt)
                    (center_x_rect, center_y_rect), (w, h), angle = rect
                    if w < h:
                        angle = 90 + angle

                    moments = cv2.moments(mask)
                    if moments["m00"] != 0:
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                    else:
                        cX, cY = 0, 0

                    offset_x = cX - cropped_frame.shape[1] // 2
                    offset_y = cY - cropped_frame.shape[0] // 2

                    cv2.circle(cropped_frame, (cX, cY), 5, (0, 0, 255), -1)
                    length = 50
                    angle_rad = math.radians(angle)
                    end_x = int(cX + length * math.cos(angle_rad))
                    end_y = int(cY + length * math.sin(angle_rad))
                    cv2.arrowedLine(cropped_frame, (cX, cY), (end_x, end_y), (255, 0, 0), 2)

                    logging.info(f"Smoothed angle: {angle:.2f} degrees")

            # --- Визуализация всех масок (для отладки) ---
            #for i, mask_tensor in enumerate(results[0].masks.data):
            #    mask_viz = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            #    cv2.imshow(f"mask_{i}", mask_viz)
            # -------------------------------------------

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
