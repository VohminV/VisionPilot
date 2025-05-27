import cv2
import numpy as np
import time
import json
import os
from ultralytics import YOLO
from collections import deque
import logging
import math

logging.basicConfig(filename='detections.log', level=logging.INFO, format='%(asctime)s - %(message)s')

model = YOLO('Y.onnx', task='segment')

screen_width = 720
screen_height = 576
cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

offset_buffer = deque(maxlen=20)
angle_buffer = deque(maxlen=20)

def average_angle_deg(angles):
    # Среднее углов с учётом цикличности
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    avg_rad = math.atan2(sin_sum, cos_sum)
    avg_deg = math.degrees(avg_rad)
    return avg_deg

def save_offset(avg_x, avg_y, angle_deg):
    data = {'x': avg_x, 'y': avg_y, 'angle': round(angle_deg, 2)}
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

def yaw_to_crsf_ticks(yaw_deg):
    yaw_norm = (yaw_deg + 180) % 360 - 180
    yaw_clamped = max(min(yaw_norm, 90), -90)
    ticks = int((yaw_clamped + 90) * (2015 - 992) / 180 + 992)
    return ticks

def display_frame(frame_resized, isolated_obj, fps, center_x, center_y, avg_x, avg_y, avg_angle_deg, crsf_ticks):
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

    # Показываем угол на основном кадре
    cv2.putText(frame_resized, f"Angle: {avg_angle_deg:.1f}°", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Detection", frame_resized)

def pca_orientation(points):
    # Центрируем точки
    mean = np.mean(points, axis=0)
    centered = points - mean

    # Ковариационная матрица
    cov = np.cov(centered, rowvar=False)

    # Собственные значения и векторы
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Главный компонент — вектор с максимальным собственным значением
    principal_component = eigvecs[:, np.argmax(eigvals)]

    # Принудительно направляем вектор вверх по оси Y (чтобы Y < 0)
    if principal_component[1] > 0:
        principal_component = -principal_component

    # Угол между главным компонентом и осью Y (0, -1 — "вверх")
    angle_rad = math.atan2(principal_component[0], -principal_component[1])
    angle_deg = math.degrees(angle_rad)

    return angle_deg, mean.astype(int)

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
        yaw_deg = 0
        crsf_ticks = yaw_to_crsf_ticks(0)

        isolated_obj = cropped_frame.copy()

        if results and results[0].masks is not None:
            classes = results[0].boxes.cls
            classes_np = classes.cpu().numpy()
            indices = np.where(classes_np == 0)[0]

            if len(indices) > 0:
                polygon_points_list = results[0].masks.xy[indices[0]]

                ys = [pt[1] for pt in polygon_points_list]
                xs = [pt[0] for pt in polygon_points_list]

                if xs and ys:
                    points = np.column_stack((xs, ys)).astype(np.float32)

                    # Получаем угол и центр через PCA
                    angle_pca_deg, center_mass = pca_orientation(points)

                    # Ограничение ±30°
                    if angle_pca_deg > 30:
                        valid_angle = False
                        yaw_deg = 30
                    elif angle_pca_deg < -30:
                        valid_angle = False
                        yaw_deg = -30
                    else:
                        valid_angle = True
                        yaw_deg = angle_pca_deg

                    angle_buffer.append(yaw_deg)
                    avg_angle_deg = average_angle_deg(angle_buffer)

                    offset_x = int(center_mass[0] - size // 2)
                    offset_y = int(center_mass[1] - size // 2)

                    # Верхняя точка — минимальный y
                    top_idx = np.argmin(ys)
                    top_point = np.array([xs[top_idx], ys[top_idx]], dtype=np.int32)

                    # Направление стрелки — по усреднённому углу PCA
                    angle_rad = math.radians(avg_angle_deg)
                    arrow_length = 40
                    tip_x = int(top_point[0] + arrow_length * math.sin(angle_rad))
                    tip_y = int(top_point[1] - arrow_length * math.cos(angle_rad))

                    arrow_color = (0, 255, 0) if valid_angle else (0, 0, 255)
                    thickness = 1

                    cv2.arrowedLine(isolated_obj, tuple(top_point), (tip_x, tip_y), arrow_color, thickness, tipLength=0.3)
                    cv2.circle(isolated_obj, tuple(top_point), 3, arrow_color, -1)
                    cv2.circle(isolated_obj, tuple(center_mass), 3, (255, 0, 0), -1)

                    cv2.putText(isolated_obj, f"Angle: {avg_angle_deg:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)

                    logging.info(f"PCA angle raw: {angle_pca_deg:.2f}, adjusted angle: {yaw_deg:.2f}, offset: ({offset_x}, {offset_y})")

        else:
            avg_angle_deg = 0

        offset_buffer.append((offset_x, offset_y))
        avg_x = int(np.mean([x for x, _ in offset_buffer]))
        avg_y = int(np.mean([y for _, y in offset_buffer]))

        save_offset(avg_x, avg_y, avg_angle_deg)

        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        display_frame(frame_resized, isolated_obj, fps, center_x, center_y, avg_x, avg_y, avg_angle_deg, crsf_ticks)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting program.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(model)
