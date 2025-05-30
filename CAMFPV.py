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

kalman_pos = cv2.KalmanFilter(4, 2)
kalman_pos.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman_pos.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman_pos.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
kalman_pos.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
kalman_pos.statePre = np.zeros((4,1), np.float32)
kalman_pos.statePost = np.zeros((4,1), np.float32)

kalman_angle = cv2.KalmanFilter(2,1)
kalman_angle.measurementMatrix = np.array([[1,0]], np.float32)
kalman_angle.transitionMatrix = np.array([[1,1],[0,1]], np.float32)
kalman_angle.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
kalman_angle.measurementNoiseCov = np.array([[1e-2]], np.float32)
kalman_angle.statePre = np.zeros((2,1), np.float32)
kalman_angle.statePost = np.zeros((2,1), np.float32)

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
    x1, x2 = max(center_x - size//2, 0), min(center_x + size//2, frame.shape[1])
    y1, y2 = max(center_y - size//2, 0), min(center_y + size//2, frame.shape[0])
    return frame[y1:y2, x1:x2].copy()

def yaw_to_crsf_ticks(yaw_deg):
    yaw_norm = (yaw_deg + 180) % 360 - 180
    yaw_clamped = max(min(yaw_norm, 90), -90)
    return int((yaw_clamped + 90) * (2015 - 992) / 180 + 992)

def display_frame(frame_resized, isolated_obj, fps, center_x, center_y, avg_x, avg_y, avg_angle_deg, crsf_ticks, yaw_dir):
    preview_size = 200
    cropped_resized = cv2.resize(isolated_obj, (preview_size, preview_size))
    y_offset = screen_height - preview_size
    x_offset = screen_width - preview_size
    frame_resized[y_offset:y_offset+preview_size, x_offset:x_offset+preview_size] = cropped_resized

    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.circle(frame_resized, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.circle(frame_resized, (center_x + avg_x, center_y + avg_y), 5, (0, 255, 0), -1)
    cv2.putText(frame_resized, f"Angle: {avg_angle_deg:.1f} deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame_resized, f"Yaw: {yaw_dir}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.imshow("Detection", frame_resized)

def pca_orientation(points):
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_component = eigvecs[:, np.argmax(eigvals)]
    if principal_component[1] > 0:
        principal_component = -principal_component
    angle_rad = math.atan2(principal_component[0], -principal_component[1])
    return math.degrees(angle_rad), mean.astype(int)

def main(model):
    cap = cv2.VideoCapture(0)
    size = 320
    prev_time = time.time()
    lost_counter = 0
    tracking_lost = False
    MAX_LOST_FRAMES = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
        cropped_frame = crop_frame(frame, center_x, center_y, size)

        results = model(cropped_frame, imgsz=320, conf=0.8)

        offset_x, offset_y = 0, 0
        yaw_deg = 0
        yaw_dir = "straight"
        isolated_obj = cropped_frame.copy()
        avg_angle_deg = 0

        if results and hasattr(results[0], "masks") and results[0].masks is not None:#if results and results[0].masks is not None:
            classes = results[0].boxes.cls.cpu().numpy()
            indices = np.where(classes == 0)[0]
            if len(indices):
                lost_counter = 0
                tracking_lost = False

                polygon = results[0].masks.xy[indices[0]]
                points = np.array(polygon, dtype=np.float32)
                angle_deg, center_mass = pca_orientation(points)

                kalman_angle.correct(np.array([[np.float32(angle_deg)]]))
                predicted_angle = kalman_angle.predict()
                avg_angle_deg = predicted_angle[0,0]

                kalman_pos.correct(np.array([[np.float32(center_mass[0])], [np.float32(center_mass[1])]]))
                predicted_pos = kalman_pos.predict()
                filtered_x, filtered_y = int(predicted_pos[0,0]), int(predicted_pos[1,0])
                offset_x, offset_y = filtered_x - size//2, filtered_y - size//2

                top_idx = np.argmin(points[:,1])
                top_point = points[top_idx].astype(int)
                angle_rad = math.radians(avg_angle_deg)
                tip_x = int(top_point[0] + 40 * math.sin(angle_rad))
                tip_y = int(top_point[1] - 40 * math.cos(angle_rad))
                arrow_color = (0,255,0) if -90 <= avg_angle_deg <= 90 else (0,0,255)
                cv2.arrowedLine(isolated_obj, tuple(top_point), (tip_x, tip_y), arrow_color, 2, tipLength=0.3)
                cv2.circle(isolated_obj, tuple(top_point), 3, arrow_color, -1)
                cv2.circle(isolated_obj, (filtered_x, filtered_y), 3, (255, 0, 0), -1)
            else:
                lost_counter += 1
        else:
            lost_counter += 1

        if lost_counter >= MAX_LOST_FRAMES:
            tracking_lost = True
            predicted_pos = kalman_pos.predict()
            predicted_angle = kalman_angle.predict()
            avg_angle_deg = predicted_angle[0, 0]
            filtered_x, filtered_y = int(predicted_pos[0,0]), int(predicted_pos[1,0])
            offset_x, offset_y = filtered_x - size//2, filtered_y - size//2

            arrow_color = (128, 128, 128)
            angle_rad = math.radians(avg_angle_deg)
            tip_x = int(filtered_x + 40 * math.sin(angle_rad))
            tip_y = int(filtered_y - 40 * math.cos(angle_rad))
            cv2.arrowedLine(isolated_obj, (filtered_x, filtered_y), (tip_x, tip_y), arrow_color, 2, tipLength=0.3)
            cv2.circle(isolated_obj, (filtered_x, filtered_y), 3, arrow_color, -1)

        offset_buffer.append((offset_x, offset_y))
        avg_x = int(np.mean([x for x, _ in offset_buffer]))
        avg_y = int(np.mean([y for _, y in offset_buffer]))

        angle_norm = (avg_angle_deg + 180) % 360 - 180
        if angle_norm > 90:
            yaw_dir = "right"
            yaw_deg = angle_norm - 90
        elif angle_norm < -90:
            yaw_dir = "left"
            yaw_deg = angle_norm + 90
        else:
            yaw_deg = 0
            
        if yaw_deg > 0:
            crsf_ticks = yaw_to_crsf_ticks(yaw_deg)
        else:
            crsf_ticks = 0
            
        save_offset(avg_x, avg_y, yaw_deg)

        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        display_frame(frame_resized, isolated_obj, fps, center_x, center_y, avg_x, avg_y, avg_angle_deg, crsf_ticks, yaw_dir)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(model)
