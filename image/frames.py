import cv2
import os
import numpy as np
import random

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def random_transform(frame):
    h, w = frame.shape[:2]

    if random.random() < 0.8:
        frame = rotate_image(frame, random.uniform(-10, 10))
    if random.random() < 0.6:
        tx, ty = random.uniform(-0.05, 0.05) * w, random.uniform(-0.05, 0.05) * h
        M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(frame, M_translate, (w, h), borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.6:
        scale = random.uniform(0.95, 1.05)
        M_scale = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        frame = cv2.warpAffine(frame, M_scale, (w, h), borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.3:
        frame = cv2.flip(frame, 1)

    return frame

def extract_frames(video_path, output_folder, target_fps=20, prefix="video"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Ошибка при открытии: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / target_fps))
    saved_count, frame_idx = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame = random_transform(frame)

            filename = os.path.join(output_folder, f"{prefix}_frame_{saved_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"✅ {os.path.basename(video_path)} → {saved_count} кадров сохранено.")

# ==== Запуск ====
video_folder = "videos"   # Где лежат видео
output_folder = "frames"  # Все кадры идут сюда
target_fps = 30

for filename in os.listdir(video_folder):
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_path = os.path.join(video_folder, filename)
        video_prefix = os.path.splitext(filename)[0]  # Убираем расширение
        extract_frames(video_path, output_folder, target_fps=target_fps, prefix=video_prefix)
