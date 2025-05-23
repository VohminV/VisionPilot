import cv2
import os
import numpy as np
import random

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def random_transform(frame):
    h, w = frame.shape[:2]

    # Поворот
    if random.random() < 0.8:
        angle = random.uniform(-10, 10)
        frame = rotate_image(frame, angle)

    # Сдвиг
    if random.random() < 0.6:
        tx = random.uniform(-0.05, 0.05) * w
        ty = random.uniform(-0.05, 0.05) * h
        M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(frame, M_translate, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Масштаб
    if random.random() < 0.6:
        scale = random.uniform(0.95, 1.05)
        M_scale = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        frame = cv2.warpAffine(frame, M_scale, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Иногда флип
    if random.random() < 0.3:
        frame = cv2.flip(frame, 1)

    return frame
    
def extract_frames(video_path, output_folder, target_fps=20, rotation_angle=None):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / original_fps
    target_total = int(duration * target_fps)

    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        transformed_frame = random_transform(frame)
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
        cv2.imwrite(frame_filename, transformed_frame)
        saved_count += 1


    cap.release()
    print(f"Saved {saved_count}")

video_path = "input.avi"
output_folder = "frames"
extract_frames(video_path, output_folder, target_fps=30, rotation_angle=random.uniform(-45, 45))