import os
import cv2

def resize_images_recursive1(folder_path, filename_pattern="frame1_", target_size=(320, 320)):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith(filename_pattern) and file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Ошибка: Не удалось загрузить {file_path}")
                    continue
                
                resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(file_path, resized_image)
                print(f"Файл {file_path} сжат до {target_size}")
               
folder_to_search = "./frames"  # Укажите нужную папку
resize_images_recursive1(folder_to_search)