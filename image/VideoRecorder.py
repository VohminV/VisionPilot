import sys
import cv2
import time
import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class VideoRecorderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Запись с камеры (PyQt5)")
        self.setGeometry(100, 100, 800, 600)

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise Exception("Камера не найдена")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Камера запущена с разрешением: {self.width}x{self.height}, FPS: {fps:.2f}")

        self.recording = False

        self.image_label = QLabel(self)
        self.start_button = QPushButton("Начать запись", self)
        self.stop_button = QPushButton("Остановить запись", self)
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame
            h, w, _ = frame.shape
            mean_color = frame.mean(axis=(0, 1)).astype(int)
            print(f"Кадр: {w}x{h}, Средний цвет: {mean_color}")

            #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_img))

            # Если запись активна, записываем кадр в файл
            if self.recording:
                self.out.write(frame)

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            # Настройки для записи видео
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            filename = datetime.datetime.now().strftime("input.avi")
            self.out = cv2.VideoWriter(filename, fourcc, 30.0, (self.width, self.height))

            self.record_start_time = time.time()  # Сохраняем время начала записи

            # Запись в течение 60 секунд или до остановки
            self.timer.timeout.connect(self.check_recording_time)

            print(f"Началась запись: {filename}")

    def check_recording_time(self):
        # Прекратить запись через 60 секунд
        if time.time() - self.record_start_time > 60:
            self.stop_recording()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            # Завершаем запись и сохраняем файл
            if hasattr(self, 'out'):
                self.out.release()

            print("Запись завершена.")
            QMessageBox.information(self, "Запись завершена", "Видео сохранено.")

            # Закрытие приложения после завершения записи
            QApplication.quit()

    def closeEvent(self, event):
        self.stop_recording()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoRecorderApp()
    window.show()
    sys.exit(app.exec_())
