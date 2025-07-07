import serial
import time
import struct
import os
import random
import json
import logging
import math
import threading
from pymavlink import mavutil
# Настройка логгера
logging.basicConfig(
    level=logging.INFO,  # Уровень логирования (можно DEBUG для более подробного вывода)
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("/home/orangepi/Documents/YOLO/uart_forwarder.log"),  # Лог в файл
        logging.StreamHandler()  # Также вывод в консоль
    ]
)

crc8tab = [
    0x00, 0xD5, 0x7F, 0xAA, 0xFE, 0x2B, 0x81, 0x54, 0x29, 0xFC, 0x56, 0x83, 0xD7, 0x02, 0xA8, 0x7D,
    0x52, 0x87, 0x2D, 0xF8, 0xAC, 0x79, 0xD3, 0x06, 0x7B, 0xAE, 0x04, 0xD1, 0x85, 0x50, 0xFA, 0x2F,
    0xA4, 0x71, 0xDB, 0x0E, 0x5A, 0x8F, 0x25, 0xF0, 0x8D, 0x58, 0xF2, 0x27, 0x73, 0xA6, 0x0C, 0xD9,
    0xF6, 0x23, 0x89, 0x5C, 0x08, 0xDD, 0x77, 0xA2, 0xDF, 0x0A, 0xA0, 0x75, 0x21, 0xF4, 0x5E, 0x8B,
    0x9D, 0x48, 0xE2, 0x37, 0x63, 0xB6, 0x1C, 0xC9, 0xB4, 0x61, 0xCB, 0x1E, 0x4A, 0x9F, 0x35, 0xE0,
    0xCF, 0x1A, 0xB0, 0x65, 0x31, 0xE4, 0x4E, 0x9B, 0xE6, 0x33, 0x99, 0x4C, 0x18, 0xCD, 0x67, 0xB2,
    0x39, 0xEC, 0x46, 0x93, 0xC7, 0x12, 0xB8, 0x6D, 0x10, 0xC5, 0x6F, 0xBA, 0xEE, 0x3B, 0x91, 0x44,
    0x6B, 0xBE, 0x14, 0xC1, 0x95, 0x40, 0xEA, 0x3F, 0x42, 0x97, 0x3D, 0xE8, 0xBC, 0x69, 0xC3, 0x16,
    0xEF, 0x3A, 0x90, 0x45, 0x11, 0xC4, 0x6E, 0xBB, 0xC6, 0x13, 0xB9, 0x6C, 0x38, 0xED, 0x47, 0x92,
    0xBD, 0x68, 0xC2, 0x17, 0x43, 0x96, 0x3C, 0xE9, 0x94, 0x41, 0xEB, 0x3E, 0x6A, 0xBF, 0x15, 0xC0,
    0x4B, 0x9E, 0x34, 0xE1, 0xB5, 0x60, 0xCA, 0x1F, 0x62, 0xB7, 0x1D, 0xC8, 0x9C, 0x49, 0xE3, 0x36,
    0x19, 0xCC, 0x66, 0xB3, 0xE7, 0x32, 0x98, 0x4D, 0x30, 0xE5, 0x4F, 0x9A, 0xCE, 0x1B, 0xB1, 0x64,
    0x72, 0xA7, 0x0D, 0xD8, 0x8C, 0x59, 0xF3, 0x26, 0x5B, 0x8E, 0x24, 0xF1, 0xA5, 0x70, 0xDA, 0x0F,
    0x20, 0xF5, 0x5F, 0x8A, 0xDE, 0x0B, 0xA1, 0x74, 0x09, 0xDC, 0x76, 0xA3, 0xF7, 0x22, 0x88, 0x5D,
    0xD6, 0x03, 0xA9, 0x7C, 0x28, 0xFD, 0x57, 0x82, 0xFF, 0x2A, 0x80, 0x55, 0x01, 0xD4, 0x7E, 0xAB,
    0x84, 0x51, 0xFB, 0x2E, 0x7A, 0xAF, 0x05, 0xD0, 0xAD, 0x78, 0xD2, 0x07, 0x53, 0x86, 0x2C, 0xF9
]

channels_old = None
data_without_crc_old = None
speed_old = None
correction_active = False
# Объект события для остановки потока
stop_event = threading.Event()
# Флаг для проверки, запущен ли поток
is_thread_running = False
is_calibrating = False

current_altitude = 0.0
climb = 0.0

from smbus import SMBus
bus = SMBus(5)
    
def crc8(data):
    crc = 0
    for byte in data:
        crc = crc8tab[crc ^ byte]
    return crc

def calibrate_qmc5883l(duration_sec=30, delay=0.01):
    global bus
    QMC5883L_ADDR = 0x0D

    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')

    start_time = time.time()
    while time.time() - start_time < duration_sec:
        data = bus.read_i2c_block_data(QMC5883L_ADDR, 0x00, 6)
        mag_x = (data[1] << 8) | data[0]
        mag_y = (data[3] << 8) | data[2]
        mag_z = (data[5] << 8) | data[4]

        if mag_x >= 32768: mag_x -= 65536
        if mag_y >= 32768: mag_y -= 65536
        if mag_z >= 32768: mag_z -= 65536

        min_x = min(min_x, mag_x)
        max_x = max(max_x, mag_x)
        min_y = min(min_y, mag_y)
        max_y = max(max_y, mag_y)
        min_z = min(min_z, mag_z)
        max_z = max(max_z, mag_z)

        time.sleep(delay)

    offset_x = -((max_x + min_x) // 2)
    offset_y = -((max_y + min_y) // 2)
    offset_z = -((max_z + min_z) // 2)

    with open('mag_offsets.json', 'w') as f:
        json.dump({"x": offset_x, "y": offset_y, "z": offset_z}, f)

    print(f"[✔] Калибровка QMC5883L завершена. Смещения: x={offset_x}, y={offset_y}, z={offset_z}")

# Function to pack channels into the CRSF payload format (16 channels, 11 bits each)
def pack_channels(channel_data):
    # channel data: array of 16 integers
    channel_data = list(reversed(channel_data))
    pack_bit = []
    for idx, channel in enumerate(channel_data):
        pack_bit[idx*11: (idx+1)*11] = "{0:011b}".format(channel)
    pack_bit=''.join(pack_bit)
    pack_byte = []
    for idx in range(22):
        current_byte = int(pack_bit[idx*8:(idx+1)*8], 2)
        pack_byte.append(current_byte)
    pack_byte = list(reversed(pack_byte))
    return pack_byte
    
# Function to extract channels from the CRSF payload (22 bytes representing 16 channels)
def extract_channels(data):
    channels = []
    if len(data) != 22:  # CRSF packed channel data is 22 bytes
        return channels

    # Convert bytes to binary string
    bits = ''.join(format(byte, '08b')[::-1] for byte in data)

    # Extract 11-bit channel values
    for i in range(16):  # CRSF supports up to 16 channels
        start = i * 11
        end = start + 11
        if end <= len(bits):
            channel_bits = bits[start:end][::-1]
            channel_value = int(channel_bits, 2)
            channels.append(channel_value)

    return channels

def update_rc_channels_in_background(channels_old, uart4, data_without_crc_old):
    import json
    import logging
    import time
    import math

    global bus

    # Адреса датчиков
    QMC5883L_ADDR = 0x0D
    MAGNETIC_DECLINATION = 8.2667
    # CRSF параметры
    CENTER_TICKS = 992
    MIN_TICKS = 172
    MAX_TICKS = 1811

    FRAME_SIZE = 320
    MAX_OFFSET_PX = FRAME_SIZE // 2
    MAX_DEFLECTION_US = 300
    MAX_DEFLECTION_TICKS = int(MAX_DEFLECTION_US * 8 / 5)

    ALT_SCALE = 100
    CLIMB_SCALE = 100

    P_int = 4   # 400 / 100
    D_int = 1   # 100 / 100
    FF_int = 2  # 200 / 100

    def calculate_heading():
        # Загружаем калибровочные смещения
        try:
            with open('mag_offsets.json', 'r') as f:
                offsets = json.load(f)
                mag_offset_x = offsets.get('x', 0)
                mag_offset_y = offsets.get('y', 0)
        except Exception:
            mag_offset_x = 0
            mag_offset_y = 0

        # Читаем 6 байт данных магнитометра
        data = bus.read_i2c_block_data(QMC5883L_ADDR, 0x00, 6)
        mag_x = (data[1] << 8) | data[0]
        mag_y = (data[3] << 8) | data[2]

        # Преобразуем в signed int
        if mag_x >= 32768: mag_x -= 65536
        if mag_y >= 32768: mag_y -= 65536

        # Применяем калибровку
        mag_x += mag_offset_x
        mag_y += mag_offset_y

        # Вычисляем азимут (heading) в градусах
        heading = math.atan2(mag_y, mag_x) * 180 / math.pi
        if heading < 0:
            heading += 360

        # Применяем магнитное склонение (восточное → +)
        heading += MAGNETIC_DECLINATION
        if heading >= 360:
            heading -= 360

        return heading

    def heading_diff(desired, current):
        return (desired - current + 540) % 360 - 180

    # 🌐 Начальные значения
    desired_heading = calculate_heading()

    #initial_throttle = channels_old[2]
    initial_yaw = channels_old[3]

    while not stop_event.is_set():
        # Чтение сенсоров
        current_heading = calculate_heading()

        try:
            with open('offsets.json', 'r') as f:
                offsets = json.load(f)
                offset_x = offsets.get('x', 0)
                offset_y = offsets.get('y', 0)
        except:
            offset_x = 0
            offset_y = 0

        offset_x = max(-MAX_OFFSET_PX, min(offset_x, MAX_OFFSET_PX))
        offset_y = max(-MAX_OFFSET_PX, min(offset_y, MAX_OFFSET_PX))

        def scale_offset_to_ticks(offset_px):
            return int(offset_px * MAX_DEFLECTION_TICKS / MAX_OFFSET_PX)

        roll_ticks = scale_offset_to_ticks(offset_x)
        pitch_ticks = scale_offset_to_ticks(offset_y)

        channels_old[0] = max(MIN_TICKS, min(MAX_TICKS, CENTER_TICKS + roll_ticks))
        channels_old[1] = max(MIN_TICKS, min(MAX_TICKS, CENTER_TICKS + pitch_ticks))
        """
        error = desired_altitude_int - current_altitude_int
        p_out = P_int * error
        d_out = -D_int * climb_int
        ff_out = FF_int * climb_int
        offset = p_out + d_out + ff_out
        offset = max(-MAX_DEFLECTION_TICKS, min(MAX_DEFLECTION_TICKS, offset))

        throttle = initial_throttle + offset
        throttle = max(MIN_TICKS, min(MAX_TICKS, throttle))
        channels_old[2] = throttle
        """
        yaw_error = max(-30, min(30, heading_diff(desired_heading, current_heading)))

        if abs(yaw_error) > 15:
            yaw_normalized = max(-1.0, min(1.0, yaw_error / 30.0))
            yaw_ticks = int(yaw_normalized * MAX_DEFLECTION_TICKS)
            yaw_channel = max(MIN_TICKS, min(MAX_TICKS, CENTER_TICKS + yaw_ticks))
            channels_old[3] = yaw_channel
        else:
            channels_old[3] = CENTER_TICKS
    
        packed_channels = pack_channels(channels_old)
        data_without_crc_old[3:25] = packed_channels
        crc = crc8(data_without_crc_old[2:25])
        updated_data = data_without_crc_old + [crc]
        uart4.write(bytes(updated_data))

        #logging.info(f"HEAD: {current_heading:.1f}, ΔH={yaw_error:.1f}°")

    global is_thread_running
    is_thread_running = False

# Функция для запуска потока обновления RC каналов
def start_update_rc_channels_thread(channels_old, uart4, data_without_crc_old):
    global is_thread_running  # нужно явно указать, что используем глобальную переменную
    if not is_thread_running:
        stop_event.clear()
        update_thread = threading.Thread(
            target=update_rc_channels_in_background,
            args=(channels_old, uart4, data_without_crc_old)
        )
        update_thread.daemon = True  # Поток завершится при завершении основного потока
        update_thread.start()
        is_thread_running = True  # Устанавливаем флаг после запуска

def calibrate_thread():
    global is_calibrating
    is_calibrating = True
    calibrate_qmc5883l(duration_sec=30, delay=0.01)
    is_calibrating = False
    
# Функция для обновления RC каналов
def update_rc_channels(data, uart4):
    global channels_old, data_without_crc_old, is_thread_running, is_calibrating

    if len(data) < 26:
        #print(f"❌ Недостаточно данных: {len(data)} байт, нужно минимум 26.")
        return data

    data_without_crc = data[:-1]  # Без последнего байта CRC
    channels = extract_channels(data_without_crc[3:25])

    if len(channels) < 16:
        #print(f"❌ Недостаточно каналов для обновления. Найдено {len(channels)} каналов.")
        return
    
    #print(f"Канал 11: {channels[11]}")  # Активность канала для контроля

    # Если канал 11 больше 1700 и поток еще не запущен, запускаем его
    if channels[11] > 1700:
        if not is_thread_running:  # Если поток еще не запущен
            if channels_old is None:
                channels_old = channels.copy()
            if data_without_crc_old is None:
                data_without_crc_old = data_without_crc
            # Запускаем поток для обновления каналов в фоне
            start_update_rc_channels_thread(channels_old, uart4, data_without_crc_old)
    elif channels[3] > 1700 and channels[0] < 300 and not is_calibrating:
        t = threading.Thread(target=calibrate_thread)
        t.start()
    # Завершаем выполнение, если канал 11 меньше или равен 1700
    else:
        stop_event.set()
        channels_old = None
        data_without_crc_old = None
        #print(f"Канал 11 меньше или равен 1700")
        uart4.write(bytes(data))
            
# Функция для форвардинга пакетов
def uart_forwarder(uart3, uart4):
    global is_thread_running
    packet_buffer = []
    
    while True:
        try:
            # Чтение данных из uart3
            data = uart3.read(512)
            if not data:
                continue

            packet_buffer.extend(data)
            # Обрабатываем пакеты
            while len(packet_buffer) >= 4:
                try:
                    # Проверяем начало пакета
                    if packet_buffer[0] != 0xC8:
                        #print(f"❌ Неправильный байт начала пакета: {packet_buffer[0]:02x}")
                        packet_buffer.pop(0)
                        continue

                    length = packet_buffer[1]  # Длина пакета из второго байта
                    print(f"Ожидаемая длина пакета: {length}")

                    if len(packet_buffer) < length + 2:
                        #print("❌ Пакет неполный, ожидаем дополнительные данные...")
                        break

                    packet = packet_buffer[:length + 2]
                    packet_buffer = packet_buffer[length + 2:]

                    #print(f"Получен пакет: {' '.join(f'{x:02x}' for x in packet)}")

                    if packet[2] == 0x16:  # Проверка на тип пакета
                        update_rc_channels(packet, uart4)
                    else:
                        if not is_thread_running:
                            uart4.write(bytes(packet))
                        #print(f"Записано байтов в UART4: {len(packet)}")

                except Exception as e:
                    logging.error(f"❌ Ошибка при обработке пакета: {e}")
                    # Очистка буфера для предотвращения зависания
                    packet_buffer.clear()

        except Exception as e:
            logging.error(f"❌ Ошибка при чтении данных с UART3: {e}")

# Основная функция
def main():
    logging.info("🚀 Запуск UART forwarder...")
    
    uart3 = serial.Serial('/dev/ttyS3', 115200, timeout=0)  # Настройте нужную скорость
    uart4 = serial.Serial('/dev/ttyS4', 420000, timeout=0)  # Настройте нужную скорость

    uart_forwarder(uart3, uart4)

    uart3.close()
    uart4.close()

if __name__ == "__main__":
    main()