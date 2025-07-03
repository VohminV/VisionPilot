#include <GY87.h>
#include <Arduino.h>

// Инициализация GY87 (SDA = GPIO21, SCL = GPIO22)
GY87 sensor(21, 22);

#define LED_PIN 2

// Структура для передачи данных по Serial
struct SensorData {
  float altitude;
  float headingX;
};

// Расчёт высоты по давлению и температуре
float calcAltitude(float pressure, float temperature) {
  const float P0 = 101325.0;
  float tempK = temperature + 273.15;
  return (tempK / 0.0065) * (1 - pow(pressure / P0, 0.1903));
}

// Считывание данных с датчиков
SensorData readSensor() {
  SensorData data = {0.0, 0.0};

  if (sensor.read()) {
    data.altitude = calcAltitude(sensor.pressure, sensor.temperature);
    data.headingX = sensor.heading;
  }

  return data;
}

// Отправка структуры как бинарных данных по Serial
void sendSensorDataSerial(const SensorData &data) {
  Serial.write((const uint8_t*)&data, sizeof(SensorData));
}

void setup() {
  pinMode(LED_PIN, OUTPUT);

  Serial.begin(115200);
  sensor.begin();
}

void loop() {
  SensorData data = readSensor();

  if (data.altitude != 0.0 || data.headingX != 0.0) {
    sendSensorDataSerial(data);
  }

  digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  delay(100);
}
