#include <GY87.h>
#include <Arduino.h>

// Khởi tạo GY87 với SDA = GPIO21, SCL = GPIO22
GY87 sensor(21, 22);

#define LED_PIN 2
bool blinkState = false;

// Функция расчёта высоты (в метрах) из давления (Па) и температуры (°C)
float calcAltitude(float pressure, float temperature) {
  const float P0 = 101325.0;   // давление на уровне моря в Па
  float tempK = temperature + 273.15;  // °C -> Кельвин
  float altitude = (tempK / 0.0065) * (1 - pow(pressure / P0, 0.1903));
  return altitude;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);

  if (sensor.begin()) {
    Serial.println("GY-87 инициализирован успешно!");
  } else {
    Serial.println("Ошибка инициализации GY-87!");
    while (1);
  }

  Serial.println("heading\taltitude");  // заголовки для Serial Plotter
}

void loop() {
  if (sensor.read()) {
    float altitude = calcAltitude(sensor.pressure, sensor.temperature);

    Serial.print(sensor.heading); Serial.print("\t");
    Serial.println(altitude);
  } else {
    Serial.println("Ошибка чтения данных!");
  }

  blinkState = !blinkState;
  digitalWrite(LED_PIN, blinkState);

  delay(100);  // обновление 10 раз в секунду
}
