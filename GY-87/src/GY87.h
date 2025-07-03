#ifndef GY87_H
#define GY87_H

#include <Wire.h>

class GY87 {
public:
  GY87(uint8_t sdaPin, uint8_t sclPin);

  bool begin();
  bool read();

  int16_t ax, ay, az; // Акселерометр (raw, ±2g)
  int16_t gx, gy, gz; // Гироскоп (raw, ±250°/с)
  float temperature;  // Температура (°C)
  int32_t pressure;   // Давление (Па)
  int16_t mx, my, mz; // Магнитометр (raw)
  float heading;      // Азимут (градусы)

private:
  uint8_t _sdaPin, _sclPin;

  static const uint8_t MPU6050_ADDR = 0x68;
  static const uint8_t BMP180_ADDR = 0x77;
  static const uint8_t QMC5883L_ADDR = 0x0D;  // Адрес QMC5883L

  // MPU6050 регистры
  static const uint8_t MPU6050_PWR_MGMT_1 = 0x6B;
  static const uint8_t MPU6050_INT_PIN_CFG = 0x37;
  static const uint8_t MPU6050_ACCEL_CONFIG = 0x1C;
  static const uint8_t MPU6050_GYRO_CONFIG = 0x1B;
  static const uint8_t MPU6050_ACCEL_XOUT_H = 0x3B;
  static const uint8_t MPU6050_GYRO_XOUT_H = 0x43;

  // BMP180 регистры
  static const uint8_t BMP180_CAL_AC1 = 0xAA;
  static const uint8_t BMP180_CONTROL = 0xF4;
  static const uint8_t BMP180_DATA = 0xF6;
  static const uint8_t BMP180_TEMP = 0x2E;
  static const uint8_t BMP180_PRESSURE = 0x34;

  // QMC5883L регистры
  static const uint8_t QMC5883L_REG_X_L = 0x00;
  static const uint8_t QMC5883L_REG_CTRL1 = 0x09;
  static const uint8_t QMC5883L_REG_CTRL2 = 0x0A;

  // Калибровочные данные BMP180
  int16_t ac1, ac2, ac3, b1, b2, mb, mc, md;
  uint16_t ac4, ac5, ac6;

  bool writeRegister(uint8_t addr, uint8_t reg, uint8_t value);
  bool readRegisters(uint8_t addr, uint8_t reg, uint8_t *buffer, uint8_t length);
  bool readMPU6050();
  bool readBMP180Calibration();
  bool readBMP180();
  bool initQMC5883L();
  bool readQMC5883L();
};

#endif
