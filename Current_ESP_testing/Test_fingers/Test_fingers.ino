#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define I2C_SDA 8
#define I2C_SCL 9
#define PCA9685_ADDRESS 0x40

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDRESS);

// Calibrated Microsecond pulse widths for the MG90S
#define MG90S_MIN_US  500  // Pulse width for 0 degrees
#define MG90S_MAX_US  2500 // Pulse width for 180 degrees

// Define the number of servos connected
const int FIRST_CH = 0;
const int LAST_CH  = 4;

void setup() {
  Serial.begin(115200);
  delay(1000); 
  
  Wire.begin(I2C_SDA, I2C_SCL);

  pwm.begin();
  pwm.setPWMFreq(50); // MG90S expects a 50Hz update rate
  Serial.println("--- 5-Servo Simultaneous Sweep Initialized ---");
}

void loop() {
  // 1. Move Channels 0-4 to 0 Degrees together
  Serial.println("Moving all servos (0-4) to 0°...");
  for (int ch = FIRST_CH; ch <= LAST_CH; ch++) {
    pwm.writeMicroseconds(ch, MG90S_MIN_US);
  }
  delay(2000);

  // 2. Move Channels 0-4 to 90 Degrees together
  Serial.println("Moving all servos (0-4) to 90°...");
  unsigned int center_us = (MG90S_MIN_US + MG90S_MAX_US) / 2;
  for (int ch = FIRST_CH; ch <= LAST_CH; ch++) {
    pwm.writeMicroseconds(ch, center_us);
  }
  delay(2000);

  // 3. Move Channels 0-4 to 180 Degrees together
  Serial.println("Moving all servos (0-4) to 180°...");
  for (int ch = FIRST_CH; ch <= LAST_CH; ch++) {
    pwm.writeMicroseconds(ch, MG90S_MAX_US);
  }
  delay(2000);
}