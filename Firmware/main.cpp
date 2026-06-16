#include <Arduino.h>

// --- Configuration Constants ---
const int NUM_ACTUATORS = 5;
const unsigned long SAMPLE_TIME_MS =
    10;                               // 100Hz low-level motion profiling loop
const float MAX_VELOCITY = 100.0;     // Max joint speed in degrees per second
const float MAX_ACCELERATION = 200.0; // Max joint acceleration in deg/s^2
const float DT = SAMPLE_TIME_MS / 1000.0;

// Hardware PWM pins for the 5 motor channels
const int PWM_PINS[NUM_ACTUATORS] = {4, 5, 6, 7, 8};

// --- Motion Profile Variables ---
float target_angles[NUM_ACTUATORS] = {0.0, 0.0, 0.0, 0.0,
                                      0.0}; // Incoming from Pi 5
float current_angles[NUM_ACTUATORS] = {0.0, 0.0, 0.0, 0.0,
                                       0.0}; // Profile tracking positions
float current_vels[NUM_ACTUATORS] = {0.0, 0.0, 0.0, 0.0,
                                     0.0}; // Profile current velocities

// RTOS Task Handles for Dual-Core Execution
TaskHandle_t CommTaskHandle = NULL;
TaskHandle_t MotionTaskHandle = NULL;

// --- Task 1: High-Speed Communication (Core 0) ---
// Handles incoming target arrays from the Raspberry Pi 5 over Serial/SPI
void CommTask(void *pvParameters) {
  Serial.begin(115200);
  while (true) {
    if (Serial.available() > 0) {
      // Expecting a simple comma-separated string string from Pi 5:
      // "th1,th2,th3,th4,th5\n"
      String incoming = Serial.readStringUntil('\n');
      int ext_index = 0;

      // Parse string values into target_angles array safely
      for (int i = 0; i < NUM_ACTUATORS; i++) {
        int next_comma = incoming.indexOf(',', ext_index);
        if (next_comma != -1) {
          target_angles[i] =
              incoming.substring(ext_index, next_comma).toFloat();
          ext_index = next_comma + 1;
        } else {
          target_angles[i] = incoming.substring(ext_index).toFloat();
          break;
        }
      }
    }
    vTaskDelay(pdMS_TO_TICK(5)); // Run communication parser at 200Hz
  }
}

// --- Task 2: Deterministic Motion Profiling & PWM (Core 1) ---
// Calculates trapezoidal profiles and updates hardware outputs
void MotionTask(void *pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();

  while (true) {
    for (int i = 0; i < NUM_ACTUATORS; i++) {
      float error = target_angles[i] - current_angles[i];

      if (abs(error) > 0.01) {
        // 1. Calculate ideal velocity based on remaining distance (stopping
        // distance constraint) v^2 = 2 * a * d  -> max_safe_vel = sqrt(2 *
        // acceleration * distance)
        float max_safe_vel = sqrt(2.0 * MAX_ACCELERATION * abs(error));

        // Target velocity cap
        float target_vel = (error > 0) ? MAX_VELOCITY : -MAX_VELOCITY;
        if (abs(target_vel) > max_safe_vel) {
          target_vel = (error > 0) ? max_safe_vel : -max_safe_vel;
        }

        // 2. Accelerate / Decelerate current velocity toward target velocity
        float vel_error = target_vel - current_vels[i];
        float max_vel_change = MAX_ACCELERATION * DT;

        if (abs(vel_error) > max_vel_change) {
          current_vels[i] += (vel_error > 0) ? max_vel_change : -max_vel_change;
        } else {
          current_vels[i] = target_vel;
        }

        // 3. Integrate velocity to update the tracking position
        current_angles[i] += current_vels[i] * DT;
      } else {
        current_vels[i] = 0.0;
        current_angles[i] = target_angles[i];
      }

      // 4. Map the joint angle to direct hardware PWM duty cycle commands
      // Example mapping: 0 to 180 degrees mapped to standard 500us - 2500us
      // servo pulses
      int duty = map(current_angles[i], 0, 180, 1638,
                     8192); // 16-bit timer values
#if defined(ESP32)
                            // Use ledcWrite or your custom low-level timer
                            // abstraction ledcWrite(i, duty);
#endif
    }

    // Ensure deterministic 100Hz execution rate
    xTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICK(SAMPLE_TIME_MS));
  }
}

void setup() {
  // Pin setup and PWM channel configuration would go here

  // Distribute computations across the ESP32-S3 XTensa cores
  xTaskCreatePinnedToCore(CommTask, "CommTask", 4096, NULL, 2, &CommTaskHandle,
                          0);
  xTaskCreatePinnedToCore(MotionTask, "MotionTask", 4096, NULL, 3,
                          &MotionTaskHandle, 1);
}

void loop() {
  // Empty! Dual cores are handled entirely via freeRTOS tasks above.
}