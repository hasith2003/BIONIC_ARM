#define LED_PIN 2 // Most ESP32 dev boards have a built-in LED on pin 2

void setup() {
  Serial.begin(115200); // Initialize serial communication at 115200 baud
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  // Check if the Raspberry Pi has sent data
  if (Serial.available() > 0) {
    String incomingData = Serial.readStringUntil('\n');
    incomingData.trim(); // Clean up any trailing whitespace/newlines

    if (incomingData == "PING") {
      // Visual confirmation on the ESP32 board
      digitalWrite(LED_PIN, HIGH);
      delay(200);
      digitalWrite(LED_PIN, LOW);

      // Send response payload back to the Pi
      // Using JSON format makes parsing on the Pi incredibly simple later
      Serial.println("{\"status\": \"PONG\", \"sensor_val\": 42}");
    }
  }
}