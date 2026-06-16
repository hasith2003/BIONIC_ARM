# High-Fidelity Bionic Arm for Transradial Amputees

A distributed, low-latency bionic prosthesis designed specifically for transradial amputees with a 5–10 cm residual forearm. This system leverages a high-performance heterogeneous architecture, splitting intensive digital signal processing (DSP) and machine learning from real-time, low-level motor actuation.

---

## 🚀 System Architecture

The control pipeline is split across two core processing units over a high-speed local serial/SPI link to balance computational power with strict real-time execution safety.

### 1. Central Intelligence Unit (Raspberry Pi 5)
* **Role:** High-level multi-channel signal processing and intent recognition.
* **Tasks:**
    * Ingests raw surface Electromyography (sEMG) data streams.
    * Executes real-time feature extraction (Time Domain, Frequency Domain features).
    * Runs a lightweight, hardware-optimized Deep Learning inference pipeline (CNN/LSTM) to classify and predict multi-gesture user intent.

### 2. Low-Level Actuation Node (ESP32-S3)
* **Role:** Real-time hardware abstraction, safety monitoring, and motor control.
* **Tasks:**
    * Directly interfaces with multi-channel sEMG analog front-ends (AFEs) via high-speed ADC sampling.
    * Receives categorized gesture commands from the Raspberry Pi 5.
    * Generates precise multi-channel PWM outputs for multi-DOF finger and thumb actuators.
    * Monitors continuous feedback loops (current sensing, thermal thresholds, and limit switches) to ensure mechanical and user safety.
