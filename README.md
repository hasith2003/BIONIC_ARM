# High-Fidelity Bionic Arm for Transradial Amputees

A distributed, low-latency bionic prosthesis designed specifically for transradial amputees with a 5–10 cm residual forearm. This system leverages a high-performance heterogeneous architecture, splitting intensive digital signal processing (DSP), machine learning intent recognition, and geometric Inverse Kinematics (IK) from real-time, low-level motor actuation and trajectory profiling.

---

## 🚀 System Architecture

The control pipeline is split across two core processing units over a high-speed local serial/SPI link to balance computational power with strict real-time execution safety.

### 1. Central Intelligence Unit (Raspberry Pi 5)
* **Role:** High-level multi-channel signal processing and intent recognition.
* **Tasks:**
    * Ingests raw surface Electromyography (sEMG) data streams.
    * Executes real-time feature extraction (Time Domain, Frequency Domain features).
    * Runs a lightweight, hardware-optimized Deep Learning inference pipeline to predict **10 different types of hand gestures**.
    * Implements a Subject-Dependent (SD) classification model architecture (`SD_CNN_LSTM_ATT.ipynb`) consisting of a Convolutional Neural Network (CNN) followed by a Bidirectional Long Short-Term Memory (BiLSTM) network with an Attention mechanism, currently attaining an inference accuracy of **over 46%**.
    * Serves as the development baseline for ongoing optimization, with the ultimate objective of scaling the current pipeline to achieve a robust, generalized **Subject-Independent (SI) classification model**.
    * Computes geometric Inverse Kinematics (IK) algorithms, converting spatial coordinate targets into precise target joint angle arrays.

### 2. Low-Level Actuation Node (ESP32-S3)
* **Role:** Real-time hardware abstraction, safety monitoring, and motor control.
* **Tasks:**
    * Directly interfaces with multi-channel sEMG analog front-ends (AFEs) via high-speed ADC sampling.
    * Receives target joint angles and gesture commands from the Raspberry Pi 5.
    * Generates real-time smooth motion control profiles (e.g., trapezoidal or S-curve velocity profiling) to interpolate transitions between target positions seamlessly.
    * Generates precise multi-channel PWM outputs for multi-DOF finger and thumb actuators.
    * Monitors continuous feedback loops (current sensing, thermal thresholds, and limit switches) to ensure mechanical and user safety.
