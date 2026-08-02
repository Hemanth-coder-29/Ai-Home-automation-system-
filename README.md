# AI-Powered Voice + Gesture Smart Home Control

An AI-based Smart Home Automation System that allows users to control home appliances using Hand Gestures, Voice Commands, and a Bluetooth Mobile Application without requiring an internet connection.

This project is being developed as a Final Year B.E. Artificial Intelligence & Machine Learning project.

---

## Project Overview

The objective of this project is to develop an intelligent offline home automation system that provides multiple methods of controlling electrical appliances.

The system combines Computer Vision, Machine Learning, Voice Recognition, Bluetooth Communication, and Embedded Systems to provide a flexible and user-friendly smart home solution.

Unlike cloud-based smart home systems, this project works completely offline, making it faster, more secure, and independent of internet connectivity.

---

## Features

- Hand Gesture Recognition using AI
- Offline Voice Commands
- Bluetooth Android Application
- Arduino-based Appliance Control
- Relay Module Integration
- Multiple Appliance Support
- Offline Processing
- Easy-to-use Interface

---

## Technologies Used

### Programming

- Python 3.12
- Arduino C++

### Computer Vision

- OpenCV
- MediaPipe

### Machine Learning

- TensorFlow
- Keras
- Scikit-Learn

### Communication

- PySerial
- HC-05 Bluetooth Module

### Voice Recognition

- SpeechRecognition

### Mobile App

- MIT App Inventor

---

## Hardware Components

- Arduino UNO
- USB Webcam
- USB Microphone
- HC-05 Bluetooth Module
- 4-Channel Relay Module
- 16x2 LCD Display
- Jumper Wires
- Power Supply

---

## Software Requirements

- Python 3.12
- Arduino IDE
- VS Code
- MIT App Inventor

---

## Project Architecture

User
│
├── Hand Gestures
├── Voice Commands
└── Mobile App
        │
        ▼
Python Controller
        │
        ▼
OpenCV + MediaPipe + TensorFlow
        │
        ▼
Serial Communication
        │
        ▼
Arduino UNO
        │
        ▼
Relay Module
        │
        ▼
Home Appliances

---

## Project Structure

```
SmartHomeAI/

├── arduino/
├── bluetooth/
├── dataset/
├── docs/
├── gesture/
├── images/
├── model/
├── voice/

├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Development Roadmap

- [x] Module 1 – Project Structure
- [x] Module 2 – Development Environment
- [ ] Module 3 – OpenCV Camera
- [ ] Module 4 – MediaPipe Hand Detection
- [ ] Module 5 – Gesture Dataset Collection
- [ ] Module 6 – Gesture Recognition Model
- [ ] Module 7 – Arduino Communication
- [ ] Module 8 – Voice Recognition
- [ ] Module 9 – Bluetooth Mobile App
- [ ] Module 10 – Final Integration

---

## Installation

Clone the repository

```bash
git clone https://github.com/Hemanth-coder-29/Ai-Home-automation-system-.git
```

Go to the project folder

```bash
cd Ai-Home-automation-system-
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

## Current Status

Project is currently under active development.

Latest Completed Module:

Development Environment Setup

---

## Future Improvements

- Face Recognition Authentication
- IoT Integration
- Mobile Notifications
- Energy Consumption Monitoring
- AI-based Activity Prediction
- Smart Scheduling
- MQTT Integration
- Home Assistant Integration

---

## References

- IEEE ICCCT 2025
- IEEE ICIPCN 2024

---

## Author

Hemanth Raghava

Bachelor of Engineering

Artificial Intelligence & Machine Learning

SEA College of Engineering & Technology

Bengaluru, India

---

## License

This project is developed for educational and research purposes.
