# gaze

# Real-Time Gaze Tracking System

A real-time gaze tracking and face recognition system powered by **YOLOv8-face**, **L2CS gaze estimation**, **RANSAC-based calibration**, and **voice interaction**. Designed for intelligent surveillance, human-computer interaction, and futuristic security systems.

---

## 🚀 Project Overview

This project implements a real-time multi-person gaze tracking system with head-pose refinement, pupil detection, and voice-enabled interaction.

It combines several advanced AI components:
- Face detection using **YOLOv8-face**
- Gaze direction prediction using **L2CS**
- Polynomial & RANSAC-based screen calibration
- Real-time face recognition and gender prediction
- Voice-command support (e.g., "Where is [name]?")
- Multi-person tracking and dynamic screen visualization

---

## 🎯 Features

- 🔍 Real-time face detection and gaze direction estimation
- 🧠 Head pose estimation and gaze refinement
- 🧿 Pupil tracking for better gaze accuracy
- 🗣 Voice command: Identify and track individuals by name
- 🖥 On-screen heatmap or pointer visualization
- 🧑‍🤝‍🧑 Multi-person gaze tracking support
- 🎯 RANSAC-based robust calibration method
- 💬 Gender classification and name overlay
- 📁 Auto-registration of unknown individuals
- 🧠 Modular pipeline for easy upgrades

---

## 🛠 Tech Stack

- **Python 3.10+**
- **OpenCV**
- **YOLOv8 (Ultralytics)**
- **L2CS Gaze Estimation**
- **PyTorch**
- **NumPy, SciPy, Scikit-learn**
- **Tkinter / PyQt (for GUI)**
- **SpeechRecognition & pyttsx3 (for voice I/O)**

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/gaze-tracking-system.git
   cd gaze-tracking-system
