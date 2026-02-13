# EmployeeSync AI: Facial Attendance & Late Monitoring System

A professional Raspberry Pi-based employee attendance system featuring AI face recognition, a real-time web dashboard, and automated late arrival monitoring.

## 🌟 Key Features

- **AI Face Recognition**: High-accuracy detection using Dlib (HOG) and 128-dimensional facial encodings.
- **Robust Registration**: Advanced feature extraction with 68-point landmark alignment and "jitter" data augmentation.
- **Web Dashboard**: Modern, light-themed Flask dashboard with Socket.IO real-time updates.
- **Late Monitoring**: Automatically flags late arrivals based on customizable office start times.
- **Hardware Feedback**: 16x2 LCD display & localized buzzer patterns (Present: 1 Beep, Late: 2 Beeps, Unknown: Long Beep).
- **Simulation Mode**: Built-in support to run on non-Raspberry Pi devices (Windows/PC) for testing purposes.

## 🛠️ Hardware Requirements

- **Raspberry Pi** (4B recommended, but supports others)
- **Camera Module** (Pi Camera or Standard USB Webcam)
- **16x2 LCD Display** (I2C Interface)
- **Active Buzzer** (Connected to GPIO 18)

## 📁 Project Structure

```text
├── app.py                # Flask Web Dashboard & Socket.IO Server
├── face_app.py           # Real-time Face Recognition Engine
├── modules/
│   ├── face_recognition_module.py  # Core AI Brain (Alignment/Extraction)
│   ├── lcd_module.py               # Hardware LCD Controller
│   └── buzzer_module.py            # Hardware Buzzer patterns
├── database/
│   ├── employees.csv     # Employee Directory
│   ├── attendance.csv    # Master Attendance Logs
│   ├── faces/            # Aligned employee photos
│   └── encodings/        # Pre-computed AI facial data (.npy)
├── templates/            # Web UI Layouts
└── static/               # Style & Asset Files
```

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configure Your Office Time

Open `face_app.py` and set your `OFFICE_START_TIME`:

```python
OFFICE_START_TIME = "09:30"
```

### 3. Launch the Dashboard

```bash
python app.py
```

Access the UI at: `http://localhost:5000`

### 4. Launch the AI Engine

```bash
python face_app.py
```

## 📝 Usage Notes

- **Registration**: Add employees via the "Directory" tab on the web dashboard. Uploading a photo will automatically trigger the AI alignment and feature extraction process.
- **Real-time Updates**: The dashboard uses WebSockets to show "Toasts" instantly when someone is recognized at the door.
- **Deployment**: For permanent use, it is recommended to run both scripts as `systemd` services on your Raspberry Pi.

---

_Refactored with ❤️ for a professional workplace environment._
