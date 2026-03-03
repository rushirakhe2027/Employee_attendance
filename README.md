# Employee Attendance and Late Arrival Alert System Using Facial Recognition

This project is a high-performance, professional-grade **IoT Facial Attendance System** designed specifically for the Raspberry Pi. This guide covers **every step** from hardware wiring to software deployment.

---

## 🏗️ Phase 1: Hardware & Power Connections

Before turning on the software, you must wire the components correctly.

### 1. Powering the System

- **Main Power**: Use a **5V 3A Micro-USB (or USB-C)** power adapter for the Raspberry Pi.
- **Voltage Rails**:
  - Connect the **VCC** of the LCD and Buzzer to the **5V Pin** (Pin 2 or 4) on the Raspberry Pi.
  - Connect the **GND** of all components to a **Ground Pin** (e.g., Pin 6) on the Pi.

### 2. Component Wiring (GPIO)

| Component   | Pin Name     | Raspberry Pi Pin        |
| :---------- | :----------- | :---------------------- |
| **I2C LCD** | VCC          | 5V (Pin 2)              |
| **I2C LCD** | GND          | Ground (Pin 6)          |
| **I2C LCD** | SDA          | GPIO 2 (Pin 3)          |
| **I2C LCD** | SCL          | GPIO 3 (Pin 5)          |
| **Buzzer**  | Positive (+) | GPIO 18 (Pin 12)        |
| **Buzzer**  | Negative (-) | Ground (Pin 14)         |
| **Camera**  | Ribbon/USB   | Camera Port or USB Port |

---

## 📶 Phase 2: Network & Remote Access

The Pi is designed to be headless (no monitor needed).

### 1. Connect to WiFi

The Pi will automatically search for this network:

- **WiFi SSID (Name):** `admin`
- **WiFi Password:** `123456789`

### 2. How to Download & Install PuTTY (Windows)

If you don't have PuTTY on your laptop, follow these steps:

1.  Go to: [www.putty.org](https://www.putty.org/)
2.  Click the link that says **"You can download PuTTY here"**.
3.  Under **"Package files"**, look for the **64-bit x86** installer (`.msi` file).
4.  Download it and run the installer.
5.  Click **Next** through the setup until it is installed.

### 3. Remote Control (PuTTY)

1.  Connect your laptop to the `admin` WiFi.
2.  Open **PuTTY**.
3.  **Host Name**: `admin2027` (If this fails, use the IP address shown on the LCD).
4.  **Login User**: `admin`
5.  **Password**: `123456789`

---

## 🚀 Phase 3: Software Execution

### Step 1: Enter Project Directory

```bash
cd ~/Employee_attendance
```

### Step 2: Sync Latest Code

```bash
git fetch origin
git reset --hard origin/main
```

### Step 3: Start the System

```bash
python3 run_system.py
```

_Wait 5 seconds. The LCD will show the Welcome message and your IP address._

---

## 🧠 Phase 4: How the System Works (Full Logic)

### 1. Facial recognition AI

The system uses a **Hybrid AI Strategy**:

- **Haar Cascade**: Scans 30 frames per second to find "shapes" of faces (low power).
- **Dlib Deep Learning**: Once a face is stable, it extracts a **128-point neural map**. This is unique like a fingerprint.
- **Matching**: It compares your live map against the `.npy` files in `database/faces/`.

### 2. Shift Timing & Rules (Kolkata IST)

- **Morning Shift (Check-IN)**:
  - **Before 9:30 AM**: Recorded accurately as **"On-Time"** (1 Long Beep confirmation).
  - **9:30 AM onwards**: Recorded as **"Late Arrived"** (3 Rapid alert beeps triggered).
  - **Automatic Entry**: If an employee scans for the first time in the afternoon, the system automatically records their _Late IN_ at 9:31 AM so their daily record is never incomplete.
- **Evening Shift (Check-OUT)**:
  - **Before 5:30 PM**: Recorded as **"Early Leaving"** (Authorized but noted).
  - **From 5:30 PM onwards**: Recorded as **"Left"** (Normal checkout).
- **Anti-Spam**: Once a person has checked OUT, the system will prevent triple-scanning and say: **"OUT Already Done" / "Try Tomorrow!"**

---

## 📊 Phase 5: The Web UI (Dashboard)

Access the dashboard by typing the Pi's IP into your laptop browser (e.g., `http://192.168.1.15:5000`).

- **Dashboard**: Real-time stats of how many workers are in/late.
- **Employee Directory**: Manage staff and verify if their Face ID is registered.
- **Full Logs**: Look at history for any specific date.
- **Data Export**: Click "Download" to get an Excel-ready CSV of all records.
- **Erase**: Completely delete an employee and all their logs from the combined database.

---

## 🔊 Signal Reference

- **LCD: "Scanning..."**: AI is currently analyzing your face.
- **Buzzer: 1 Long**: Successful Attendance.
- **Buzzer: 3 Short**: Late or Early scan recorded.
- **Buzzer: 5 Rapid**: Error or Already Scanned.

---
