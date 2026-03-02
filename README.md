# 🌐 IT SOLUTIONS Pvt Ltd — Advanced IoT Facial Attendance System

This project is a high-performance, professional-grade **IoT Facial Attendance System** designed specifically for the Raspberry Pi environment. It combines real-time facial recognition, physical hardware feedback (LCD & Buzzer), and a modern Web Dashboard for management.

---

## 📑 Project Overview

This system provides an end-to-end solution for corporate or educational attendance tracking.

- **Facial Recognition**: Uses Dlib's Deep Learning model (128-point encoding) for human-level accuracy.
- **Dual Interface**: Operates via a Physical Console (Pi + Camera) and a Remote Web Dashboard (PC/Mobile).
- **Smart Business Logic**: Automatically handles "On-Time," "Late Arrivals," and "Early Leavings" based on custom time rules (9:30 AM / 5:30 PM).

---

## 🛠️ Hardware Requirements

1.  **Raspberry Pi** (Model 1 B+, 3, 4, or Zero 2W)
2.  **Pi Camera Module** or USB Webcam
3.  **I2C LCD Display (16x2)** (for status messages)
4.  **Active Buzzer** (for audio feedback)
5.  **Jumper Wires & Power Supply**

---

## ⚡ Initial Setup Guide (For New Users)

### 1. Network Connection (Hotspot)

The Raspberry Pi is pre-configured to connect to the following network for professional deployment:

- **WiFi Name (SSID):** `admin`
- **WiFi Password:** `123456789`

### 2. Connecting via PuTTY (Windows)

To control the system from your laptop:

1.  Download and install **PuTTY** from [putty.org](https://www.putty.org/).
2.  Connect your laptop to the same `admin` WiFi.
3.  Open PuTTY.
4.  In **Host Name**, enter: `admin2027` (or the IP address shown on the LCD).
5.  Click **Open**.
6.  **Username:** `admin`
7.  **Password:** `123456789`

---

## 🚀 How to Run the Project

Follow these commands exactly to start the system:

### Step 1: Navigate to Project Folder

```bash
cd ~/Employee_attendance
```

### Step 2: Update to Latest Version

```bash
git fetch origin
git reset --hard origin/main
```

### Step 3: Start the Combined System

This single command starts both the **Camera System** and the **Web Dashboard**:

```bash
python3 run_system.py
```

---

## 📋 Understanding the Logic (Shift Rules)

The system is programmed with "Smart Shift" logic (Kolkata IST):

- **Check-IN (Morning)**:
  - **9:00 AM - 9:30 AM**: Recorded as `On-Time`.
  - **9:31 AM - 12:00 PM**: Recorded as `Late Arrived`.
- **Check-OUT (Afternoon/Evening)**:
  - **12:01 PM - 5:30 PM**: Recorded as `Early Leaving`.
  - **After 5:30 PM**: Recorded as `Left` (Normal Checkout).
- **Anti-Spam**: Once you Check-OUT, the system will say **"Try Tomorrow!"** to prevent double marking.

---

## 💻 Web Dashboard Guide

Once the system starts, it will show a URL (e.g., `http://192.168.x.x:5000`). Open this in any browser:

1.  **Main Dashboard**: See total employees, today's presence, and late counts.
2.  **Employee Directory**: View all registered staff and their "Face Data" status.
3.  **Full Logs**: Filter attendance by date and download the data as an **Excel/CSV** file.
4.  **Erase Data**: A single click can delete an employee and all their associated history files.

---

## 🔊 Signal Meanings (Buzzer & LCD)

- **1 Long Beep (▬)**: Success! Welcome or Good-bye.
- **3 Short Beeps (▪ ▪ ▪)**: Warning! Late arrival or Early leaving recorded.
- **5 Rapid Beeps (▪▪▪▪▪)**: Rejected! Already scanned or unknown face.

---

## 🔧 Troubleshooting

- **Camera not starting?** Ensure the ribbon cable is tight and `libcamerify` is installed.
- **Face not detected?** Stand 1-2 feet from the camera with good lighting on your face.
- **LCD Blank?** Check the I2C wires (SDA/SCL) and adjust the contrast screw on the back of the LCD module.

---

**Developed by:** IT SOLUTIONS Pvt Ltd
**Version:** 2.1.0 (Advanced AI Edition)
