#!/usr/bin/env python3
"""
TEST SCRIPT — Reference implementation for comparison.
Run with: libcamerify python3 test_script.py

NOTE: This script requires .npy face encodings to already exist in database/faces/
      Register faces using the main system (face_app.py) first, then test here.

IMPORTANT: This will be SLOW on Pi 1 B+ because face_recognition.face_locations()
           runs a full HOG scan on every frame. This is for comparison purposes only.
"""
import os
import time
import traceback
import smbus
import face_recognition
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

print("Starting Test Facial Attendance System...")

# === LCD Configuration ===
I2C_ADDR = 0x27
I2C_BUS = 1
LCD_WIDTH = 16
LCD_CHR = 1
LCD_CMD = 0
LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0
LCD_BACKLIGHT = 0x08
LCD_ENABLE = 0b00000100

# === Database Configuration ===
DATABASE_DIR = "database"
FACE_DIR = f"{DATABASE_DIR}/faces"
ATTENDANCE_FILE = f"{DATABASE_DIR}/attendance.csv"


class LCD:
    def __init__(self):
        try:
            self.bus = smbus.SMBus(I2C_BUS)
            self.addr = I2C_ADDR
            self.lcd_init()
            print("LCD Initialized")
        except Exception as e:
            print(f"LCD Error (running without LCD): {e}")
            self.bus = None

    def lcd_init(self):
        if not self.bus: return
        self.lcd_write(0x33, LCD_CMD)
        self.lcd_write(0x32, LCD_CMD)
        self.lcd_write(0x06, LCD_CMD)
        self.lcd_write(0x0C, LCD_CMD)
        self.lcd_write(0x28, LCD_CMD)
        self.lcd_write(0x01, LCD_CMD)
        time.sleep(0.0005)

    def lcd_write(self, bits, mode):
        if not self.bus: return
        bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
        bits_low = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
        self.bus.write_byte(self.addr, bits_high)
        self.lcd_toggle_enable(bits_high)
        self.bus.write_byte(self.addr, bits_low)
        self.lcd_toggle_enable(bits_low)

    def lcd_toggle_enable(self, bits):
        if not self.bus: return
        time.sleep(0.0005)
        self.bus.write_byte(self.addr, (bits | LCD_ENABLE))
        time.sleep(0.0005)
        self.bus.write_byte(self.addr, (bits & ~LCD_ENABLE))
        time.sleep(0.0005)

    def lcd_string(self, message, line):
        if not self.bus: return
        message = message.ljust(LCD_WIDTH, " ")
        self.lcd_write(line, LCD_CMD)
        for i in range(LCD_WIDTH):
            self.lcd_write(ord(message[i]), LCD_CHR)


class TestAttendanceSystem:
    def __init__(self):
        self.camera = None
        self.lcd = LCD()
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        self.initialize_camera()

    def load_known_faces(self):
        print("Loading known faces from .npy files...")
        self.lcd.lcd_string("Loading faces...", LCD_LINE_1)
        os.makedirs(FACE_DIR, exist_ok=True)

        loaded = 0
        for filename in os.listdir(FACE_DIR):
            if filename.endswith('.npy'):
                try:
                    encoding = np.load(os.path.join(FACE_DIR, filename))
                    name = os.path.splitext(filename)[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    loaded += 1
                    print(f"  Loaded: {name}")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")

        print(f"Total loaded: {loaded} face(s)")
        if loaded == 0:
            print("WARNING: No .npy files found. Register faces via face_app.py first!")

    def initialize_camera(self):
        # Use V4L2 backend — more stable on Pi 1 B+ than libcamera
        self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.camera.set(cv2.CAP_PROP_FPS, 5)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(1)
        print("Camera initialized")

    def mark_attendance(self, name):
        try:
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H:%M:%S')

            if os.path.exists(ATTENDANCE_FILE):
                df = pd.read_csv(ATTENDANCE_FILE)
            else:
                df = pd.DataFrame(columns=['name', 'date', 'time'])

            today = df[(df['name'] == name) & (df['date'] == date)]
            if not today.empty:
                print(f"  Already marked for {name} today.")
                return False

            new_row = pd.DataFrame([[name, date, current_time]], columns=['name', 'date', 'time'])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            print(f"  [MARKED] {name} at {current_time}")
            return True
        except Exception as e:
            print(f"  Attendance error: {e}")
            return False

    def run(self):
        print("\nTest system running. Press Ctrl+C to stop.")
        print("NOTE: face_recognition.face_locations() is slow on Pi 1. Expect 5-15s per frame.\n")
        self.lcd.lcd_string("IT SOLUTIONS Pvt", LCD_LINE_1)
        self.lcd.lcd_string("Scan Face", LCD_LINE_2)

        frame_count = 0
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                frame_count += 1
                frame = cv2.flip(frame, 1)

                # Only run dlib face_locations every 10th frame
                if frame_count % 10 != 0:
                    continue

                print(f"[Frame {frame_count}] Scanning for faces...")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # HOG-based face detection (accurate but slow on Pi 1)
                t_start = time.time()
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                t_detect = time.time() - t_start
                print(f"  Detection took: {t_detect:.2f}s | Found: {len(face_locations)} face(s)")

                if not face_locations:
                    continue

                self.lcd.lcd_string("IT SOLUTIONS Pvt", LCD_LINE_1)
                self.lcd.lcd_string("Verifying...", LCD_LINE_2)

                # Encode detected faces
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="small")

                for face_encoding in face_encodings:
                    if not self.known_face_encodings:
                        print("  No registered faces to compare against.")
                        self.lcd.lcd_string("IT SOLUTIONS Pvt", LCD_LINE_1)
                        self.lcd.lcd_string("Unknown Face", LCD_LINE_2)
                        continue

                    distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_idx = int(np.argmin(distances))
                    best_dist = distances[best_idx]
                    print(f"  Best match: {self.known_face_names[best_idx]} | Distance: {best_dist:.3f}")

                    # Tolerance 0.45 for this test script
                    if best_dist <= 0.45:
                        name = self.known_face_names[best_idx]
                        print(f"  RECOGNIZED: {name}")
                        self.lcd.lcd_string("IT SOLUTIONS Pvt", LCD_LINE_1)
                        self.lcd.lcd_string(f"Hello {name}", LCD_LINE_2)
                        self.mark_attendance(name)
                        time.sleep(2)
                    else:
                        print(f"  UNKNOWN (distance too high: {best_dist:.3f})")
                        self.lcd.lcd_string("IT SOLUTIONS Pvt", LCD_LINE_1)
                        self.lcd.lcd_string("Unknown Face", LCD_LINE_2)
                        time.sleep(1)

                self.lcd.lcd_string("IT SOLUTIONS Pvt", LCD_LINE_1)
                self.lcd.lcd_string("Scan Face", LCD_LINE_2)
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nTest stopped.")
            self.lcd.lcd_string("Shutting down...", LCD_LINE_1)
            self.lcd.lcd_string("Goodbye!", LCD_LINE_2)
            time.sleep(1)
        finally:
            if self.camera:
                self.camera.release()


def main():
    try:
        system = TestAttendanceSystem()
        system.run()
    except Exception as e:
        print("\n" + "="*50)
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        print("="*50)


if __name__ == "__main__":
    main()
