#!/usr/bin/env python3
import os
import sys
import time
import traceback
import serial
from picamera2 import Picamera2
import pandas as pd
from datetime import datetime
import RPi.GPIO as GPIO
import smbus
import face_recognition
import numpy as np
import cv2

print("Starting Facial Attendance System...")

# I2C LCD Configuration
I2C_ADDR = 0x27  # LCD I2C address (0x27 or 0x3F)
I2C_BUS = 1      # I2C bus number
LCD_WIDTH = 16   # Maximum characters per line

# Define some device constants
LCD_CHR = 1  # Character mode
LCD_CMD = 0  # Command mode
LCD_LINE_1 = 0x80  # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0  # LCD RAM address for the 2nd line
LCD_BACKLIGHT = 0x08  # On
LCD_ENABLE = 0b00000100  # Enable bit

# Database Configuration
DATABASE_DIR = "database"
FACE_DIR = f"{DATABASE_DIR}/faces"
ATTENDANCE_FILE = f"{DATABASE_DIR}/attendance.csv"
USERS_FILE = f"{DATABASE_DIR}/users.csv"

class LCD:
    def __init__(self):
        self.bus = smbus.SMBus(I2C_BUS)
        self.addr = I2C_ADDR
        self.lcd_init()

    def lcd_init(self):
        self.lcd_write(0x33, LCD_CMD)  # Initialize
        self.lcd_write(0x32, LCD_CMD)  # Set to 4-bit mode
        self.lcd_write(0x06, LCD_CMD)  # Cursor move direction
        self.lcd_write(0x0C, LCD_CMD)  # Turn cursor off
        self.lcd_write(0x28, LCD_CMD)  # 2 line display
        self.lcd_write(0x01, LCD_CMD)  # Clear display
        time.sleep(0.0005)

    def lcd_write(self, bits, mode):
        bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
        bits_low = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
        self.bus.write_byte(self.addr, bits_high)
        self.lcd_toggle_enable(bits_high)
        self.bus.write_byte(self.addr, bits_low)
        self.lcd_toggle_enable(bits_low)

    def lcd_toggle_enable(self, bits):
        time.sleep(0.0005)
        self.bus.write_byte(self.addr, (bits | LCD_ENABLE))
        time.sleep(0.0005)
        self.bus.write_byte(self.addr, (bits & ~LCD_ENABLE))
        time.sleep(0.0005)

    def lcd_string(self, message, line):
        message = message.ljust(LCD_WIDTH, " ")
        self.lcd_write(line, LCD_CMD)
        for i in range(LCD_WIDTH):
            self.lcd_write(ord(message[i]), LCD_CHR)

class FacialAttendanceSystem:
    def __init__(self):
        self.camera = None
        self.lcd = LCD()
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        self.initialize_camera()

    def load_known_faces(self):
        print("Loading known faces...")
        self.lcd.lcd_string("Loading faces...", LCD_LINE_1)
        if not os.path.exists(FACE_DIR):
            os.makedirs(FACE_DIR)
        for filename in os.listdir(FACE_DIR):
            if filename.endswith('.npy'):
                try:
                    face_encoding = np.load(os.path.join(FACE_DIR, filename))
                    name = filename.split('.')[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                except Exception as e:
                    print(f"Error loading face {filename}: {str(e)}")

    def initialize_camera(self):
        self.camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (640, 480)},
            lores={"size": (320, 240)},
            display="lores"
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)  # Wait for camera to warm up

    def mark_attendance(self, student_id, name):
        try:
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H:%M:%S')
            
            # Load existing attendance
            if os.path.exists(ATTENDANCE_FILE):
                df = pd.read_csv(ATTENDANCE_FILE)
            else:
                df = pd.DataFrame(columns=['id', 'name', 'date', 'time'])
            
            # Check if already marked for today
            today_attendance = df[(df['id'] == student_id) & (df['date'] == date)]
            if not today_attendance.empty:
                self.lcd.lcd_string("Already Marked!", LCD_LINE_1)
                self.lcd.lcd_string(f"for {name}", LCD_LINE_2)
                time.sleep(2)
                return False
            
            # Add new attendance record
            new_record = pd.DataFrame({
                'id': [student_id],
                'name': [name],
                'date': [date],
                'time': [current_time]
            })
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            return True
            
        except Exception as e:
            print(f"Error marking attendance: {str(e)}")
            return False

    def run(self):
        print("Starting face recognition...")
        self.lcd.lcd_string("System Ready", LCD_LINE_1)
        self.lcd.lcd_string("Place Face", LCD_LINE_2)
        
        try:
            while True:
                frame = self.camera.capture_array()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in frame
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    self.lcd.lcd_string("Processing...", LCD_LINE_1)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                        
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = self.known_face_names[first_match_index]
                            student_id = name  # Assuming filename is student ID
                            
                            if self.mark_attendance(student_id, name):
                                self.lcd.lcd_string("Welcome", LCD_LINE_1)
                                self.lcd.lcd_string(name, LCD_LINE_2)
                                time.sleep(2)
                        else:
                            self.lcd.lcd_string("Unknown Face", LCD_LINE_1)
                            self.lcd.lcd_string("Not Registered", LCD_LINE_2)
                            time.sleep(2)
                
                self.lcd.lcd_string("System Ready", LCD_LINE_1)
                self.lcd.lcd_string("Place Face", LCD_LINE_2)
                time.sleep(0.1)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\nStopping system...")
            self.lcd.lcd_string("Shutting down...", LCD_LINE_1)
            self.lcd.lcd_string("Goodbye!", LCD_LINE_2)
            time.sleep(2)
        finally:
            if self.camera:
                self.camera.stop()
            GPIO.cleanup()

def main():
    try:
        print("\nStarting attendance system with enhanced error handling...")
        system = FacialAttendanceSystem()
        system.run()
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"ERROR: {type(e).__name__}")
        print(f"Details: {str(e)}")
        print("\nDetailed error information:")
        traceback.print_exc()
        print("="*50)
    finally:
        try:
            GPIO.cleanup()
            print("GPIO cleanup completed")
        except:
            pass

if __name__ == "__main__":
    main()
