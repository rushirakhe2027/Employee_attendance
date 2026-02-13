import cv2
import time
import pandas as pd
from datetime import datetime
import os
import signal
import sys
import socketio

from modules.face_recognition_module import FaceRecognitionModule
from modules.lcd_module import LCDModule
from modules.buzzer_module import BuzzerModule

# Configuration
OFFICE_START_TIME = "09:30"
DATABASE_DIR = "database"
ATTENDANCE_FILE = f"{DATABASE_DIR}/attendance.csv"
EMPLOYEES_FILE = f"{DATABASE_DIR}/employees.csv"

# SocketIO Client to talk to the Flask server
sio = socketio.Client()

class EmployeeAttendanceApp:
    def __init__(self):
        self.face_module = FaceRecognitionModule()
        self.lcd = LCDModule()
        self.buzzer = BuzzerModule()
        
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        self.last_seen = {} 
        
        self.connect_to_server()

    def connect_to_server(self):
        try:
            sio.connect('http://localhost:5000')
            print("Connected to Dashboard Server via Socket.IO")
        except Exception as e:
            print(f"Server connection failed (real-time updates disabled): {e}")

    def mark_attendance(self, name):
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        
        # Avoid duplicate marking within same day (simple logic)
        if name in self.last_seen:
            _, last_date = self.last_seen[name]
            if last_date == current_date:
                return False
        
        # Determine Status
        start_time = datetime.strptime(OFFICE_START_TIME, "%H:%M").time()
        is_late = now.time() > start_time
        status = "Late" if is_late else "Present"
        
        # Get Employee ID
        try:
            df_emp = pd.read_csv(EMPLOYEES_FILE)
            emp_id = str(df_emp[df_emp['Name'] == name]['Employee_ID'].values[0])
        except:
            emp_id = "Unknown"

        # Save to CSV
        data_row = [emp_id, name, current_date, current_time, status]
        df_new = pd.DataFrame([data_row], columns=['Employee_ID', 'Name', 'Date', 'Time', 'Status'])
        df_new.to_csv(ATTENDANCE_FILE, mode='a', header=not os.path.exists(ATTENDANCE_FILE), index=False)
        
        # Emit Real-time event to Web Dashboard
        if sio.connected:
            sio.emit('recognition_event', {
                'emp_id': emp_id,
                'name': name,
                'time': current_time,
                'status': status
            })

        # Feedback
        print(f"Attendance Marked: {name} ({status}) at {current_time}")
        self.lcd.display(f"Welcome {name}", f"Status: {status}")
        
        if is_late:
            self.buzzer.beep_late()
        else:
            self.buzzer.beep_present()
            
        self.last_seen[name] = (now, current_date)
        return True

    def run(self):
        print("Starting Employee Attendance System...")
        self.lcd.display("System Ready", "Scan Face")
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Update global frame in server if they are in same memory space 
                # (Since they are separate processes, we'll use a trick or just simple camera sharing if on same machine)
                # For this demonstration, we'll assume they might run together or we stream via socket
                # Actually, I'll update app.py to expose a frame updating method if they are imported.
                # But here we'll just run them and use individual camera access if available, 
                # or better: we'll have app.py read the camera if we are on a single-user demo.
                # For advanced: WebSockets could stream frames, but that's heavy.
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                names = self.face_module.detect_and_recognize(rgb_frame)
                
                for name in names:
                    if name == "Unknown":
                        self.lcd.display("Unknown Face", "Access Denied")
                        self.buzzer.beep_unknown()
                        time.sleep(1)
                        self.lcd.display("System Ready", "Scan Face")
                    else:
                        if self.mark_attendance(name):
                            time.sleep(2)
                            self.lcd.display("System Ready", "Scan Face")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        self.running = False
        self.camera.release()
        self.lcd.clear()
        self.buzzer.cleanup()
        if sio.connected:
            sio.disconnect()
        print("System Shutdown.")

if __name__ == "__main__":
    app = EmployeeAttendanceApp()
    app.run()
