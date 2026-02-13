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
OFFICE_END_TIME = "17:30"
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
        self.last_record = {} # name -> (last_type, last_date)
        
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
        
        start_time = datetime.strptime(OFFICE_START_TIME, "%H:%M").time()
        end_time = datetime.strptime(OFFICE_END_TIME, "%H:%M").time()
        
        # Simple Logic: 
        # Before 1:00 PM is CHECK-IN
        # After 1:00 PM is CHECK-OUT
        record_type = "IN" if now.hour < 13 else "OUT"
        
        # Prevent spam (only allow one IN and one OUT per day)
        if name in self.last_record:
            l_type, l_date = self.last_record[name]
            if l_date == current_date and l_type == record_type:
                return False

        if record_type == "IN":
            is_late = now.time() > start_time
            status = "Late Arrival" if is_late else "On-Time"
        else:
            is_early = now.time() < end_time
            status = "Early Leaving" if is_early else "Left"
        
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
        print(f"[{record_type}] Marked: {name} as {status} at {current_time}")
        self.lcd.display(f"{name}: {record_type}", f"{status}")
        
        if status == "Late Arrival":
            self.buzzer.beep_late()
        elif status == "Early Leaving":
            self.buzzer.beep_late() # Same double beep for early leaving warning
        else:
            self.buzzer.beep_present()
            
        self.last_record[name] = (record_type, current_date)
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
