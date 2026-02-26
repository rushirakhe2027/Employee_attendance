import cv2
import time
import pandas as pd
from datetime import datetime
import os
import signal
import sys
import socketio
import pytz

from modules.face_recognition_module import FaceRecognitionModule
from modules.lcd_module import LCDModule
from modules.buzzer_module import BuzzerModule

# Configuration
OFFICE_START_TIME = "09:30"
OFFICE_END_TIME = "17:30"
IST = pytz.timezone('Asia/Kolkata')
DATABASE_DIR = "database"
ATTENDANCE_FILE = f"{DATABASE_DIR}/attendance.csv"
EMPLOYEES_FILE = f"{DATABASE_DIR}/employees.csv"
LIVE_FRAME_PATH = f"{DATABASE_DIR}/live_frame.jpg"  # Shared with app.py

# SocketIO Client to talk to the Flask server
sio = socketio.Client()

class EmployeeAttendanceApp:
    def __init__(self):
        self.face_module = FaceRecognitionModule()
        self.lcd = LCDModule()
        self.buzzer = BuzzerModule()
        
        # Use V4L2 backend — more stable on Pi 1 B+ (avoids GStreamer memory errors)
        self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.camera.set(cv2.CAP_PROP_FPS, 10)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
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
        now = datetime.now(IST)
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


    def write_frame_to_disk(self, frame):
        """Write the current camera frame to a shared file for the web dashboard."""
        try:
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                with open(LIVE_FRAME_PATH, 'wb') as f:
                    f.write(buf.tobytes())
        except Exception:
            pass

    def run(self):
        print("Starting Employee Attendance System...")
        self.lcd.display("System Ready", "Scan Face")
        
        frame_count = 0
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                frame_count += 1
                # Only process every 5th frame to save CPU on Pi 1
                if frame_count % 5 != 0:
                    self.write_frame_to_disk(frame)
                    continue

                # Mirror frame
                frame = cv2.flip(frame, 1)
                self.write_frame_to_disk(frame)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                names = self.face_module.detect_and_recognize(rgb_frame)
                
                for name in names:

                    if name == "Unknown":
                        self.buzzer.beep_unknown()
                        self.lcd.display("Unknown! 1:Rescan", "2:Register")
                        print("\n" + "="*40)
                        print(" [!] UNKNOWN FACE DETECTED")
                        print("="*40)
                        print(" 1. Rescan (Try again)")
                        print(" 2. Register New Employee (Directly here)")
                        print("="*40)
                        
                        choice = input("Enter Choice (1/2): ").strip()
                        
                        if choice == '2':
                            self.lcd.display("Enter Name", "in Terminal...")
                            new_name = input("Enter Full Name: ").strip()
                            
                            self.lcd.display("Enter Emp ID", "in Terminal...")
                            new_id = input("Enter Employee ID: ").strip()
                            
                            self.lcd.display("Enter Dept", "in Terminal...")
                            new_dept = input("Enter Department: ").strip()
                            
                            self.lcd.display("Enter Phone", "in Terminal...")
                            new_phone = input("Enter Phone Number: ").strip()
                            
                            if new_name and new_id:
                                # Save current frame as photo
                                temp_photo = os.path.join(DATABASE_DIR, f"temp_reg.jpg")
                                cv2.imwrite(temp_photo, frame) # frame is BGR
                                
                                self.lcd.display("Processing...", "Please wait")
                                success, msg = self.face_module.register_new_face(new_name, temp_photo)
                                
                                if success:
                                    # Update CSV
                                    df = pd.read_csv(EMPLOYEES_FILE)
                                    new_emp = pd.DataFrame([[new_id, new_name, new_phone, new_dept, datetime.now(IST).strftime("%Y-%m-%d")]], 
                                                           columns=['Employee_ID', 'Name', 'Phone', 'Department', 'Join_Date'])
                                    df = pd.concat([df, new_emp], ignore_index=True)
                                    df.to_csv(EMPLOYEES_FILE, index=False)
                                    
                                    self.lcd.display("Reg Success!", new_name)
                                    print(f"Successfully registered {new_name}")
                                    time.sleep(2)
                                else:
                                    self.lcd.display("Reg Failed", "Try again")
                                    print(f"Error during registration: {msg}")
                                    time.sleep(2)
                                
                                if os.path.exists(temp_photo):
                                    os.remove(temp_photo)
                            else:
                                print("Registration cancelled (Missing details).")
                        
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
