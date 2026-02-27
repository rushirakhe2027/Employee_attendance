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
        self.camera.set(cv2.CAP_PROP_FPS, 5) # Lower FPS reduces the "packet queue" crash on Pi 1
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.running = True
        self.last_record = {} # name -> (last_type, last_date)
        
        # Proper Recognition Buffer
        # 5 consistent votes required for "Professional" accuracy
        self.recognition_buffer = {} # name -> count
        self.buffer_threshold = 5
        
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

        # Advanced Greeting Logic
        current_hour = now.hour
        if current_hour < 12:
            greeting = "Good Morning!"
        elif current_hour < 17:
            greeting = "Good Afternoon!"
        else:
            greeting = "Good Evening!"
            
        print(f"[{record_type}] Marked: {name} as {status} at {current_time}")
        self.lcd.display("IT SOLUTIONS Pvt", f"Hello {name}")
        time.sleep(1.5)
        self.lcd.display(f"{record_type} Success", status)
        
        if status == "Late Arrival":
            self.buzzer.beep_late()
        elif status == "Early Leaving":
            self.buzzer.beep_late() 
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
        self.lcd.display("IT SOLUTIONS Pvt", "Scan Face")
        
        frame_count = 0
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Mirror frame immediately
                frame = cv2.flip(frame, 1)
                
                # --- LIVE STREAM OPTIMIZATION ---
                # Only write to disk every 3 frames to prevent SD Card/IO lag
                if frame_count % 3 == 0:
                    small_stream_frame = cv2.resize(frame, (320, 240))
                    self.write_frame_to_disk(small_stream_frame)

                # Professional Accuracy: Process every 5th frame for 160x160 detail
                if frame_count % 5 != 0:
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check for faces using the fast Haar Cascade first
                gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                faces = self.face_module.haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                
                if len(faces) > 0:
                    # self.lcd.display("Face Detected", "Verifying...") # Optional: Hide this to make it feel smoother
                    names = self.face_module.detect_and_recognize(rgb_frame)
                else:
                    # Clear buffer if no face seen
                    self.recognition_buffer = {}
                    continue
                
                for name in names:
                    if name == "Unknown":
                        # ... already handled below ...
                        pass
                    else:
                        # VOTE for this person
                        self.recognition_buffer[name] = self.recognition_buffer.get(name, 0) + 1
                        
                        if self.recognition_buffer[name] >= self.buffer_threshold:
                            # 5 Consistent Frames FOUND!
                            if self.mark_attendance(name):
                                self.recognition_buffer = {} # Reset
                                time.sleep(2)
                                self.lcd.display("IT SOLUTIONS Pvt", "Scan Face")
                            continue
                        else:
                            # Still analyzing
                            print(f" [BUFFER] {name} identified {self.recognition_buffer[name]}/{self.buffer_threshold}")
                            continue

                    if name == "Unknown":
                        self.buzzer.beep_unknown()
                        self.lcd.display("IT SOLUTIONS Pvt", "Unknown Face")
                        print("\n" + "="*40)
                        print(" [!] UNKNOWN / ACCURACY FAIL")
                        print("="*40)
                        print(" 1. Rescan (Try again)")
                        print(" 2. Register New Employee")
                        print("="*40)
                        
                        choice = input("Enter Choice (1/2): ").strip()
                        
                        if choice == '2':
                            self.lcd.display("Enter Name", "in Terminal...")
                            new_name = input("Enter Full Name: ").strip()
                            
                            self.lcd.display("Enter Emp ID", "in Terminal...")
                            new_id = input("Enter Employee ID: ").strip()
                            
                            self.lcd.display("Enter Phone", "in Terminal...")
                            new_phone = input("Enter Phone Number: ").strip()
                            
                            new_dept = "IT SOLUTIONS" # Fixed as requested
                            
                            if new_name and new_id:
                                # Save current frame as first photo profile
                                temp_photo = os.path.join(DATABASE_DIR, f"temp_reg.jpg")
                                cv2.imwrite(temp_photo, frame) 
                                
                                self.lcd.display("Stay Still...", "Syncing (1/3)")
                                self.face_module.register_new_face(f"{new_name}_1", temp_photo)
                                
                                time.sleep(0.5)
                                ret, f2 = self.camera.read()
                                if ret:
                                    self.lcd.display("Stay Still...", "Syncing (2/3)")
                                    p2 = os.path.join(DATABASE_DIR, f"temp_2.jpg")
                                    cv2.imwrite(p2, f2)
                                    self.face_module.register_new_face(f"{new_name}_2", p2)
                                    os.remove(p2)
                                    
                                time.sleep(0.5)
                                ret, f3 = self.camera.read()
                                if ret:
                                    self.lcd.display("Stay Still...", "Syncing (3/3)")
                                    p3 = os.path.join(DATABASE_DIR, f"temp_3.jpg")
                                    cv2.imwrite(p3, f3)
                                    self.face_module.register_new_face(f"{new_name}_3", p3)
                                    os.remove(p3)

                                # Update CSV
                                df = pd.read_csv(EMPLOYEES_FILE)
                                new_emp = pd.DataFrame([[new_id, new_name, new_phone, new_dept, datetime.now(IST).strftime("%Y-%m-%d")]], 
                                                       columns=['Employee_ID', 'Name', 'Phone', 'Department', 'Join_Date'])
                                df = pd.concat([df, new_emp], ignore_index=True)
                                df.to_csv(EMPLOYEES_FILE, index=False)
                                
                                self.lcd.display("Reg Success!", new_name)
                                print(f"Successfully registered {new_name} with 3D profile")
                                time.sleep(2)
                                
                                if os.path.exists(temp_photo):
                                    os.remove(temp_photo)
                            else:
                                print("Registration cancelled (Missing details).")
                        
                        self.lcd.display("IT SOLUTIONS Pvt", "Scan Face")
                        
                    else:
                        if self.mark_attendance(name):
                            time.sleep(2)
                            self.lcd.display("IT SOLUTIONS Pvt", "Scan Face")

                # Minimal non-blocking sleep
                time.sleep(0.01)
                
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
