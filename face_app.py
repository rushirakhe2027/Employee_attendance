import cv2
import time
import pandas as pd
from datetime import datetime
import os
import pytz

from modules.face_recognition_module import FaceRecognitionModule
from modules.lcd_module import LCDModule
from modules.buzzer_module import BuzzerModule

# ─── Configuration ──────────────────────────────────────────────────────────
OFFICE_START   = "09:30"   # On-time cutoff
OFFICE_END     = "17:30"   # Normal checkout time
NOON_CUTOFF    = "13:00"   # Before this hour = CHECK-IN, after = CHECK-OUT
IST            = pytz.timezone('Asia/Kolkata')
DATABASE_DIR   = "database"
ATTENDANCE_FILE = f"{DATABASE_DIR}/attendance.csv"
EMPLOYEES_FILE  = f"{DATABASE_DIR}/employees.csv"
FACE_DIR        = f"{DATABASE_DIR}/faces"

def now_ist():
    return datetime.now(IST)

class EmployeeAttendanceApp:
    def __init__(self):
        os.makedirs(DATABASE_DIR, exist_ok=True)
        os.makedirs(FACE_DIR, exist_ok=True)
        self._ensure_db()

        self.face_module = FaceRecognitionModule()
        self.lcd         = LCDModule()
        self.buzzer      = BuzzerModule()

        # Camera — V4L2 for stability on Pi 1 B+
        self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.camera.set(cv2.CAP_PROP_FPS,          5)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self.running          = True
        self.detection_counter = 0
        self.cooldown_until   = {}  # name -> datetime — prevents re-scan within 30s

    # ─── DB Bootstrap ────────────────────────────────────────────────────────
    def _ensure_db(self):
        # 1. Employees DB
        if not os.path.exists(EMPLOYEES_FILE):
            pd.DataFrame(columns=['Employee_ID','Name','Phone','Department','Join_Date']
                         ).to_csv(EMPLOYEES_FILE, index=False)
        
        # 2. Attendance DB
        if not os.path.exists(ATTENDANCE_FILE):
            pd.DataFrame(columns=['Employee_ID','Name','Date','Time','Status','Type']
                         ).to_csv(ATTENDANCE_FILE, index=False)
        else:
            # Migration: Ensure existing file has mandatory columns
            try:
                df = pd.read_csv(ATTENDANCE_FILE)
                df.columns = [str(c).strip() for c in df.columns]
                changed = False
                for col in ['Employee_ID','Name','Date','Time','Status','Type']:
                    if col not in df.columns:
                        df[col] = "Legacy"
                        changed = True
                if changed:
                    df.to_csv(ATTENDANCE_FILE, index=False)
                    print("[DB] Migrated attendance.csv to include new columns.")
            except Exception as e:
                print(f"[DB] Migration error: {e}")

    # ─── Core Attendance Logic ────────────────────────────────────────────────
    def _get_todays_records(self, name):
        """Return today's attendance rows for this employee."""
        today = now_ist().strftime("%Y-%m-%d")
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Defensive check for columns
            if 'Name' not in df.columns or 'Date' not in df.columns:
                return pd.DataFrame()
                
            return df[(df['Name'] == name) & (df['Date'] == today)]
        except Exception:
            return pd.DataFrame()

    def mark_attendance(self, name):
        now         = now_ist()
        today       = now.strftime("%Y-%m-%d")
        current_t   = now.strftime("%H:%M:%S")
        current_time = now.time()

        office_start = datetime.strptime(OFFICE_START, "%H:%M").time()
        office_end   = datetime.strptime(OFFICE_END,   "%H:%M").time()
        noon         = datetime.strptime(NOON_CUTOFF,  "%H:%M").time()

        # Anti-spam: ignore for 30 s after last scan
        if name in self.cooldown_until and now < self.cooldown_until[name]:
            return

        # Get employee ID
        try:
            df_emp = pd.read_csv(EMPLOYEES_FILE)
            emp_row = df_emp[df_emp['Name'] == name]
            emp_id  = str(emp_row['Employee_ID'].values[0]) if not emp_row.empty else "?"
        except Exception:
            emp_id = "?"

        # What records exist for today?
        today_df   = self._get_todays_records(name)
        
        # Robust column check
        has_in = False
        has_out = False
        
        if not today_df.empty and 'Type' in today_df.columns:
            has_in  = not today_df[today_df['Type'] == 'IN'].empty
            has_out = not today_df[today_df['Type'] == 'OUT'].empty
        elif not today_df.empty:
            # Fallback for very old formats: assume first scan today is IN
            has_in = True 


        # ── Decision Tree ────────────────────────────────────────────────────
        if has_out:
            # Already OUT for today → refuse
            self.lcd.display("Already Done", "Try Tomorrow!")
            self.buzzer.beep_rejected()      # ▪▪▪▪▪  5 rapid — access denied
            print(f"[SKIP] {name} already checked OUT today.")
            self.cooldown_until[name] = now.__class__(
                now.year, now.month, now.day, 23, 59, 59, tzinfo=IST)
            return

        if not has_in:
            # ── FIRST SCAN OF THE DAY = CHECK-IN ──────────────────────────
            if current_time <= office_start:
                status = "On-Time"
                lcd2   = "On-Time Arrival"
                self.buzzer.beep_on_time()          # ▬  1 long
            else:
                status = "Late Arrived"
                lcd2   = "Late Arrived!"
                self.buzzer.beep_late_or_early()    # ▪ ▪ ▪  3 short
            rec_type = "IN"
            self.lcd.display(f"Welcome!", name[:16])
            time.sleep(1)
            self.lcd.display("IN  Recorded", lcd2)

        else:
            # ── SECOND SCAN = CHECK-OUT ────────────────────────────────────
            if current_time < office_end:
                status = "Early Leaving"
                lcd2   = "Early Leaving!"
                self.buzzer.beep_late_or_early()    # ▪ ▪ ▪  3 short
            else:
                status = "Left"
                lcd2   = "Have a nice day"
                self.buzzer.beep_on_time()          # ▬  1 long
            rec_type = "OUT"
            self.lcd.display("Goodbye!", name[:16])
            time.sleep(1)
            self.lcd.display("OUT Recorded", lcd2)

        # Save to CSV
        new_row = pd.DataFrame([[emp_id, name, today, current_t, status, rec_type]],
                               columns=['Employee_ID','Name','Date','Time','Status','Type'])
        new_row.to_csv(ATTENDANCE_FILE, mode='a',
                       header=not os.path.exists(ATTENDANCE_FILE), index=False)

        print(f"[{rec_type}] {name} | {status} | {current_t}")

        # 30-second cooldown so they don't triple-scan
        from datetime import timedelta
        self.cooldown_until[name] = now + timedelta(seconds=30)

        time.sleep(2)
        self.lcd.display("IT SOLUTIONS Pvt", "Scan Face")

    # ─── Registration ─────────────────────────────────────────────────────────
    def register_employee(self, rgb_frame):
        self.lcd.display("Enter Name", "in Terminal...")
        new_name = input("\nEnter Full Name      : ").strip()
        if not new_name:
            self.lcd.display("Cancelled", "No name given")
            return

        self.lcd.display("Enter Emp ID", "in Terminal...")
        new_id = input("Enter Employee ID    : ").strip()
        if not new_id:
            self.lcd.display("Cancelled", "No ID given")
            return

        self.lcd.display("Enter Phone", "in Terminal...")
        new_phone = input("Enter Phone Number   : ").strip()

        # Capture 5 frames for multi-sample registration
        self.lcd.display("Stay Still...", "5 Samples...")
        print("\n[REG] Hold still — capturing 5 samples over 2.5 seconds...")
        reg_frames = []
        for i in range(5):
            print(f"[REG] Sample {i+1}/5...")
            ret, rf = self.camera.read()
            if ret:
                rf = cv2.flip(rf, 1)
                reg_frames.append(cv2.cvtColor(rf, cv2.COLOR_BGR2RGB))
            else:
                reg_frames.append(rgb_frame)
            time.sleep(0.5)

        success, msg = self.face_module.register_new_face(new_name, reg_frames)

        if success:
            try:
                df = pd.read_csv(EMPLOYEES_FILE)
            except FileNotFoundError:
                df = pd.DataFrame(columns=['Employee_ID','Name','Phone','Department','Join_Date'])

            new_emp = pd.DataFrame(
                [[new_id, new_name, new_phone, "IT SOLUTIONS",
                  now_ist().strftime("%Y-%m-%d")]],
                columns=['Employee_ID','Name','Phone','Department','Join_Date'])
            df = pd.concat([df, new_emp], ignore_index=True)
            df.to_csv(EMPLOYEES_FILE, index=False)

            self.lcd.display("Registered!", new_name[:16])
            self.buzzer.beep_present()
            print(f"[REG] SUCCESS: {msg}")
        else:
            self.lcd.display("Reg Failed", "Try again")
            self.buzzer.beep_unknown()
            print(f"[REG] FAILED: {msg}")

        time.sleep(2)
        self.lcd.display("IT SOLUTIONS Pvt", "Scan Face")

    # ─── Main Loop ────────────────────────────────────────────────────────────
    def run(self):
        print("=" * 44)
        print("  IT SOLUTIONS Pvt Ltd — Attendance System")
        print("=" * 44)
        self.lcd.display("IT SOLUTIONS Pvt", "Scan Face")

        frame_count = 0
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                frame_count += 1
                frame = cv2.flip(frame, 1)

                # Check every 2 frames (fast Haar check)
                if frame_count % 2 != 0:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ── STEP 1: Fast Haar pre-detect ──────────────────────────
                bbox = self.face_module.just_detect(rgb_frame)
                if bbox:
                    self.detection_counter += 1
                else:
                    self.detection_counter = 0
                    continue

                # ── STEP 2: Require 3 consecutive detections ──────────────
                if self.detection_counter < 3:
                    continue

                # ── STEP 3: Heavy dlib recognition ────────────────────────
                names = self.face_module.detect_and_recognize(rgb_frame)
                self.detection_counter = 0

                if not names:
                    continue

                name = names[0]

                if name != "Unknown":
                    self.mark_attendance(name)
                else:
                    # Unknown face menu
                    self.buzzer.beep_unknown()
                    self.lcd.display("Unknown Face", "1:Reg  2:Skip")
                    print("\n" + "=" * 44)
                    print("  [!] UNKNOWN FACE")
                    print("=" * 44)
                    print("  1. Register New Employee")
                    print("  2. Skip / Rescan")
                    print("=" * 44)
                    choice = input("  Choice (1/2): ").strip()
                    if choice == "1":
                        self.register_employee(rgb_frame)
                    else:
                        self.lcd.display("IT SOLUTIONS Pvt", "Scan Face")

                time.sleep(0.01)

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
        self.camera.release()
        self.lcd.clear()
        self.buzzer.cleanup()
        print("\nSystem Shutdown.")


if __name__ == "__main__":
    app = EmployeeAttendanceApp()
    app.run()
