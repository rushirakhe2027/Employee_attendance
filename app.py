from flask import Flask, render_template, request, redirect, url_for, send_file, flash, Response
from flask_socketio import SocketIO, emit
import pandas as pd
import os
import cv2
import requests
import json
from datetime import datetime
import time
import pytz
from modules.face_recognition_module import FaceRecognitionModule

app = Flask(__name__)
app.secret_key = "supersecretkey"
socketio = SocketIO(app, cors_allowed_origins="*")

IST = pytz.timezone('Asia/Kolkata')

DATABASE_DIR = "database"
EMPLOYEES_FILE = f"{DATABASE_DIR}/employees.csv"
ATTENDANCE_FILE = f"{DATABASE_DIR}/attendance.csv"
FACES_DIR = f"{DATABASE_DIR}/faces"
WEBHOOK_URL = os.environ.get("ATTENDANCE_WEBHOOK_URL") # Optional environment variable

# Global frame for video streaming
video_frame = None
face_reg_module = FaceRecognitionModule() # Initialize for registration

IST = pytz.timezone('Asia/Kolkata')

@app.context_processor
def inject_now():
    return {'datetime': datetime, 'now': datetime.now(IST)}

def ensure_db():
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)
    if not os.path.exists(EMPLOYEES_FILE):
        pd.DataFrame(columns=['Employee_ID', 'Name', 'Phone', 'Department', 'Join_Date']).to_csv(EMPLOYEES_FILE, index=False)
    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=['Employee_ID', 'Name', 'Date', 'Time', 'Status']).to_csv(ATTENDANCE_FILE, index=False)

def send_webhook(data):
    """Optional webhook for external integration"""
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json=data, timeout=2)
        except Exception as e:
            print(f"Webhook error: {e}")

@app.route('/')
def index():
    ensure_db()
    df_att = pd.read_csv(ATTENDANCE_FILE)
    df_emp = pd.read_csv(EMPLOYEES_FILE)
    
    today = datetime.now(IST).strftime("%Y-%m-%d")
    df_today = df_att[df_att['Date'] == today]
    
    stats = {
        'total_employees': len(df_emp),
        'total_attendance': len(df_att),
        'late_today': len(df_today[df_today['Status'] == 'Late Arrival']),
        'early_leaving_today': len(df_today[df_today['Status'] == 'Early Leaving']),
        'present_today': len(df_today[df_today['Status'].isin(['On-Time', 'Late Arrival'])])
    }

    
    latest_attendance = df_att.tail(10).iloc[::-1].to_dict('records')
    return render_template('index.html', stats=stats, attendance=latest_attendance)

@app.route('/employees', methods=['GET', 'POST'])
def employees():
    ensure_db()
    if request.method == 'POST':
        emp_id = request.form.get('emp_id')
        name = request.form.get('name')
        phone = request.form.get('phone')
        dept = request.form.get('dept')
        file = request.files.get('photo')
        
        if file and file.filename != '':
            temp_path = os.path.join(FACES_DIR, f"temp_{name}.jpg")
            file.save(temp_path)
            
            # Use the module to align and extract features
            success, message = face_reg_module.register_new_face(name, temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if not success:
                flash(f"Error registering face: {message}", "danger")
                return redirect(url_for('employees'))
        
        df = pd.read_csv(EMPLOYEES_FILE)
        new_emp = pd.DataFrame([[emp_id, name, phone, dept, datetime.now(IST).strftime("%Y-%m-%d")]], 
                               columns=['Employee_ID', 'Name', 'Phone', 'Department', 'Join_Date'])
        
        df = pd.concat([df, new_emp], ignore_index=True)
        df.to_csv(EMPLOYEES_FILE, index=False)
        flash(f"Employee {name} registered with face features data!", "success")
        return redirect(url_for('employees'))


    df_emp = pd.read_csv(EMPLOYEES_FILE)
    employees_list = df_emp.to_dict('records')
    return render_template('employees.html', employees=employees_list)

@app.route('/delete_employee/<emp_id>')
def delete_employee(emp_id):
    df = pd.read_csv(EMPLOYEES_FILE)
    df = df[df['Employee_ID'].astype(str) != str(emp_id)]
    df.to_csv(EMPLOYEES_FILE, index=False)
    flash("Employee deleted.", "info")
    return redirect(url_for('employees'))

@app.route('/employee_history/<emp_id>')
def employee_history(emp_id):
    ensure_db()
    df_emp = pd.read_csv(EMPLOYEES_FILE)
    df_att = pd.read_csv(ATTENDANCE_FILE)
    
    # Get employee details
    employee = df_emp[df_emp['Employee_ID'].astype(str) == str(emp_id)].to_dict('records')
    if not employee:
        flash("Employee not found.", "danger")
        return redirect(url_for('employees'))
    
    employee = employee[0]
    
    # Filter attendance for this employee
    history = df_att[df_att['Employee_ID'].astype(str) == str(emp_id)].iloc[::-1].to_dict('records')
    
    # Calculate stats for this specific employee
    stats = {
        'total_present': len(history),
        'late_count': len([h for h in history if h['Status'] == 'Late']),
        'on_time_count': len([h for h in history if h['Status'] == 'Present'])
    }
    
    return render_template('employee_history.html', employee=employee, history=history, stats=stats)
@app.route('/attendance')
def attendance():
    ensure_db()
    df_att = pd.read_csv(ATTENDANCE_FILE)
    attendance_list = df_att.iloc[::-1].to_dict('records')
    return render_template('attendance.html', attendance=attendance_list)

@app.route('/download')
def download():
    return send_file(ATTENDANCE_FILE, as_attachment=True)

# --- Real-time Video Stream ---
def generate_frames():
    global video_frame
    while True:
        if video_frame is not None:
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- SocketIO Events for face_app.py to push data ---
@socketio.on('recognition_event')
def handle_recognition(data):
    """Called by face_app.py when someone is recognized"""
    print(f"Real-time update: {data['name']} marked as {data['status']}")
    emit('new_attendance', data, broadcast=True)
    send_webhook(data)

@socketio.on('frame_upload')
def handle_frame(data):
    """Optionally handle frame streaming over sockets if preferred"""
    global video_frame
    # In this implementation, we assume face_app will update global state or similar
    # For simplicity, we'll let face_app.py call a local function or use a shared queue
    pass

def update_global_frame(frame):
    global video_frame
    video_frame = frame

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)