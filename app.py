from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import os
from datetime import datetime
import pytz

app = Flask(__name__)
app.secret_key = "itsolpvtltd2024"

IST            = pytz.timezone('Asia/Kolkata')
DATABASE_DIR   = "database"
EMPLOYEES_FILE = f"{DATABASE_DIR}/employees.csv"
ATTENDANCE_FILE = f"{DATABASE_DIR}/attendance.csv"
FACE_DIR       = f"{DATABASE_DIR}/faces"

def now_ist():
    return datetime.now(IST)

def ensure_db():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(FACE_DIR, exist_ok=True)
    if not os.path.exists(EMPLOYEES_FILE):
        pd.DataFrame(columns=['Employee_ID','Name','Phone','Department','Join_Date']
                     ).to_csv(EMPLOYEES_FILE, index=False)
    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=['Employee_ID','Name','Date','Time','Status','Type']
                     ).to_csv(ATTENDANCE_FILE, index=False)

@app.context_processor
def inject_now():
    return {'datetime': datetime, 'now': now_ist()}

# ─── Dashboard ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    ensure_db()
    try:
        df_att = pd.read_csv(ATTENDANCE_FILE)
        # Fix possible empty file or missing headers
        if df_att.empty:
            df_att = pd.DataFrame(columns=['Employee_ID','Name','Date','Time','Status','Type'])
    except Exception:
        df_att = pd.DataFrame(columns=['Employee_ID','Name','Date','Time','Status','Type'])
        
    try:
        df_emp = pd.read_csv(EMPLOYEES_FILE)
        if df_emp.empty:
             df_emp = pd.DataFrame(columns=['Employee_ID','Name','Phone','Department','Join_Date'])
    except Exception:
        df_emp = pd.DataFrame(columns=['Employee_ID','Name','Phone','Department','Join_Date'])
    
    # Normalize column names
    df_att.columns = [str(c).strip() for c in df_att.columns]
    
    today = now_ist().strftime("%Y-%m-%d")
    
    # Safely filter for today
    if 'Date' in df_att.columns and not df_att.empty:
        df_today = df_att[df_att['Date'] == today]
    else:
        df_today = pd.DataFrame(columns=df_att.columns)
    
    # Identify columns safely
    col_status = 'Status' if 'Status' in df_today.columns else None
    col_type   = 'Type'   if 'Type'   in df_today.columns else None
    col_id     = 'Employee_ID' if 'Employee_ID' in df_today.columns else None

    # Calculate Attendance
    present_ids = []
    if not df_today.empty and col_id:
        if col_type:
            present_ids = df_today[df_today[col_type] == 'IN'][col_id].unique()
        else:
            present_ids = df_today[col_id].unique()

    stats = {
        'total_employees'   : len(df_emp),
        'present_today'     : len(present_ids),
        'late_today'        : len(df_today[df_today[col_status] == 'Late Arrived']) if col_status and not df_today.empty else 0,
        'early_leaving_today': len(df_today[df_today[col_status] == 'Early Leaving']) if col_status and not df_today.empty else 0,
        'absent_today'      : max(0, len(df_emp) - len(present_ids)),
    }

    # Prepare latest records (handling potential missing columns in dict)
    latest = df_att.tail(15).iloc[::-1].to_dict('records')
    return render_template('index.html', stats=stats, attendance=latest, today=today)


# ─── All Employees ───────────────────────────────────────────────────────────
@app.route('/employees')
def employees():
    ensure_db()
    df_emp  = pd.read_csv(EMPLOYEES_FILE)
    df_att  = pd.read_csv(ATTENDANCE_FILE)
    today   = now_ist().strftime("%Y-%m-%d")

    emp_list = df_emp.to_dict('records')
    # Annotate with today's status
    for emp in emp_list:
        emp_today = df_att[
            (df_att['Employee_ID'].astype(str) == str(emp['Employee_ID'])) &
            (df_att['Date'] == today)
        ]
        has_in  = not emp_today[emp_today['Type'] == 'IN'].empty
        has_out = not emp_today[emp_today['Type'] == 'OUT'].empty
        emp['today_status'] = (
            "Checked Out" if has_out else
            "Present"     if has_in  else
            "Absent"
        )
        emp['has_face'] = os.path.exists(os.path.join(FACE_DIR, f"{emp['Name']}.npy"))

    return render_template('employees.html', employees=emp_list)


# ─── Employee Attendance History ─────────────────────────────────────────────
@app.route('/employee/<emp_id>')
def employee_detail(emp_id):
    ensure_db()
    df_emp = pd.read_csv(EMPLOYEES_FILE)
    df_att = pd.read_csv(ATTENDANCE_FILE)

    emp_rows = df_emp[df_emp['Employee_ID'].astype(str) == str(emp_id)]
    if emp_rows.empty:
        flash("Employee not found.", "danger")
        return redirect(url_for('employees'))

    employee = emp_rows.iloc[0].to_dict()
    history  = df_att[df_att['Employee_ID'].astype(str) == str(emp_id)].iloc[::-1].to_dict('records')

    total_in      = len([h for h in history if h['Type'] == 'IN'])
    late_count    = len([h for h in history if h['Status'] == 'Late Arrived'])
    early_count   = len([h for h in history if h['Status'] == 'Early Leaving'])
    on_time_count = len([h for h in history if h['Status'] == 'On-Time'])

    stats = {
        'total_days'   : total_in,
        'on_time'      : on_time_count,
        'late'         : late_count,
        'early_leaving': early_count,
    }

    # Photo if available
    photo_path = f"database/faces/{employee['Name']}.jpg"
    has_photo  = os.path.exists(photo_path)

    return render_template('employee_history.html',
                           employee=employee,
                           history=history,
                           stats=stats,
                           has_photo=has_photo)


# ─── Full Attendance Log ─────────────────────────────────────────────────────
@app.route('/attendance')
def attendance():
    ensure_db()
    date_filter = request.args.get('date', '')
    df_att = pd.read_csv(ATTENDANCE_FILE)

    if date_filter:
        df_att = df_att[df_att['Date'] == date_filter]

    attendance_list = df_att.iloc[::-1].to_dict('records')
    return render_template('attendance.html',
                           attendance=attendance_list,
                           date_filter=date_filter)


# ─── Delete Employee ─────────────────────────────────────────────────────────
@app.route('/delete_employee/<emp_id>')
def delete_employee(emp_id):
    ensure_db()
    df = pd.read_csv(EMPLOYEES_FILE)
    name_rows = df[df['Employee_ID'].astype(str) == str(emp_id)]

    if not name_rows.empty:
        name = name_rows.iloc[0]['Name']
        # Remove face files
        for ext in ['.npy', '.jpg']:
            fp = os.path.join(FACE_DIR, f"{name}{ext}")
            if os.path.exists(fp):
                os.remove(fp)

    df = df[df['Employee_ID'].astype(str) != str(emp_id)]
    df.to_csv(EMPLOYEES_FILE, index=False)
    flash("Employee and face data deleted.", "info")
    return redirect(url_for('employees'))


# ─── Download Attendance CSV ─────────────────────────────────────────────────
@app.route('/download')
def download():
    return send_file(ATTENDANCE_FILE, as_attachment=True)


if __name__ == '__main__':
    ensure_db()
    app.run(host='0.0.0.0', port=5000, debug=False)