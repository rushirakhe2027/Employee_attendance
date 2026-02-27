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
    
    # Ensure Employees file
    if not os.path.exists(EMPLOYEES_FILE):
        pd.DataFrame(columns=['Employee_ID','Name','Phone','Department','Join_Date']
                     ).to_csv(EMPLOYEES_FILE, index=False)
    
    # Ensure Attendance file and migrate if needed
    mandatory_cols = ['Employee_ID','Name','Date','Time','Status','Type']
    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=mandatory_cols).to_csv(ATTENDANCE_FILE, index=False)
    else:
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            df.columns = [str(c).strip() for c in df.columns]
            changed = False
            for col in mandatory_cols:
                if col not in df.columns:
                    df[col] = "N/A"
                    changed = True
            if changed:
                df.to_csv(ATTENDANCE_FILE, index=False)
        except:
            pass

@app.context_processor
def inject_now():
    return {'datetime': datetime, 'now': now_ist()}

# ─── Dashboard ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    ensure_db()
    try:
        df_att = pd.read_csv(ATTENDANCE_FILE)
        df_att.columns = [str(c).strip() for c in df_att.columns]
        if df_att.empty:
            df_att = pd.DataFrame(columns=['Employee_ID','Name','Date','Time','Status','Type'])
    except:
        df_att = pd.DataFrame(columns=['Employee_ID','Name','Date','Time','Status','Type'])
        
    try:
        df_emp = pd.read_csv(EMPLOYEES_FILE)
        df_emp.columns = [str(c).strip() for c in df_emp.columns]
    except:
        df_emp = pd.DataFrame(columns=['Employee_ID','Name','Phone','Department','Join_Date'])
    
    today = now_ist().strftime("%Y-%m-%d")
    
    # Safe Stats Logic
    df_today = df_att[df_att['Date'] == today] if 'Date' in df_att.columns else pd.DataFrame()
    
    present_ids = []
    if not df_today.empty and 'Employee_ID' in df_today.columns:
        if 'Type' in df_today.columns:
            present_ids = df_today[df_today['Type'] == 'IN']['Employee_ID'].unique()
        else:
            present_ids = df_today['Employee_ID'].unique()

    stats = {
        'total_employees'   : len(df_emp),
        'present_today'     : len(present_ids),
        'late_today'        : len(df_today[df_today['Status'] == 'Late Arrived']) if 'Status' in df_today.columns else 0,
        'early_leaving_today': len(df_today[df_today['Status'] == 'Early Leaving']) if 'Status' in df_today.columns else 0,
    }
    stats['absent_today'] = max(0, stats['total_employees'] - stats['present_today'])

    # presentation ready dictionary (ensures all keys exist for Jinja)
    latest_df = df_att.tail(15).iloc[::-1]
    for col in ['Employee_ID','Name','Date','Time','Status','Type']:
        if col not in latest_df.columns:
            latest_df[col] = "N/A"
            
    latest = latest_df.to_dict('records')
    return render_template('index.html', stats=stats, attendance=latest, today=today)


# ─── All Employees ───────────────────────────────────────────────────────────
@app.route('/employees')
def employees():
    ensure_db()
    try:
        df_emp = pd.read_csv(EMPLOYEES_FILE)
        df_emp.columns = [str(c).strip() for c in df_emp.columns]
    except:
        df_emp = pd.DataFrame(columns=['Employee_ID','Name'])
        
    try:
        df_att = pd.read_csv(ATTENDANCE_FILE)
        df_att.columns = [str(c).strip() for c in df_att.columns]
    except:
        df_att = pd.DataFrame(columns=['Employee_ID','Date','Type'])

    today = now_ist().strftime("%Y-%m-%d")
    emp_list = df_emp.to_dict('records')
    
    for emp in emp_list:
        eid = str(emp.get('Employee_ID', ''))
        nm = str(emp.get('Name', ''))
        
        emp_today = df_att[
            (df_att['Employee_ID'].astype(str) == eid) &
            (df_att['Date'] == today)
        ] if 'Employee_ID' in df_att.columns and 'Date' in df_att.columns else pd.DataFrame()
        
        has_in  = not emp_today[emp_today['Type'] == 'IN'].empty if 'Type' in emp_today.columns else False
        has_out = not emp_today[emp_today['Type'] == 'OUT'].empty if 'Type' in emp_today.columns else False
        
        emp['today_status'] = "Checked Out" if has_out else "Present" if has_in else "Absent"
        emp['has_face'] = os.path.exists(os.path.join(FACE_DIR, f"{nm}.npy"))

    return render_template('employees.html', employees=emp_list)


# ─── Employee Attendance History ─────────────────────────────────────────────
@app.route('/employee/<emp_id>')
def employee_detail(emp_id):
    ensure_db()
    try:
        df_emp = pd.read_csv(EMPLOYEES_FILE)
        df_att = pd.read_csv(ATTENDANCE_FILE)
        df_emp.columns = [str(c).strip() for c in df_emp.columns]
        df_att.columns = [str(c).strip() for c in df_att.columns]
    except:
        flash("Database Error", "danger")
        return redirect(url_for('employees'))

    emp_rows = df_emp[df_emp['Employee_ID'].astype(str) == str(emp_id)]
    if emp_rows.empty:
        flash("Employee not found.", "danger")
        return redirect(url_for('employees'))

    employee = emp_rows.iloc[0].to_dict()
    
    history_df = df_att[df_att['Employee_ID'].astype(str) == str(emp_id)].iloc[::-1].copy()
    # Ensure columns for Jinja
    for col in ['Date','Time','Status','Type']:
        if col not in history_df.columns:
            history_df[col] = "N/A"
            
    history = history_df.to_dict('records')

    stats = {
        'total_days'   : len(history_df[history_df['Type'] == 'IN']) if 'Type' in history_df.columns else 0,
        'on_time'      : len(history_df[history_df['Status'] == 'On-Time']) if 'Status' in history_df.columns else 0,
        'late'         : len(history_df[history_df['Status'] == 'Late Arrived']) if 'Status' in history_df.columns else 0,
        'early_leaving': len(history_df[history_df['Status'] == 'Early Leaving']) if 'Status' in history_df.columns else 0,
    }

    photo_path = f"database/faces/{employee.get('Name','')}.jpg"
    has_photo  = os.path.exists(photo_path)

    return render_template('employee_history.html', employee=employee, history=history, stats=stats, has_photo=has_photo)


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