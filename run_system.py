import subprocess
import time
import socket
import os
import sys

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def launch():
    ip = get_ip()
    print("\n" + "="*50)
    print("  ⭐ IT SOLUTIONS PVT LTD — ATTENDANCE SYSTEM ⭐")
    print("="*50)
    print(f"  🌐 WEB DASHBOARD: http://{ip}:5000")
    print("  📷 CAMERA SYSTEM: Starting...")
    print("="*50 + "\n")

    # 1. Start the Flask Dashboard in the background
    log_file = open("web_system.log", "w")
    web_process = subprocess.Popen([sys.executable, "app.py"], 
                                   stdout=log_file, 
                                   stderr=log_file)
    
    time.sleep(2) # Give it a moment to bind to the port

    # 2. Start the Facial Attendance System in the foreground
    # Note: We use libcamerify for the camera system
    try:
        print("[Launcher] Attendance System is LIVE. Press Ctrl+C to stop both.")
        subprocess.run(["libcamerify", "python3", "face_app.py"])
    except KeyboardInterrupt:
        print("\n[Launcher] Shutting down...")
    finally:
        web_process.terminate()
        print("[Launcher] All systems stopped.")

if __name__ == "__main__":
    launch()
