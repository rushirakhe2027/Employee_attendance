import serial
import time

class GSMModule:
    def __init__(self, port="/dev/serial0", baud=9600):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            print(f"[GSM] Connected on {port}")
            self._init_gsm()
        except Exception as e:
            print(f"[GSM] Error: {e}")
            self.ser = None

    def _init_gsm(self):
        if not self.ser: return
        self.ser.write(b"AT\r\n")
        time.sleep(0.5)
        self.ser.write(b"AT+CMGF=1\r\n") # Set to Text Mode
        time.sleep(0.5)
        print("[GSM] SMS mode initialized.")

    def send_sms(self, phone, message):
        if not self.ser: 
            print("[GSM] SMS failed: Hardware not found.")
            return False
        
        try:
            print(f"[GSM] Sending SMS to {phone}...")
            self.ser.write(f'AT+CMGS="{phone}"\r\n'.encode())
            time.sleep(0.5)
            self.ser.write(f'{message}\x1A'.encode()) # \x1A is CTRL+Z
            time.sleep(3)
            print("[GSM] SMS Sent successfully.")
            return True
        except Exception as e:
            print(f"[GSM] SMS Error: {e}")
            return False
