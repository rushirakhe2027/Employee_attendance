import RPi.GPIO as GPIO
import time

class BuzzerModule:
    def __init__(self, pin=18):
        self.pin = pin
        self.setup()

    def setup(self):
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            self.GPIO.setup(self.pin, self.GPIO.OUT)
            self.GPIO.output(self.pin, self.GPIO.LOW)
            self.simulation = False
        except Exception as e:
            print(f"Buzzer Simulation Mode: {e}")
            self.simulation = True


    def beep_present(self):
        """Single beep for Present"""
        self._beep(0.2)

    def beep_late(self):
        """Double beep for Late"""
        self._beep(0.1)
        time.sleep(0.1)
        self._beep(0.1)

    def beep_unknown(self):
        """Long beep for Unknown"""
        self._beep(1.0)

    def _beep(self, duration):
        if self.simulation:
            print(f"[BUZZER SIMULATION] BEEP for {duration}s")
            return
        try:
            self.GPIO.output(self.pin, self.GPIO.HIGH)
            time.sleep(duration)
            self.GPIO.output(self.pin, self.GPIO.LOW)
        except Exception:
            pass

    def cleanup(self):
        if self.simulation:
            return
        try:
            self.GPIO.output(self.pin, self.GPIO.LOW)
            self.GPIO.cleanup()
        except Exception:
            pass

