import time

BUZZER_PIN = 18

class BuzzerModule:
    def __init__(self, pin=BUZZER_PIN):
        self.pin = pin
        self.simulation = False
        self._setup()

    def _setup(self):
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            self.GPIO.setup(self.pin, self.GPIO.OUT)
            self.GPIO.output(self.pin, self.GPIO.LOW)
            self.simulation = False
            print(f"[Buzzer] Ready on GPIO {self.pin}")
        except Exception as e:
            print(f"[Buzzer] Simulation mode: {e}")
            self.simulation = True

    # ─── Beep Patterns ──────────────────────────────────────────────────────

    def beep_on_time(self):
        """
        1 long beep (0.5s) — Normal check-in / check-out.
        Pattern: ▬
        """
        self._beep(0.5)

    def beep_late_or_early(self):
        """
        3 short beeps — Late Arrived OR Early Leaving.
        Pattern: ▪ ▪ ▪
        """
        for _ in range(3):
            self._beep(0.15)
            time.sleep(0.12)

    def beep_unknown(self):
        """
        2 long beeps — Unknown face detected.
        Pattern: ▬ ▬
        """
        self._beep(0.6)
        time.sleep(0.2)
        self._beep(0.6)

    def beep_rejected(self):
        """
        5 rapid beeps — Already done / Try Tomorrow (access denied).
        Pattern: ▪▪▪▪▪
        """
        for _ in range(5):
            self._beep(0.07)
            time.sleep(0.07)

    def beep_registered(self):
        """
        2 short + 1 long — Registration success.
        Pattern: ▪ ▪ ▬
        """
        self._beep(0.1)
        time.sleep(0.1)
        self._beep(0.1)
        time.sleep(0.1)
        self._beep(0.4)

    # ─── Keep legacy names so old calls don't crash ─────────────────────────

    def beep_present(self):
        """Alias → On-Time beep"""
        self.beep_on_time()

    def beep_late(self):
        """Alias → Late/Early beep"""
        self.beep_late_or_early()

    # ─── Internal ────────────────────────────────────────────────────────────

    def _beep(self, duration):
        if self.simulation:
            print(f"[BUZZER] BEEP {duration:.2f}s")
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
