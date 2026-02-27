import time

class BuzzerModule:
    """
    Buzzer control using GPIO PWM.
    PWM ensures clean beep cutoff — even if the program crashes,
    the duty cycle drops to 0 so the buzzer stops on its own.
    """
    def __init__(self, pin=18):
        self.pin        = pin
        self.pwm        = None
        self.simulation = False
        self._setup()

    def _setup(self):
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setwarnings(False)
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setup(self.pin, self.GPIO.OUT)

            # PWM at 2000 Hz — standard buzzer frequency
            self.pwm = self.GPIO.PWM(self.pin, 2000)
            self.pwm.start(0)  # Start silent (0% duty cycle)

            print(f"[BUZZER] Initialized on GPIO {self.pin}")
        except Exception as e:
            print(f"[BUZZER] Simulation mode: {e}")
            self.simulation = True

    def _beep(self, duration, duty=50):
        """
        Beep for `duration` seconds with `duty`% volume.
        Always cuts off cleanly — no stuck buzzer.
        """
        if self.simulation:
            print(f"[BUZZER] BEEP {duration:.1f}s")
            time.sleep(duration)
            return
        try:
            self.pwm.ChangeDutyCycle(duty)
            time.sleep(duration)
            self.pwm.ChangeDutyCycle(0)   # Silence — always reached
        except Exception as e:
            print(f"[BUZZER] Error: {e}")
            try:
                self.pwm.ChangeDutyCycle(0)
            except Exception:
                pass

    # ── Public beep patterns ─────────────────────────────────────────────────

    def beep_present(self):
        """One clean short beep — On-Time check-in / normal checkout."""
        self._beep(0.25)

    def beep_late(self):
        """Two quick beeps — Late Arrived or Early Leaving."""
        self._beep(0.15)
        time.sleep(0.08)
        self._beep(0.15)

    def beep_unknown(self):
        """Three short sharp beeps — Unknown face detected."""
        for _ in range(3):
            self._beep(0.1)
            time.sleep(0.08)

    def cleanup(self):
        """Always call this on shutdown to silence the buzzer."""
        try:
            if self.pwm:
                self.pwm.ChangeDutyCycle(0)
                self.pwm.stop()
            if not self.simulation:
                self.GPIO.cleanup(self.pin)
        except Exception:
            pass
