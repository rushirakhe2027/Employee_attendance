import time
from RPLCD.i2c import CharLCD
import smbus2

class LCDModule:
    def __init__(self, address=0x27, port=1, cols=16, rows=2):
        self.address = address
        self.port = port
        self.cols = cols
        self.rows = rows
        self.lcd = None
        self.initialize()

    def initialize(self):
        try:
            # Try to detect LCD
            bus = smbus2.SMBus(self.port)
            try:
                bus.read_byte(self.address)
                print(f"LCD found at 0x{self.address:02x}")
            except Exception:
                # Try another common address
                for alt_addr in [0x3F, 0x20]:
                    try:
                        bus.read_byte(alt_addr)
                        self.address = alt_addr
                        break
                    except Exception:
                        continue
            
            self.lcd = CharLCD(i2c_expander='PCF8574', address=self.address, 
                               port=self.port, cols=self.cols, rows=self.rows)
            self.lcd.clear()
            print("LCD Initialized")
        except Exception as e:
            print(f"LCD Simulation Mode: {e}")
            self.lcd = None

    def display(self, line1="", line2=""):
        if self.lcd:
            try:
                self.lcd.clear()
                self.lcd.write_string(line1[:16])
                if line2:
                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(line2[:16])
            except Exception:
                pass
        else:
            print(f"[LCD SIMULATION] L1: {line1} | L2: {line2}")


    def clear(self):
        if self.lcd:
            self.lcd.clear()
