import asyncio
import serial
import time
from app.helpers import handle_detection, send_result_to_arduino
from app.config import settings
import os


SERIAL_PORT = settings.arduino_serial_port
BAUD_RATE = 9600
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)


def connect_serial():
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print("[Serial] Connected to Arduino. Listening...")
            return ser
        except serial.SerialException as e:
            print(f"[Serial] Failed to connect: {e}. Retrying in 2s...")
            time.sleep(2)


async def serial_listener_task():
    ser = connect_serial()

    while True:
        try:
            line = ser.readline().decode().strip()
            if not line:
                continue

            print(f"[Serial] Received from Arduino: {line}")

            if line == "ALERT":
                result = handle_detection()
                if result:
                    final_group = result["group"]
                    confidence = result["confidence"]
                    message = f"{final_group}:{confidence:.2f}"

                    send_result_to_arduino(final_group, arduino=ser)
                    print(f"[Serial] Sent to Arduino: {message}")

        except (serial.SerialException, OSError) as e:
            print(f"[Serial Listen Error] {e}")
            try:
                ser.close()
            except:
                pass
            print("[Serial] Reconnecting...")
            await asyncio.sleep(2)
            ser = connect_serial()

        except Exception as e:
            print(f"[Serial Listen Error] Unexpected error: {e}")
            await asyncio.sleep(1)