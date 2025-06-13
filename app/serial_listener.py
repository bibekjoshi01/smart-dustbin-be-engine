import asyncio
import serial
from app.helpers import send_result_to_arduino, handle_capture_image
from ml.inference import make_inference
from ml.models import get_models
from app.config import settings
import os


SERIAL_PORT = settings.arduino_serial_port
BAUD_RATE = 9600
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)


def handle_detection():
    frame = handle_capture_image()
    if frame is None:
        print("[Detection] No frame to process.")
        return None

    densenet, yolo = get_models()
    prediction = make_inference(
        densenet, yolo, frame
    )

    return prediction


async def serial_listener_task():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("[Serial] Connected to Arduino. Listening...")
    except serial.SerialException as e:
        print(f"[Serial Error] Failed to open port: {e}")
        return

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

                    send_result_to_arduino(final_group)
                    print(f"[Serial] Sent to Arduino: {message}")

        except Exception as e:
            print(f"[Serial Listen Error] {e}")
            await asyncio.sleep(1)  # Prevent busy-loop on error
