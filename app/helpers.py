import cv2
import os
import time
import serial
from .config import settings


MOBILE_VIDEO_STREAM = settings.mobile_video_stream
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)


arduino = None
SERIAL_PORT = settings.arduino_serial_port
BAUD_RATE = 9600


def send_result_to_arduino(pred_group: str):
    global arduino

    if pred_group not in {"O", "R"}:
        print(f"[Serial] Invalid group '{pred_group}' — not sending.")
        return

    try:
        if arduino is None or not arduino.is_open:
            arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)  # Wait for Arduino to be ready

        command = b"L\n" if pred_group == "O" else b"R\n"
        arduino.write(command)
        print(f"[Serial] Sent control command: {command.strip().decode()}")

    except serial.SerialException as e:
        print(f"[Serial Error] Connection issue: {e}")
    except Exception as e:
        print(f"[Serial Error] Unexpected: {e}")


def handle_capture_image():
    try:
        cap = cv2.VideoCapture(MOBILE_VIDEO_STREAM)
        if not cap.isOpened():
            print("[Camera] Failed to open video stream.")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("[Camera] Failed to read frame.")
            return None

        print("[Camera] Frame captured successfully.")
        return frame  # 🔁 Return image as NumPy array

    except Exception as e:
        print(f"[Camera Error] {e}")
        return None

