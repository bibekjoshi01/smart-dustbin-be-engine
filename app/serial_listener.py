import serial
import time
import asyncio
from app.helpers import handle_detection, send_result_to_arduino
from app.config import settings
from app.websocket_manager import manager  

SERIAL_PORT = settings.arduino_serial_port
BAUD_RATE = 9600

def connect_serial():
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print("[Serial] Connected to Arduino. Listening...")
            return ser
        except serial.SerialException as e:
            print(f"[Serial] Failed to connect: {e}. Retrying in 2s...")
            time.sleep(2)


def serial_listener_task(stop_event):
    ser = connect_serial()

    while not stop_event.is_set():
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
                    image_base64 = result.get("image")  

                    message = f"{final_group}:{confidence:.2f}"
                    send_result_to_arduino(final_group, arduino=ser)
                    print(f"[Serial] Sent to Arduino: {message}")

                    # Send full detection data to frontend via WebSocket
                    asyncio.run(manager.broadcast({
                        "group": final_group,
                        "confidence": confidence,
                        "image": image_base64,
                    }))

        except (serial.SerialException, OSError) as e:
            print(f"[Serial Listen Error] {e}")
            try:
                ser.close()
            except:
                pass
            print("[Serial] Reconnecting...")
            time.sleep(2)

        except Exception as e:
            print(f"[Serial Listen Error] Unexpected error: {e}")
            time.sleep(1)
