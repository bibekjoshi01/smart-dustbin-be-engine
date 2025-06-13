from fastapi import APIRouter
import cv2
import os
from datetime import datetime
from app.config import settings

router = APIRouter()

MOBILE_VIDEO_STREAM = settings.mobile_video_stream
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)


@router.get("/trigger-capture")
async def trigger_capture():
    try:
        cap = cv2.VideoCapture(MOBILE_VIDEO_STREAM)

        if not cap.isOpened():
            return {
                "status": False,
                "message": "Failed to open video stream."
            }

        # Capture single frame
        ret, frame = cap.read()
        cap.release()

        if ret:
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)

            cv2.imwrite(filepath, frame)

            return {
                "status": True,
                "message": "Image captured successfully.",
            }
        else:
            return {
                "status": False,
                "message": "Failed to capture image from stream."
            }

    except Exception as e:
        return {"status": False, "error": str(e)}
