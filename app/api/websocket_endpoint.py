from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.websocket_manager import manager
import asyncio


router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)  # Just keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
