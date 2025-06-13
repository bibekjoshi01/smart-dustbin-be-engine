import asyncio
from fastapi import FastAPI
from app.config import settings
from app.api.endpoints import router as api_router
from contextlib import asynccontextmanager

from app.serial_listener import serial_listener_task

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(serial_listener_task())
    print("Serial listener started ✌🏻")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("🛑 Serial listener stopped")


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.include_router(api_router, tags=["API"])

@app.get("/health", tags=["Site"])
async def health_check():
    return {"status": "ok"}
