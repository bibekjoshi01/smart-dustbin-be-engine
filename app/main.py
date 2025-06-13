import asyncio
from fastapi import FastAPI
from app import models
from app.config import settings
from app.api.endpoints import router as api_router
from app.api.auth import router as auth_router
from contextlib import asynccontextmanager
from .middleware import AuthMiddleware
from .database import engine

from app.serial_listener import serial_listener_task


import threading

stop_event = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(
        target=serial_listener_task, args=(stop_event,), daemon=True
    )
    thread.start()
    print("Serial listener started ✌🏻")
    yield
    stop_event.set()
    thread.join()
    print("🛑 Serial listener stopped")


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(AuthMiddleware)

models.Base.metadata.create_all(bind=engine)

app.include_router(api_router, tags=["API"])
app.include_router(auth_router)


@app.get("/health", tags=["Site"])
async def health_check():
    return {"status": "ok"}
