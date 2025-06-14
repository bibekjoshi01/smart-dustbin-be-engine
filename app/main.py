import threading
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app import models
from app.config import settings
from app.database import engine
from app.api.endpoints import router as api_router
from app.api.auth import router as auth_router
from app.middleware import AuthMiddleware
from app.serial_listener import serial_listener_task
from app.api.websocket_endpoint import router as websocket_router


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
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)

# routers
app.include_router(websocket_router)
app.include_router(api_router, tags=["API"])
app.include_router(auth_router)


@app.get("/health", tags=["Site"])
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")
