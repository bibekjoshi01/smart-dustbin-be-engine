from fastapi import FastAPI

from app.config import settings
from app.api.endpoints import router as api_router


app = FastAPI(title=settings.app_name, version=settings.app_version)

app.include_router(api_router, tags=["IOT"])

@app.get("/health", tags=["Site"])
async def health_check():
    return {"status": "ok"}
