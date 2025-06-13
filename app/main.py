from fastapi import FastAPI

from app.config import settings


app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/health", tags=["Site"])
async def health_check():
    return {"status": "ok"}
