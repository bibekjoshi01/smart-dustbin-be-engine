from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Smart Dustbin"
    app_version: str = "1.0.0"
    secret_key: str = "you_secret"
    database_url: str
    mobile_video_stream: str
    arduino_serial_port: str

    model_config = SettingsConfigDict(
        env_file=".env" if Path(".env").exists() else ".env.example"
    )


settings = Settings()
