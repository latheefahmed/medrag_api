from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "MedRAG API"
    CORS_ORIGINS: str = "http://localhost:3000"

    # Auth / JWT
    JWT_SECRET: str = Field(default="change-me", alias="JWT_SECRET")
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # DB
    DB_FAKE: bool = Field(default=False, alias="DB_FAKE")
    COSMOS_ENDPOINT: Optional[str] = None
    COSMOS_KEY: Optional[str] = None
    COSMOS_DB: str = "medrag"
    COSMOS_USERS_CONTAINER: str = "users"
    COSMOS_SESSIONS_CONTAINER: str = "sessions"

settings = Settings()
