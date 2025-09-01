from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


    APP_NAME: str = "MedRAG API"
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    CORS_ORIGINS: str = "http://localhost:3000"


    JWT_SECRET: str = Field(default="change-me", alias="JWT_SECRET")
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # default 60m local


    COOKIE_SECURE: bool = Field(default=False, alias="COOKIE_SECURE")
    COOKIE_SAMESITE: str = Field(default="Lax", alias="COOKIE_SAMESITE")


    COSMOS_URL: Optional[str] = None
    COSMOS_KEY: Optional[str] = None
    COSMOS_DB: str = "medrag"
    COSMOS_USERS_CONTAINER: str = "users"
    COSMOS_SESSIONS_CONTAINER: str = "sessions"
    COSMOS_LOGS_CONTAINER: str = "logs"


    CONTACT_EMAIL: str = "you@example.com"


    EMAIL_PROVIDER: str = Field(default="console", alias="EMAIL_PROVIDER")  # "smtp" or "console"
    EMAIL_FROM: str = Field(default="MedRAG <no-reply@medrag.local>", alias="EMAIL_FROM")

    SMTP_HOST: Optional[str] = Field(default=None, alias="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, alias="SMTP_PORT")
    SMTP_USER: Optional[str] = Field(default=None, alias="SMTP_USER")
    SMTP_PASS: Optional[str] = Field(default=None, alias="SMTP_PASS")
    SMTP_USE_TLS: bool = Field(default=True, alias="SMTP_USE_TLS")  # STARTTLS

    USE_EMBED_CACHE: int = 1
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "medrag_pmid_768"
    EMBED_DIM: int = 768


settings = Settings()
