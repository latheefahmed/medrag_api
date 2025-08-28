import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

COOKIE_NAME = "access_token"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "Lax")
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None,
                        token_type: str = "bearer", minutes: Optional[int] = None) -> str:
    to_encode = data.copy()
    if minutes is not None:
        expires_delta = timedelta(minutes=minutes)
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": token_type})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _read_token_from_request(request: Request) -> Optional[str]:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        return token
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1]
    return None


async def get_current_user(request: Request) -> dict:
    token = _read_token_from_request(request) or await oauth2_scheme(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "bearer":
            raise HTTPException(status_code=401, detail="Invalid token type")
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Missing subject")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    from .db import get_user_by_email
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_optional_user(request: Request) -> Optional[dict]:
    try:
        return await get_current_user(request)
    except HTTPException:
        return None


def set_auth_cookie(response, token: str):
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=COOKIE_MAX_AGE,
        path="/",
    )


def clear_auth_cookie(response):
    response.delete_cookie(COOKIE_NAME, path="/")
