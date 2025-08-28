# app/routers/auth.py
from fastapi import APIRouter, HTTPException, Response, Depends, Request
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError

from app.schemas import SignupInput, LoginInput
from app.security import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, get_optional_user, set_auth_cookie, clear_auth_cookie,
    ALGORITHM, SECRET_KEY,
)
from app.db import get_user_by_email, create_user, update_user
from app.settings import settings
from app.services.emailer import send_verification_email, send_reset_email

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup")
def signup(body: SignupInput, request: Request):
    email = body.email.lower().strip()
    if body.confirm_password is not None and body.password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="User already exists")
    hashed = get_password_hash(body.password)
    user = {"id": email, "email": email, "password_hash": hashed, "role": body.role or "student", "verified": False}
    create_user(user)
    try:
        send_verification_email(email, settings.FRONTEND_ORIGIN)
    except Exception:
        pass
    return {"ok": True}


@router.post("/login")
def login(body: LoginInput, response: Response):
    email = body.email.lower().strip()
    user = get_user_by_email(email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": email})
    set_auth_cookie(response, token)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"email": email, "role": user.get("role"), "verified": user.get("verified", False)},
    }


@router.get("/me")
async def me(user=Depends(get_current_user)):
    return {k: v for k, v in user.items() if k not in {"password_hash"}}


@router.post("/logout")
def logout(response: Response):
    clear_auth_cookie(response)
    return {"ok": True}


# ---------- Email verification ----------
class ResendBody(BaseModel):
    email: EmailStr | None = None


@router.post("/resend-verification")
async def resend_verification(request: Request, body: ResendBody | None = None, current=Depends(get_optional_user)):
    email = current.get("email") if current else (str(body.email).lower().strip() if body and body.email else None)
    # Always 200 to avoid enumeration
    if not email:
        return {"ok": True}
    u = get_user_by_email(email)
    if not u or u.get("verified"):
        return {"ok": True}
    try:
        send_verification_email(email, settings.FRONTEND_ORIGIN)
    except Exception:
        pass
    return {"ok": True}


@router.get("/verify")
def verify(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "verify":
            raise HTTPException(status_code=400, detail="Invalid token type")
        email = payload.get("sub")
        assert email
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    u = get_user_by_email(email)
    if not u:
        return {"ok": True}
    update_user(email, {"verified": True})
    return {"ok": True, "verified": True}


# ---------- Password reset ----------
class ResetRequestBody(BaseModel):
    email: EmailStr

class DoResetBody(BaseModel):
    token: str
    password: str

@router.post("/request-reset")
def request_reset(body: ResetRequestBody):
    # Always 200 to avoid enumeration
    try:
        email = str(body.email).lower().strip()
        if get_user_by_email(email):
            send_reset_email(email, settings.FRONTEND_ORIGIN)
    except Exception:
        pass
    return {"ok": True}

@router.post("/reset")
def do_reset(body: DoResetBody):
    try:
        payload = jwt.decode(body.token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "reset":
            raise HTTPException(status_code=400, detail="Invalid token type")
        email = payload.get("sub")
        assert email
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    u = get_user_by_email(email)
    if not u:  # still 200 for safety
        return {"ok": True}
    update_user(email, {"password_hash": get_password_hash(body.password)})
    return {"ok": True}
