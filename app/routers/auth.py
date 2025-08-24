from fastapi import APIRouter, HTTPException, Response, Depends
from azure.cosmos import exceptions as cosmos_exc
from app.schemas import SignupInput, LoginInput
from app.security import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, set_auth_cookie, clear_auth_cookie,
)
from app.db import get_user_by_email, create_user

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/signup")
def signup(body: SignupInput):
    email = body.email.lower().strip()

    if body.confirm_password is not None and body.password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    try:
        if get_user_by_email(email):
            raise HTTPException(status_code=409, detail="User already exists")

        hashed = get_password_hash(body.password)
        user = {"id": email, "email": email, "password_hash": hashed, "role": body.role or "student"}
        create_user(user)
        return {"ok": True}

    except cosmos_exc.CosmosResourceExistsError:
        raise HTTPException(status_code=409, detail="User already exists")
    except cosmos_exc.CosmosHttpResponseError as e:
        # common: Partition key mismatch, 401 auth errors, etc.
        raise HTTPException(status_code=500, detail=f"Cosmos error: {e.message or str(e)}")
    except RuntimeError as e:
        # our db layer raises this if env is missing
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@router.post("/login")
def login(body: LoginInput, response: Response):
    email = body.email.lower().strip()
    user = get_user_by_email(email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": email})
    set_auth_cookie(response, token)
    return {"access_token": token, "token_type": "bearer", "user": {"email": email, "role": user.get("role")}}

@router.get("/me")
async def me(user = Depends(get_current_user)):
    return {k: v for k, v in user.items() if k not in {"password_hash"}}

@router.post("/logout")
def logout(response: Response):
    clear_auth_cookie(response)
    return {"ok": True}
