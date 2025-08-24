from fastapi import APIRouter, Depends, HTTPException
from app.security import get_current_user
from app.schemas import SessionCreate, SessionPatch
from app.db import (
    list_sessions_for_user, create_session, get_session_by_id,
    update_session, delete_session,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.get("")
async def list_my_sessions(user = Depends(get_current_user)):
    return list_sessions_for_user(user_id=user["email"])

@router.post("")
async def create_my_session(body: SessionCreate, user = Depends(get_current_user)):
    return create_session({
        "user_id": user["email"],
        "title": body.title,
        "messages": body.messages or [],
        "meta": body.meta or {},
    })

@router.get("/{session_id}")
async def get_my_session(session_id: str, user = Depends(get_current_user)):
    doc = get_session_by_id(session_id, user["email"])
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return doc

@router.patch("/{session_id}")
async def patch_my_session(session_id: str, body: SessionPatch, user = Depends(get_current_user)):
    patch = {k: v for k, v in body.dict().items() if v is not None}
    doc = update_session(session_id, user["email"], patch)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return doc

@router.delete("/{session_id}")
async def delete_my_session(session_id: str, user = Depends(get_current_user)):
    ok = delete_session(session_id, user["email"])
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}
