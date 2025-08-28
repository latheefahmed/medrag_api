from fastapi import APIRouter, Depends, HTTPException
from app.security import require_admin
from app.db import list_users, list_sessions_all

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/users")
def admin_users(user=Depends(require_admin)):
    items = list_users(limit=200)
    return {"items": items}

@router.get("/sessions")
def admin_sessions(user=Depends(require_admin)):
    items = list_sessions_all(limit=200)
    return {"items": items}

@router.get("/usage")
def admin_usage(user=Depends(require_admin)):
    # Minimal stub; expand later with real counts
    return {"ok": True, "metrics": {"users_sampled":  min(200, len(list_users())), "sessions_sampled": min(200, len(list_sessions_all()))}}
