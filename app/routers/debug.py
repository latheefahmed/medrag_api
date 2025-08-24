from fastapi import APIRouter
from app.db import ensure_cosmos

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/health")
def health():
    info = ensure_cosmos()
    return {"status": "ok" if info.get("ok") else "degraded", **info}
