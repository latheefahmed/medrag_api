# app/routers/ask.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from uuid import uuid4
import time
from datetime import datetime

from app.security import get_current_user
from app.db import (
    get_session_by_id,
    create_session,
    update_session,
)
from app.services.rag import run_rag_pipeline

router = APIRouter(prefix="/ask", tags=["ask"])


# --------- Accept both old/new FE payloads gracefully ----------
class AskBody(BaseModel):
    session_id: Optional[str] = None  # new style
    id: Optional[str] = None          # old style
    question: Optional[str] = None
    text: Optional[str] = None
    max_tokens: Optional[int] = 512   # ignored by backend (kept for compat)
    role_override: Optional[str] = None  # optional (rare)


# --------- Helpers ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _iso_to_ms(s: Optional[str]) -> int:
    if not s:
        return _now_ms()
    try:
        # handle both "...Z" and "+00:00"
        s = s.replace("Z", "+00:00") if "Z" in s else s
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except Exception:
        return _now_ms()

def _present_session(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Map Cosmos snake_case doc → FE camelCase session."""
    return {
        "id": doc["id"],
        "title": doc.get("title") or "Untitled",
        "createdAt": doc.get("createdAt") or _iso_to_ms(doc.get("created_at")),
        "updatedAt": doc.get("updatedAt") or _iso_to_ms(doc.get("updated_at")),
        "messages": doc.get("messages") or [],
        "rightPane": doc.get("rightPane") or None,
    }


# --------- Route ----------
@router.post("")
async def ask(body: AskBody, user=Depends(get_current_user)):
    """
    Runs RAG, appends messages, saves `rightPane`, and returns updated session.
    Accepts either:
      { "session_id": "...", "question": "..." }
    or
      { "id": "...", "text": "..." }
    """
    # ---- auth/user ----
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = user.get("user_id") or user.get("id") or user.get("email")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    # ---- payload normalization ----
    session_id = body.session_id or body.id or str(uuid4())
    question = (body.question or body.text or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question is required")

    role = (body.role_override or user.get("role") or "").strip() or None

    # ---- fetch or create session ----
    sess = get_session_by_id(session_id, user_id)
    if not sess:
        # create minimal session
        sess = create_session({
            "id": session_id,
            "user_id": user_id,
            "title": "Untitled",
            "messages": [],
            "meta": {},
        })

    # ---- append user message (server-side canonical) ----
    user_msg = {
        "id": str(uuid4()),
        "role": "user",
        "content": question,
        "ts": _now_ms(),
    }
    messages: List[Dict[str, Any]] = (sess.get("messages") or []) + [user_msg]

    # ---- run your pipeline (exact logic lives in app/services/rag.py) ----
    try:
        assistant_text, right_pane = run_rag_pipeline(question, role=role, verbose=False)
    except Exception as e:
        # don't break the chat; return a soft error + echo preview
        assistant_text = f"(pipeline error) {str(e)[:400]}"
        right_pane = {
            "results": [],
            "booleans": [],
            "plan": {"chunks": [], "time_tags": [], "exclusions": []},
        }

    # ---- append assistant message ----
    assistant_msg = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": assistant_text,
        "ts": _now_ms(),
    }
    messages.append(assistant_msg)

    # ---- persist session ----
    patch = {
        "messages": messages,
        "meta": sess.get("meta") or {},
    }
    # Put the right-pane data where the FE reads it
    patch["rightPane"] = right_pane

    updated = update_session(session_id, user_id, patch)
    if not updated:
        # extremely rare: if update fails, return in-memory state so FE still renders
        updated = dict(sess)
        updated["messages"] = messages
        updated["rightPane"] = right_pane
        updated["updated_at"] = datetime.utcnow().isoformat()

    # ---- present for FE (camelCase) ----
    return _present_session(updated)
