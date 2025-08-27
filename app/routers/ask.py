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

# Import the whole module so we can support BOTH run_rag_pipeline(...) and run_pipeline(...)
import app.services.rag as rag

router = APIRouter(prefix="/ask", tags=["ask"])

# --------- Accept both old/new FE payloads gracefully ----------
class AskBody(BaseModel):
    session_id: Optional[str] = None  # new style
    id: Optional[str] = None          # old style
    q: Optional[str] = None           # new style (preferred by FE hook)
    question: Optional[str] = None    # legacy
    text: Optional[str] = None        # legacy
    max_tokens: Optional[int] = 512   # ignored by backend (compat only)
    role_override: Optional[str] = None  # optional (rare)


# --------- Helpers ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _iso_to_ms(s: Optional[str]) -> int:
    if not s:
        return _now_ms()
    try:
        s = s.replace("Z", "+00:00") if "Z" in s else s
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except Exception:
        return _now_ms()

def _mk_references_from_rightpane(rp: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """
    Normalize rightPane.results -> message.references[]
    """
    if not rp:
        return []
    results = rp.get("results") or rp.get("final_docs") or rp.get("documents") or []
    out: List[Dict[str, Any]] = []
    for d in results[:20]:  # cap to keep chat payload small
        pmid = str(d.get("pmid") or d.get("id") or "").strip() or None
        url = d.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None)
        out.append({
            "pmid": pmid,
            "title": d.get("title") or "",
            "journal": d.get("journal"),
            "year": d.get("year"),
            "score": d.get("score") or d.get("fused_raw"),
            "url": url,
            "abstract": d.get("abstract"),
        })
    return [r for r in out if r.get("title") or r.get("pmid")]


def _present_session(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Map Cosmos snake_case doc → FE camelCase session (not used as /ask response, but handy for debugging)."""
    return {
        "id": doc["id"],
        "title": doc.get("title") or "Untitled",
        "createdAt": doc.get("createdAt") or _iso_to_ms(doc.get("created_at")),
        "updatedAt": doc.get("updatedAt") or _iso_to_ms(doc.get("updated_at")),
        "messages": doc.get("messages") or [],
        "rightPane": doc.get("rightPane") or None,
    }


def _compute_answer(question: str, role: Optional[str]) -> tuple[str, Dict[str, Any]]:
    """
    Call the available RAG entrypoint:
      - prefer rag.run_rag_pipeline(q, role, verbose=False) -> (assistant_text, right_pane)
      - else rag.run_pipeline(q) -> we adapt its output into the same pair
    """
    # Case 1: modern wrapper exists
    if hasattr(rag, "run_rag_pipeline"):
        return rag.run_rag_pipeline(question, role=role, verbose=False)

    # Case 2: only run_pipeline exists – adapt to (assistant_text, right_pane)
    pipe = rag.run_pipeline(question)

    # Build assistant text from structured summary if present
    summary = pipe.get("summary") or {}
    heading = (role or "Answer").replace("_", " ").title()
    if isinstance(summary, dict) and "answer" in summary:
        a = summary.get("answer") or {}
        conclusion = a.get("conclusion") or ""
        kf = a.get("key_findings") or []
        ql = a.get("quality_and_limits") or []
        cites = a.get("evidence_citations") or []
        lines = [f"### {heading}", conclusion.strip()]
        if kf:
            lines.append("\n**Key findings**")
            for item in kf:
                lines.append(f"- {item}")
        if ql:
            lines.append("\n**Quality & limitations**")
            for item in ql:
                lines.append(f"- {item}")
        if cites:
            lines.append("\n**Citations**")
            lines.append(", ".join([f"[{c}]" for c in cites]))
        assistant_text = "\n".join([s for s in lines if s and s.strip()])
    else:
        assistant_text = f"### {heading}\n(no summary generated)"

    # Build rightPane-like object
    final_docs = pipe.get("final_docs") or pipe.get("docs") or []
    def pmid_url(pmid: str | None) -> Optional[str]:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
    results = [{
        "pmid": d.get("pmid"),
        "title": d.get("title"),
        "journal": d.get("journal"),
        "year": d.get("year"),
        "url": d.get("url") or pmid_url(d.get("pmid")),
        "score": (d.get("scores") or {}).get("fused_raw") if d.get("scores") else d.get("score"),
        "abstract": d.get("abstract"),
    } for d in final_docs]

    booleans = [{"group": b.get("chunk",""), "query": b.get("boolean",""), "note": b.get("note")} for b in (pipe.get("buckets") or [])]
    overview = None
    if isinstance(summary, dict) and "answer" in summary:
        overview = {
            "conclusion": (summary["answer"] or {}).get("conclusion",""),
            "key_findings": (summary["answer"] or {}).get("key_findings", []),
            "quality_and_limits": (summary["answer"] or {}).get("quality_and_limits", []),
        }
    right_pane = {
        "results": results,
        "booleans": booleans,
        "plan": {
            "chunks": (pipe.get("plan") or {}).get("chunks", []),
            "time_tags": pipe.get("time_tags", []),
            "exclusions": pipe.get("exclusions", []),
        },
        "overview": overview,
    }
    return assistant_text, right_pane


# --------- Route ----------
@router.post("")
async def ask(body: AskBody, user=Depends(get_current_user)):
    """
    Runs RAG, appends messages (including references[] on AI msg), saves rightPane,
    and returns a compact payload the FE hook expects:

      { session_id, message: {id, role, content, ts, references[]}, rightPane }
    """
    # ---- auth/user ----
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = user.get("user_id") or user.get("id") or user.get("email")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    # ---- payload normalization ----
    session_id = body.session_id or body.id or str(uuid4())
    question = (body.q or body.question or body.text or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question is required")

    role = (body.role_override or user.get("role") or "").strip() or None

    # ---- fetch or create session ----
    sess = get_session_by_id(session_id, user_id)
    if not sess:
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

    # ---- run pipeline ----
    try:
        assistant_text, right_pane = _compute_answer(question, role=role)
    except Exception as e:
        assistant_text = f"(pipeline error) {str(e)[:400]}"
        right_pane = {"results": [], "booleans": [], "plan": {"chunks": [], "time_tags": [], "exclusions": []}}

    # ---- build references from rightPane ----
    references = _mk_references_from_rightpane(right_pane)

    # ---- append assistant message (with references) ----
    assistant_msg = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": assistant_text,
        "ts": _now_ms(),
        "references": references,
    }
    messages.append(assistant_msg)

    # ---- persist session ----
    patch = {
        "messages": messages,
        "meta": sess.get("meta") or {},
        "rightPane": right_pane,  # FE reads this for the right pane
    }
    updated = update_session(session_id, user_id, patch)
    if not updated:
        # Soft fallback so FE still renders
        updated = dict(sess)
        updated["messages"] = messages
        updated["rightPane"] = right_pane
        updated["updated_at"] = datetime.utcnow().isoformat()

    # ---- return compact FE payload ----
    return {
        "session_id": session_id,
        "message": assistant_msg,
        "rightPane": right_pane,
    }
