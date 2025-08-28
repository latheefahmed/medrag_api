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
    create_log,
)

import app.services.rag as rag

router = APIRouter(prefix="/ask", tags=["ask"])

# --------- Accept both old/new FE payloads gracefully ----------
class AskBody(BaseModel):
    session_id: Optional[str] = None
    id: Optional[str] = None
    q: Optional[str] = None
    question: Optional[str] = None
    text: Optional[str] = None
    max_tokens: Optional[int] = 512
    role_override: Optional[str] = None


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
    if not rp:
        return []
    results = rp.get("results") or rp.get("final_docs") or rp.get("documents") or []
    out: List[Dict[str, Any]] = []
    for d in results[:20]:
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
    return {
        "id": doc["id"],
        "title": doc.get("title") or "Untitled",
        "createdAt": doc.get("createdAt") or _iso_to_ms(doc.get("created_at")),
        "updatedAt": doc.get("updatedAt") or _iso_to_ms(doc.get("updated_at")),
        "messages": doc.get("messages") or [],
        "rightPane": doc.get("rightPane") or None,
    }


def _build_response_from_pipe(pipe: Dict[str, Any], role: Optional[str]) -> tuple[str, Dict[str, Any], Dict[str,int], int]:
    """
    Build (assistant_text, right_pane, timings, n_results) from rag.run_pipeline output.
    The assistant_text is PLAIN TEXT (no markdown symbols or bullets).
    """
    summary = pipe.get("summary") or {}
    docs = pipe.get("docs") or []
    timings = pipe.get("timings") or {"retrieve_ms": 0, "summarize_ms": 0}
    n_results = len(docs)

    # Assistant text (plain)
    if isinstance(summary, dict) and "answer" in summary:
        a = summary.get("answer") or {}
        conclusion = (a.get("conclusion") or "").strip()
        kf = [s for s in (a.get("key_findings") or []) if s]
        ql = [s for s in (a.get("quality_and_limits") or []) if s]
        cites = a.get("evidence_citations") or []

        lines: List[str] = []
        if conclusion:
            lines.append(conclusion)
        if kf:
            lines.append("")  # blank line
            lines.append("Key findings:")
            for i, item in enumerate(kf, 1):
                lines.append(f"{i}. {item}")
        if ql:
            lines.append("")
            lines.append("Quality and limitations:")
            for i, item in enumerate(ql, 1):
                lines.append(f"{i}. {item}")
        if cites:
            lines.append("")
            lines.append("Citations:")
            # include title + link for each cited index if we have it
            links = {str(k): v for k, v in (summary.get("citation_links") or {}).items()}
            for n in cites:
                try:
                    idx = int(n)
                except Exception:
                    continue
                title = docs[idx-1]["title"] if 1 <= idx <= len(docs) else f"Reference {idx}"
                url = links.get(str(idx)) or (docs[idx-1].get("url") if 1 <= idx <= len(docs) else "")
                lines.append(f"[{idx}] {title}{(' - ' + url) if url else ''}")

        assistant_text = "\n".join([s for s in lines if s is not None])
    else:
        assistant_text = "No summary generated."

    # Right pane (unchanged shape)
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
    } for d in docs]

    booleans = [{"group": b.get("label",""), "query": b.get("query",""), "note": b.get("note")} for b in (pipe.get("booleans") or [])]

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
        "timings": timings,
    }
    return assistant_text, right_pane, timings, n_results


@router.post("")
async def ask(body: AskBody, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = user.get("user_id") or user.get("id") or user.get("email")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    session_id = body.session_id or body.id or str(uuid4())
    question = (body.q or body.question or body.text or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question is required")

    role = (body.role_override or user.get("role") or "").strip() or None

    # ensure session
    sess = get_session_by_id(session_id, user_id)
    if not sess:
        sess = create_session({
            "id": session_id,
            "user_id": user_id,
            "title": "Untitled",
            "messages": [],
            "meta": {},
        })

    # append user msg
    user_msg = {"id": str(uuid4()), "role": "user", "content": question, "ts": _now_ms()}
    messages: List[Dict[str, Any]] = (sess.get("messages") or []) + [user_msg]

    # run RAG (pipeline; we want timings)
    try:
        pipe = rag.run_pipeline(question, role=role)
        assistant_text, right_pane, timings, n_results = _build_response_from_pipe(pipe, role)
    except Exception as e:
        assistant_text = f"(pipeline error) {str(e)[:400]}"
        right_pane = {"results": [], "booleans": [], "plan": {"chunks": [], "time_tags": [], "exclusions": []}, "timings": {"retrieve_ms": 0, "summarize_ms": 0}}
        timings = {"retrieve_ms": 0, "summarize_ms": 0}
        n_results = 0

    references = _mk_references_from_rightpane(right_pane)

    assistant_msg = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": assistant_text,
        "ts": _now_ms(),
        "references": references,
    }
    messages.append(assistant_msg)

    # persist session
    patch = {"messages": messages, "meta": sess.get("meta") or {}, "rightPane": right_pane}
    updated = update_session(session_id, user_id, patch)
    if not updated:
        updated = dict(sess)
        updated["messages"] = messages
        updated["rightPane"] = right_pane
        updated["updated_at"] = datetime.utcnow().isoformat()

    # ---- LOG ROW ----
    try:
        create_log({
            "user_id": user_id,
            "session_id": session_id,
            "query": question,
            "response": assistant_text,
            "fetch_ms": int(timings.get("retrieve_ms", 0)),
            "summarize_ms": int(timings.get("summarize_ms", 0)),
            "n_results": int(n_results),
            "ts": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"[LOGS] failed to write log: {e!r}")

    return {"session_id": session_id, "message": assistant_msg, "rightPane": right_pane}
