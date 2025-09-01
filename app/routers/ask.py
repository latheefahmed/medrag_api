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
            "score": d.get("score") or (d.get("scores") or {}).get("fused_raw"),
            "url": url,
            "abstract": d.get("abstract"),
        })
    return [r for r in out if r.get("title") or r.get("pmid")]

def _assistant_text_from_summary(summary: Dict[str, Any], docs: List[Dict[str, Any]]) -> str:
    if not isinstance(summary, dict) or "answer" not in summary:
        return "No summary generated."
    a = summary.get("answer") or {}
    conclusion = (a.get("conclusion") or "").strip()
    kf = [s for s in (a.get("key_findings") or []) if s]
    ql = [s for s in (a.get("quality_and_limits") or []) if s]
    cites = a.get("evidence_citations") or []

    lines: List[str] = []
    if conclusion:
        lines.append(conclusion)
    if kf:
        lines.append("")
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
        links = {str(k): v for k, v in (summary.get("citation_links") or {}).items()}
        for n in cites:
            try:
                idx = int(n)
            except Exception:
                continue
            title = docs[idx-1]["title"] if 1 <= idx <= len(docs) else f"Reference {idx}"
            url = links.get(str(idx)) or (docs[idx-1].get("url") if 1 <= idx <= len(docs) else "")
            lines.append(f"[{idx}] {title}{(' - ' + url) if url else ''}")
    return "\n".join([s for s in lines if s is not None])

def _right_pane_from_docs(docs: List[Dict[str, Any]], tried: List[Dict[str, str]], timings: Dict[str,int], overview=None, intent=None):
    results = [{
        "pmid": d.get("pmid"),
        "title": d.get("title"),
        "journal": d.get("journal"),
        "year": d.get("year"),
        "url": d.get("url"),
        "score": (d.get("scores") or {}).get("fused_raw") if d.get("scores") else d.get("score"),
        "abstract": d.get("abstract"),
    } for d in docs]
    return {
        "results": results,
        "booleans": [{"group": b.get("label",""), "query": b.get("query",""), "note": b.get("note")} for b in (tried or [])],
        "plan": {"chunks": [], "time_tags": [], "exclusions": []},
        "overview": overview,
        "timings": timings,
        "intent": intent or {},
    }

def _collect_history_payload(sess_doc: Dict[str,Any]) -> Dict[str,Any]:
    """
    Build a compact history object for rag.resolve_context:
    - last ~3 turns of messages
    - prior docs from the last 1–2 assistant messages (their 'references')
    """
    msgs = list(sess_doc.get("messages") or [])
    recent = msgs[-6:] if msgs else []  # ~3 turns
    prior_docs: List[Dict[str,Any]] = []
    # walk backward, collect references from the last two assistant messages
    seen_pmids = set()
    cnt = 0
    for m in reversed(recent):
        if m.get("role") != "assistant":
            continue
        refs = m.get("references") or []
        for r in refs:
            pmid = str(r.get("pmid") or "").strip()
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                prior_docs.append({
                    "pmid": pmid,
                    "title": r.get("title"),
                    "journal": r.get("journal"),
                    "year": r.get("year"),
                    "abstract": r.get("abstract"),
                    "score": r.get("score"),
                    "url": r.get("url"),
                    "from_label": "history"
                })
        cnt += 1
        if cnt >= 2:
            break
    return {
        "recent_messages": recent,
        "prior_docs": prior_docs[:8],
    }

@router.post("")
async def ask(body: AskBody, user=Depends(get_current_user)):
    # ---------- auth ----------
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = user.get("user_id") or user.get("id") or user.get("email")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    # ---------- payload ----------
    session_id = body.session_id or body.id or str(uuid4())
    question = (body.q or body.question or body.text or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question is required")
    role = (body.role_override or user.get("role") or "").strip() or None

    # ---------- ensure session ----------
    sess = get_session_by_id(session_id, user_id)
    if not sess:
        sess = create_session({
            "id": session_id,
            "user_id": user_id,
            "title": "Untitled",
            "messages": [],
            "meta": {},
        })

    # Build history BEFORE appending current user message
    history_payload = _collect_history_payload(sess)

    # append user msg immediately
    user_msg = {"id": str(uuid4()), "role": "user", "content": question, "ts": _now_ms()}
    messages: List[Dict[str, Any]] = (sess.get("messages") or []) + [user_msg]

    # ---------- RETRIEVAL + ANSWER (synchronous; fast) ----------
    t0 = time.perf_counter()
    q_norm = rag.norm(question); lo, hi = rag.time_tags(q_norm); ex = rag.parse_not(q_norm)

    # intent + guardrail
    ctx = rag.resolve_context(q_norm, history_payload)
    mode = rag.detect_mode(q_norm)
    guard = rag.guardrail_domain(q_norm)
    if not guard.get("domain_relevant", False) and not ctx.follow_up:
        right_pane = _right_pane_from_docs([], [], {"retrieve_ms": 0, "summarize_ms": 0}, overview={
            "conclusion": "The query didn’t look biomedical. Try clinical/biomedical terms.",
            "key_findings": [],
            "quality_and_limits": []
        }, intent={"follow_up": ctx.follow_up, "reason": ctx.reason, "mode": mode})
        assistant_msg = {
            "id": str(uuid4()),
            "role": "assistant",
            "content": "We can’t search relevant biomedical terms for this question.",
            "ts": _now_ms(),
            "references": [],
        }
        messages.append(assistant_msg)
        update_session(session_id, user_id, {"messages": messages, "rightPane": right_pane})
        return {"session_id": session_id, "message": assistant_msg, "rightPane": right_pane}

    # pipeline
    pipe = rag.run_pipeline(question, role=role, history=history_payload)
    docs_payload = pipe.get("docs") or []
    timings = pipe.get("timings") or {"retrieve_ms": 0, "summarize_ms": 0}
    intent_payload = pipe.get("intent") or {}
    right_pane = _right_pane_from_docs(docs_payload, pipe.get("booleans") or [], timings, overview=None, intent=intent_payload)

    # answer text
    summary_obj = pipe.get("summary") or {}
    assistant_text = _assistant_text_from_summary(summary_obj, right_pane["results"])
    references = _mk_references_from_rightpane(right_pane)

    assistant_msg = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": assistant_text,
        "ts": _now_ms(),
        "references": references,
    }
    messages.append(assistant_msg)

    update_session(session_id, user_id, {
        "messages": messages,
        "meta": sess.get("meta") or {},
        "rightPane": right_pane
    })

    # log (non-blocking best-effort)
    try:
        create_log({
            "user_id": user_id,
            "session_id": session_id,
            "query": question,
            "response": assistant_text[:8000],
            "fetch_ms": int(timings.get("retrieve_ms") or 0),
            "summarize_ms": int(timings.get("summarize_ms") or 0),
            "n_results": int(len(docs_payload)),
            "ts": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        print(f"[LOGS] failed to write log: {e!r}")

    return {"session_id": session_id, "message": assistant_msg, "rightPane": right_pane}
