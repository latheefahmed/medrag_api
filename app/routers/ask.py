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

# ---------- helpers ----------

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

def _mk_references(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in results[:20]:
        pmid = str(d.get("pmid") or d.get("id") or "").strip() or None
        url = d.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None)
        out.append({
            "pmid": pmid,
            "title": d.get("title") or "",
            "journal": d.get("journal"),
            "year": d.get("year"),
            "score": (d.get("scores") or {}).get("fused_raw") if d.get("scores") else d.get("score"),
            "url": url,
            "abstract": d.get("abstract"),
        })
    return [r for r in out if (r.get("title") or r.get("pmid"))]

def _collect_history_payload(sess_doc: Dict[str,Any]) -> Dict[str,Any]:
    """
    Keep ONLY the last 3 assistant turns that had references.
    Guardrail messages (no references) are ignored.
    """
    msgs = list(sess_doc.get("messages") or [])
    recent: List[Dict[str,Any]] = []
    prior_docs: List[Dict[str,Any]] = []
    seen_pmids = set()

    i = len(msgs) - 1
    found = 0
    while i >= 0 and found < 3:
        m = msgs[i]
        if m.get("role") == "assistant" and (m.get("references") or []):
            if i-1 >= 0 and msgs[i-1].get("role") == "user":
                recent.append(msgs[i-1])
            recent.append(m)
            for r in (m.get("references") or []):
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
            found += 1
        i -= 1

    return {"recent_messages": recent[-6:], "prior_docs": prior_docs[:8]}

def _assistant_text_from_summary(summary: Dict[str, Any], docs: List[Dict[str, Any]]) -> str:
    if not isinstance(summary, dict) or "answer" not in summary:
        return "No summary generated."
    a = summary.get("answer") or {}
    plain = (a.get("plain") or "").strip()
    if plain:
        cites = a.get("evidence_citations") or []
        links = {str(k): v for k, v in (summary.get("citation_links") or {}).items()}
        if cites:
            plain += "\n\nCitations:\n" + "\n".join([f"{i}. {links.get(str(i),'')}" for i in cites])
        return plain

    # structured case
    conclusion = (a.get("conclusion") or "").strip()
    kf = [s for s in (a.get("key_findings") or []) if s]
    ql = [s for s in (a.get("quality_and_limits") or []) if s]
    cites = a.get("evidence_citations") or []
    links = {str(k): v for k, v in (summary.get("citation_links") or {}).items()}

    lines: List[str] = []
    if conclusion: lines.append(conclusion)
    if kf:
        lines += ["", "Key findings:"]
        for i, item in enumerate(kf, 1):
            lines.append(f"{i}. {item}")
    if ql:
        lines += ["", "Quality and limitations:"]
        for i, item in enumerate(ql, 1):
            lines.append(f"{i}. {item}")
    if cites:
        lines += ["", "Citations:"]
        for n in cites:
            try:
                idx = int(n)
            except Exception:
                continue
            title = docs[idx-1]["title"] if 1 <= idx <= len(docs) else f"Reference {idx}"
            url = links.get(str(idx)) or (docs[idx-1].get("url") if 1 <= idx <= len(docs) else "")
            lines.append(f"{idx}. {title} {('- ' + url) if url else ''}")
    return "\n".join(lines)

# ---------- main route ----------

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

    sess = get_session_by_id(session_id, user_id)
    if not sess:
        sess = create_session({
            "id": session_id,
            "user_id": user_id,
            "title": "Untitled",
            "messages": [],
            "meta": {},
        })

    # Build memory (3 citation-bearing turns; guardrails excluded)
    history_payload = _collect_history_payload(sess)

    # Append user message
    user_msg = {"id": str(uuid4()), "role": "user", "content": question, "ts": _now_ms()}
    messages: List[Dict[str, Any]] = (sess.get("messages") or []) + [user_msg]

    # Run the RAG pipeline (sync)
    pipe = rag.run_pipeline(question, role=role, history=history_payload)

    # Prepare right-pane + references
    results = [{
        "pmid": d.get("pmid"), "title": d.get("title"), "journal": d.get("journal"),
        "year": d.get("year"), "url": d.get("url"),
        "score": (d.get("scores") or {}).get("fused_raw") if d.get("scores") else None,
        "abstract": d.get("abstract")
    } for d in (pipe.get("docs") or [])]
    right_pane = {
        "results": results,
        "booleans": [{"group": b.get("label",""), "query": b.get("query",""), "note": ""} for b in (pipe.get("booleans") or [])],
        "plan": {"chunks": (pipe.get("plan") or {}).get("chunks", []),
                 "time_tags": pipe.get("time_tags", []),
                 "exclusions": pipe.get("exclusions", [])},
        "overview": (pipe.get("summary") or {}).get("answer") or {},
        "timings": pipe.get("timings") or {"retrieve_ms": 0, "summarize_ms": 0},
        "intent": pipe.get("intent") or {},
    }
    references = _mk_references(results)

    # Build assistant message text (citations included when present)
    assistant_text = _assistant_text_from_summary(pipe.get("summary") or {}, results)

    assistant_msg = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": assistant_text,
        "ts": _now_ms(),
        "references": references,  # empty when guardrail fired
    }
    messages.append(assistant_msg)

    update_session(session_id, user_id, {"messages": messages, "rightPane": right_pane})

    # (optional) log
    try:
        create_log({
            "user_id": user_id,
            "session_id": session_id,
            "query": question,
            "response": assistant_text[:8000],
            "fetch_ms": int((pipe.get("timings") or {}).get("retrieve_ms") or 0),
            "summarize_ms": int((pipe.get("timings") or {}).get("summarize_ms") or 0),
            "n_results": int(len(results)),
            "ts": datetime.utcnow().isoformat(),
        })
    except Exception:
        pass

    return {"session_id": session_id, "message": assistant_msg, "rightPane": right_pane}
