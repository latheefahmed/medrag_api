# app/routers/ask.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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
            "score": d.get("score") or (d.get("scores") or {}).get("fused_raw"),
            "url": url,
            "abstract": d.get("abstract"),
        })
    return [r for r in out if r.get("title") or r.get("pmid")]

def _assistant_text_from_summary(summary: Dict[str, Any], docs: List[Dict[str, Any]]) -> str:
    # Always include summary/plain + sections + citations
    if not isinstance(summary, dict) or "answer" not in summary:
        return "No summary generated."
    a = summary.get("answer") or {}
    plain = (a.get("plain") or "").strip()
    conclusion = (a.get("conclusion") or "").strip()
    kf = [s for s in (a.get("key_findings") or []) if s]
    ql = [s for s in (a.get("quality_and_limits") or []) if s]
    cites = a.get("evidence_citations") or []

    lines: List[str] = []
    if plain:
        lines.append(plain)
        lines.append("")
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
    last ~3 turns (6 msgs) + references from the last 1–2 assistant messages.
    guardrail messages are excluded from context.
    """
    msgs = list(sess_doc.get("messages") or [])
    msgs = [m for m in msgs if not (m.get("flags") or {}).get("guardrail")]
    recent = msgs[-6:] if msgs else []
    prior_docs: List[Dict[str,Any]] = []
    seen_pmids = set()
    cnt = 0
    for m in reversed(recent):
        if m.get("role") != "assistant":
            continue
        if (m.get("flags") or {}).get("guardrail"):
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
async def ask(body: AskBody, background_tasks: BackgroundTasks, user=Depends(get_current_user)):
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

    # ---------- RETRIEVAL STAGE ----------
    t0 = time.perf_counter()
    q = rag.norm(question); lo, hi = rag.time_tags(q); ex = rag.parse_not(q)

    guard = rag.guardrail_domain(q)
    if not guard.get("domain_relevant", False):
        right_pane = _right_pane_from_docs([], [], {"retrieve_ms": 0, "summarize_ms": 0}, overview={
            "conclusion": "The query didn’t look biomedical. Try clinical/biomedical terms.",
            "key_findings": [],
            "quality_and_limits": []
        }, intent={"follow_up": False, "reason": guard.get("reason",""), "augmented_query": q, "chat_brief": ""})
        assistant_msg = {
            "id": str(uuid4()),
            "role": "assistant",
            "content": "We can’t search relevant biomedical terms for this question.",
            "ts": _now_ms(),
            "references": [],
            "flags": {"guardrail": True}
        }
        messages.append(assistant_msg)
        update_session(session_id, user_id, {"messages": messages, "rightPane": right_pane})
        return {"session_id": session_id, "message": assistant_msg, "rightPane": right_pane}

    # intent resolution
    ctx = rag.resolve_context(q, history_payload)
    q_eff = ctx.augmented_query if ctx.follow_up else q

    # plan + compose variants
    try:
        plan = rag.plan_tokens(q_eff)
    except Exception:
        plan = {"chunks":[t for t in rag.tok(q_eff) if t.lower() not in rag.STOP][:4],"anchors":{},"domain_relevant":True}

    variants = rag.compose(plan, lo, hi, ex, qlen=len(rag.tok(q_eff)))
    tried_booleans, merged = [], {}

    # allow direct PMID short-circuit
    for pid in rag._extract_pmids_from_query(q_eff):
        for d in rag._fetch_docs_by_pmids([pid]):
            merged[d.pmid] = d

    for v in variants:
        tried_booleans.append({"label": v["label"], "query": v["query"]})
        try:
            if not rag.esearch(v["query"], rag.CFG.pubmed.retmax_probe):  # cheap probe
                continue
            for d in rag.retrieve(v["query"], q_eff, v["label"]):
                merged[d.pmid] = d
            if len(merged) >= rag.CFG.retrieval.topk_titles:
                break
        except Exception:
            continue

    if len(merged) < rag.CFG.retrieval.min_final:
        for lv in rag.widen_variants(plan, q_eff, lo, hi, ex):
            tried_booleans.append({"label": lv["label"], "query": lv["query"]})
            try:
                if not rag.esearch(lv["query"], rag.CFG.pubmed.retmax_probe):  # cheap probe
                    continue
                for d in rag.retrieve(lv["query"], q_eff, lv["label"]):
                    if d.pmid not in merged:
                        merged[d.pmid] = d
                if len(merged) >= rag.CFG.retrieval.min_final:
                    break
            except Exception:
                continue

    # add history docs if follow-up
    if ctx.follow_up and ctx.prior_docs:
        for d in ctx.prior_docs:
            if d.pmid not in merged:
                merged[d.pmid] = d

    # detail-paper hoist (best-effort)
    used_docs_mode = "default"
    if ctx.mode == "detail_paper" and (history_payload.get("prior_docs") or []):
        idx = max(1, int(ctx.paper_index or 1))
        last_answer_refs = history_payload.get("prior_docs") or []
        try:
            pmid_wanted = None
            unique = []
            seen = set()
            for d in last_answer_refs:
                if d["pmid"] not in seen:
                    unique.append(d); seen.add(d["pmid"])
            if 1 <= idx <= len(unique):
                pmid_wanted = unique[idx-1]["pmid"]
            if pmid_wanted:
                if pmid_wanted not in merged:
                    for d in rag._fetch_docs_by_pmids([pmid_wanted]):
                        merged[d.pmid] = d
                used_docs_mode = "detail"
        except Exception:
            pass

    all_docs = list(merged.values())
    final_docs = rag.fuse_mmr(q_eff, all_docs) if all_docs else []
    used_docs = final_docs if final_docs else all_docs

    t_retrieve_end = time.perf_counter()
    timings_now = {"retrieve_ms": int((t_retrieve_end - t0)*1000), "summarize_ms": 0}

    # prepare right pane + placeholder assistant message
    docs_payload = [{
        "pmid": d.pmid, "title": d.title, "journal": d.journal, "year": d.year,
        "url": rag.pmid_url(d.pmid), "scores": d.scores, "abstract": d.abstract
    } for d in used_docs]

    intent_payload = {
        "follow_up": ctx.follow_up, "reason": ctx.reason, "augmented_query": ctx.augmented_query, "chat_brief": ctx.brief,
        "mode": ctx.mode, "include_citations": True, "paper_index": ctx.paper_index
    }
    right_pane_initial = _right_pane_from_docs(docs_payload, tried_booleans, timings_now, overview=None, intent=intent_payload)
    references_initial = _mk_references_from_rightpane(right_pane_initial)

    assistant_msg = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": "",  # to be filled by background summary
        "ts": _now_ms(),
        "references": references_initial,
    }
    messages.append(assistant_msg)

    update_session(session_id, user_id, {
        "messages": messages,
        "meta": sess.get("meta") or {},
        "rightPane": right_pane_initial
    })

    # ---------- SUMMARY STAGE (background) ----------
    exact_single = (len(used_docs) == 1 and not rag._extract_pmids_from_query(q_eff)) or (used_docs_mode == "detail")

    def _finish_summary():
        t_sum_start = time.perf_counter()
        try:
            if ctx.mode == "summarize":
                summary_obj = rag.summarize_plain(
                    q, used_docs, role=role, style="summary",
                    history_brief=(ctx.brief if ctx.follow_up else ""),
                    prior_docs=(ctx.prior_docs if ctx.follow_up else []),
                    include_citations=True
                )
            elif ctx.mode == "elaborate":
                summary_obj = rag.summarize_plain(
                    q, used_docs, role=role, style="elaborate",
                    history_brief=(ctx.brief if ctx.follow_up else ""),
                    prior_docs=(ctx.prior_docs if ctx.follow_up else []),
                    include_citations=True
                )
            elif used_docs_mode == "detail":
                summary_obj = rag.summarize_detail_paper(
                    q, used_docs, role=role, include_citations=True
                )
            else:
                if ctx.follow_up and ctx.prior_docs:
                    summary_obj = rag.summarize_with_context(q, used_docs, exact_flag=exact_single, role=role,
                                                             history_brief=ctx.brief, prior_docs=ctx.prior_docs,
                                                             include_citations=True)
                else:
                    summary_obj = rag.summarize(q, used_docs, exact_flag=exact_single, role=role,
                                                include_citations=True)
        except Exception as e:
            summary_obj = {"answer":{"conclusion": f"(summary error) {str(e)[:200]}", "key_findings": [], "quality_and_limits": [], "evidence_citations": [], "plain": ""}, "citation_links": {}}
        t_sum_end = time.perf_counter()

        assistant_text = _assistant_text_from_summary(summary_obj, docs_payload)
        refs = _mk_references_from_rightpane(right_pane_initial)

        # update session message + right pane
        doc = get_session_by_id(session_id, user_id) or {}
        msgs = doc.get("messages") or []
        new_msgs = []
        for m in msgs:
            if m.get("id") == assistant_msg["id"]:
                nm = dict(m)
                nm["content"] = assistant_text
                nm["references"] = refs
                new_msgs.append(nm)
            else:
                new_msgs.append(m)

        rp = doc.get("rightPane") or right_pane_initial
        ans = (summary_obj.get("answer") or {})
        rp["overview"] = {
            "conclusion": ans.get("conclusion","") if not ans.get("plain") else ans.get("plain")[:500],
            "key_findings": (ans.get("key_findings") or []),
            "quality_and_limits": (ans.get("quality_and_limits") or []),
        }
        rp["timings"] = {
            "retrieve_ms": timings_now["retrieve_ms"],
            "summarize_ms": int((t_sum_end - t_sum_start)*1000),
        }
        update_session(session_id, user_id, {"messages": new_msgs, "rightPane": rp})

        try:
            create_log({
                "user_id": user_id,
                "session_id": session_id,
                "query": question,
                "response": assistant_text[:8000],
                "fetch_ms": int(timings_now["retrieve_ms"]),
                "summarize_ms": int((t_sum_end - t_sum_start)*1000),
                "n_results": int(len(used_docs)),
                "ts": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            print(f"[LOGS] failed to write log: {e!r}")

    background_tasks.add_task(_finish_summary)

    return {"session_id": session_id, "message": assistant_msg, "rightPane": right_pane_initial}
