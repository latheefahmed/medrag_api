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
    Compact history for intent_decide + continuity:
    - last ~3 turns of messages (6 items)
    - prior docs from the last assistant message (ordered), plus union of previous assistant for recall
    """
    msgs = list(sess_doc.get("messages") or [])
    recent = msgs[-6:] if msgs else []
    prior_docs: List[Dict[str,Any]] = []
    seen_pmids = set()
    ordered_last = []
    count_assistant = 0
    for m in reversed(recent):
        if m.get("role") != "assistant":
            continue
        refs = m.get("references") or []
        if not ordered_last:
            ordered_last = refs  # preserve order for [n] mapping
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
        count_assistant += 1
        if count_assistant >= 2:
            break

    return {
        "recent_messages": recent,
        "prior_docs": prior_docs[:8],
        "last_refs_ordered": ordered_last[:20],
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

    # ---------- INTENT (LLM-only; conservative) ----------
    q_norm = rag.norm(question)
    intent_decision = rag.intent_decide(q_norm, history_payload)
    STRICT_FU = intent_decision.intent != "new_query" and intent_decision.confidence >= 0.55

    # ---------- GUARDRAIL ----------
    # Bypass only when we truly use prior docs; otherwise guard the query we will search with.
    guard_ok = True
    if not STRICT_FU or (STRICT_FU and not (history_payload.get("prior_docs") and intent_decision.intent in {"followup_summary","followup_detail","followup_citation_detail","followup_compare","followup_refine"})):
        g = rag.guardrail_domain(intent_decision.augmented_query or q_norm)
        guard_ok = bool(g.get("domain_relevant", False))

    if not guard_ok:
        right_pane = _right_pane_from_docs([], [], {"retrieve_ms": 0, "summarize_ms": 0}, overview={
            "conclusion": "The query didn’t look biomedical. Try clinical/biomedical terms.",
            "key_findings": [],
            "quality_and_limits": []
        }, intent={"follow_up": False, "reason": "domain-guard"})
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

    # ---------- ROUTING ----------
    last_refs_ordered = history_payload.get("last_refs_ordered") or []
    prior_docs = [rag._to_doc(d) for d in (history_payload.get("prior_docs") or [])]
    prior_docs = [d for d in prior_docs if d]

    # Helper: build right-pane + placeholder and persist
    def _setup_placeholder(doc_list: List["rag.Doc"], intent_desc: str, tried=None, timings=None):
        tried = tried or []
        timings = timings or {"retrieve_ms": 0, "summarize_ms": 0}
        docs_payload = [{
            "pmid": d.pmid, "title": d.title, "journal": d.journal, "year": d.year,
            "url": rag.pmid_url(d.pmid), "scores": d.scores, "abstract": d.abstract
        } for d in doc_list]
        right = _right_pane_from_docs(docs_payload, tried, timings, overview=None,
                                      intent={"follow_up": STRICT_FU, "reason": intent_desc,
                                              "augmented_query": intent_decision.augmented_query,
                                              "controller_conf": intent_decision.confidence})
        assistant = {"id": str(uuid4()), "role": "assistant", "content": "", "ts": _now_ms(),
                     "references": _mk_references_from_rightpane(right)}
        messages.append(assistant)
        update_session(session_id, user_id, {"messages": messages, "rightPane": right})
        return assistant, right, docs_payload

    # --- A) follow-up: summary only (no citations) ---
    if STRICT_FU and intent_decision.intent == "followup_summary":
        focus_docs = prior_docs
        if not focus_docs:
            # small, cheap retrieval if prior empty
            lo, hi = rag.time_tags(q_norm); ex = rag.parse_not(q_norm)
            plan = rag.plan_tokens(intent_decision.augmented_query or q_norm)
            var = rag.compose(plan, lo, hi, ex, qlen=len(rag.tok(q_norm)))
            merged = {}
            for v in var:
                try:
                    if not rag.esearch(v["query"], rag.CFG.pubmed.retmax_probe): continue
                    for d in rag.retrieve(v["query"], q_norm, v["label"]): merged[d.pmid] = d
                    if len(merged) >= 5: break
                except Exception: continue
            focus_docs = list(merged.values())[:5]

        assistant_msg, right0, docs_payload = _setup_placeholder(focus_docs, "plain-summary")
        def _finish():
            t0 = time.perf_counter()
            try:
                plain = rag.summarize_plain(question, focus_docs, role=role,
                                            history_brief=intent_decision.brief, prior_docs=None)
                text = plain.get("plain","") or "No summary generated."
            except Exception as e:
                text = f"(summary error) {str(e)[:200]}"
            t1 = time.perf_counter()
            doc = get_session_by_id(session_id, user_id) or {}
            msgs = doc.get("messages") or []
            nm = []
            for m in msgs:
                if m.get("id") == assistant_msg["id"]:
                    mm = dict(m); mm["content"] = text; nm.append(mm)
                else:
                    nm.append(m)
            rp = doc.get("rightPane") or right0
            rp["overview"] = {"conclusion": text, "key_findings": [], "quality_and_limits": []}
            rp["timings"]["summarize_ms"] = int((t1 - t0)*1000)
            update_session(session_id, user_id, {"messages": nm, "rightPane": rp})
            try:
                create_log({"user_id": user_id,"session_id": session_id,"query": question,
                            "response": text[:8000],"fetch_ms": 0,
                            "summarize_ms": int((t1 - t0)*1000),"n_results": len(docs_payload),
                            "ts": datetime.utcnow().isoformat()})
            except Exception as e:
                print(f"[LOGS] failed to write log: {e!r}")
        background_tasks.add_task(_finish)
        return {"session_id": session_id, "message": assistant_msg, "rightPane": right0}

    # --- B) follow-up: detail/citation focus (use indices if provided) ---
    if STRICT_FU and intent_decision.intent in {"followup_detail","followup_citation_detail"} and last_refs_ordered:
        focus = []
        if intent_decision.referenced_indices:
            for i in intent_decision.referenced_indices:
                if 1 <= i <= len(last_refs_ordered):
                    r = last_refs_ordered[i-1]
                    focus.append(rag._to_doc(r))
        if not focus:
            focus = prior_docs or []
        assistant_msg, right0, docs_payload = _setup_placeholder(focus, "detail-citation")
        def _finish():
            t0 = time.perf_counter()
            try:
                exact_single = len(focus) == 1
                summary_obj = rag.summarize(question, focus, exact_flag=exact_single, role=role)
                text = _assistant_text_from_summary(summary_obj, docs_payload)
            except Exception as e:
                text = f"(summary error) {str(e)[:200]}"
            t1 = time.perf_counter()
            doc = get_session_by_id(session_id, user_id) or {}
            msgs = doc.get("messages") or []
            nm = []
            for m in msgs:
                if m.get("id") == assistant_msg["id"]:
                    mm = dict(m); mm["content"] = text; nm.append(mm)
                else:
                    nm.append(m)
            rp = doc.get("rightPane") or right0
            rp["overview"] = {"conclusion": (summary_obj.get("answer") or {}).get("conclusion",""),
                              "key_findings": (summary_obj.get("answer") or {}).get("key_findings", []),
                              "quality_and_limits": (summary_obj.get("answer") or {}).get("quality_and_limits", [])}
            rp["timings"]["summarize_ms"] = int((t1 - t0)*1000)
            update_session(session_id, user_id, {"messages": nm, "rightPane": rp})
            try:
                create_log({"user_id": user_id,"session_id": session_id,"query": question,
                            "response": text[:8000],"fetch_ms": 0,
                            "summarize_ms": int((t1 - t0)*1000),"n_results": len(docs_payload),
                            "ts": datetime.utcnow().isoformat()})
            except Exception as e:
                print(f"[LOGS] failed to write log: {e!r}")
        background_tasks.add_task(_finish)
        return {"session_id": session_id, "message": assistant_msg, "rightPane": right0}

    # --- C) standalone NEW or follow-up REFINE/COMPARE (needs retrieval) ---
    t0 = time.perf_counter()
    lo, hi = rag.time_tags(q_norm); ex = rag.parse_not(q_norm)
    q_eff = intent_decision.augmented_query or q_norm
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
            if not rag.esearch(v["query"], rag.CFG.pubmed.retmax_probe):
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
                if not rag.esearch(lv["query"], rag.CFG.pubmed.retmax_probe):
                    continue
                for d in rag.retrieve(lv["query"], q_eff, lv["label"]):
                    if d.pmid not in merged:
                        merged[d.pmid] = d
                if len(merged) >= rag.CFG.retrieval.min_final:
                    break
            except Exception:
                continue

    # If the controller says refine/compare and we have prior docs, union them (no duplicates)
    if STRICT_FU and intent_decision.intent in {"followup_compare","followup_refine"} and prior_docs:
        for d in prior_docs:
            if d and d.pmid not in merged:
                merged[d.pmid] = d

    all_docs = list(merged.values())
    final_docs = rag.fuse_mmr(q_eff, all_docs) if all_docs else []
    used_docs = final_docs if final_docs else all_docs

    t_retrieve_end = time.perf_counter()
    timings_now = {"retrieve_ms": int((t_retrieve_end - t0)*1000), "summarize_ms": 0}

    docs_payload = [{
        "pmid": d.pmid, "title": d.title, "journal": d.journal, "year": d.year,
        "url": rag.pmid_url(d.pmid), "scores": d.scores, "abstract": d.abstract
    } for d in used_docs]

    right_pane_initial = _right_pane_from_docs(
        docs_payload, tried_booleans, timings_now, overview=None,
        intent={"follow_up": STRICT_FU, "reason": intent_decision.reason or intent_decision.intent,
                "augmented_query": intent_decision.augmented_query, "controller_conf": intent_decision.confidence}
    )
    references_initial = _mk_references_from_rightpane(right_pane_initial)

    assistant_msg = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": "",  # summary (WITH citations for this path)
        "ts": _now_ms(),
        "references": references_initial,
    }
    messages.append(assistant_msg)
    update_session(session_id, user_id, {"messages": messages, "rightPane": right_pane_initial})

    exact_single = len(used_docs) == 1 and not rag._extract_pmids_from_query(q_eff)

    def _finish_summary():
        t_sum_start = time.perf_counter()
        try:
            if STRICT_FU and intent_decision.intent in {"followup_compare","followup_refine"} and prior_docs:
                summary_obj = rag.summarize_with_context(question, used_docs, exact_flag=exact_single, role=role,
                                                         history_brief=intent_decision.brief, prior_docs=prior_docs)
            else:
                summary_obj = rag.summarize(question, used_docs, exact_flag=exact_single, role=role)
        except Exception as e:
            summary_obj = {"answer":{"conclusion": f"(summary error) {str(e)[:200]}",
                                     "key_findings": [], "quality_and_limits": [], "evidence_citations": []},
                           "citation_links": {}}
        t_sum_end = time.perf_counter()

        assistant_text = _assistant_text_from_summary(summary_obj, docs_payload)
        refs = _mk_references_from_rightpane(right_pane_initial)

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
        rp["overview"] = {
            "conclusion": (summary_obj.get("answer") or {}).get("conclusion",""),
            "key_findings": (summary_obj.get("answer") or {}).get("key_findings", []),
            "quality_and_limits": (summary_obj.get("answer") or {}).get("quality_and_limits", []),
        }
        rp["timings"] = {"retrieve_ms": timings_now["retrieve_ms"], "summarize_ms": int((t_sum_end - t_sum_start)*1000)}
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
