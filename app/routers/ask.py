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

def _right_pane_from_docs(docs: List[Dict[str, Any]], tried: List[Dict[str, str]], timings: Dict[str,int], overview=None):
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

    # Ensure a session doc exists
    sess = get_session_by_id(session_id, user_id)
    if not sess:
        sess = create_session({"id": session_id, "user_id": user_id, "title":"Untitled", "messages": [], "meta": {}})
    memory = (sess or {}).get("memory")

    # ---------- guard + intent (prompt-engineered, no keyword lists) ----------
    cls = rag.classify_query(question, memory_exists=bool((memory or {}).get("last_pack")))
    if not cls.get("domain_relevant", False) or cls.get("intent") == "NON_BIOMEDICAL":
        right_pane = _right_pane_from_docs([], [], {"retrieve_ms": 0, "summarize_ms": 0}, overview={
            "conclusion": "This assistant answers biomedical questions only.",
            "key_findings": [],
            "quality_and_limits": []
        })
        assistant_msg = {
            "id": str(uuid4()),
            "role": "assistant",
            "content": "Your query appears non-biomedical. Please ask a clinical/biomedical question.",
            "ts": _now_ms(),
            "references": [],
        }
        messages = (sess.get("messages") or []) + [{"id": str(uuid4()), "role": "user", "content": question, "ts": _now_ms()}, assistant_msg]
        update_session(session_id, user_id, {"messages": messages, "rightPane": right_pane})
        return {"session_id": session_id, "message": assistant_msg, "rightPane": right_pane}

    intent = str(cls.get("intent","NEW_QUERY")).upper()
    conf = float(cls.get("confidence", 0.7))

    # ---------- memory follow-up path (only when classifier is confident and memory exists) ----------
    if intent in ("SUMMARIZE_THREAD","EXPAND_CONCEPT","FOLLOWUP_CITATION") and conf >= 0.7 and ((memory or {}).get("last_pack") or {}).get("docs"):
        mem_docs = memory["last_pack"]["docs"]

        # FOLLOWUP narrowing if citation index or PMID provided by classifier
        narrowed = mem_docs
        idx_map = (memory.get("last_pack") or {}).get("idx_map") or {}
        if intent == "FOLLOWUP_CITATION":
            pmid_target = None
            if cls.get("citation_index"):
                pmid_target = idx_map.get(str(int(cls["citation_index"])))
            if not pmid_target and cls.get("pmid"):
                pmid_target = cls["pmid"]
            if pmid_target:
                narrowed = [d for d in mem_docs if str(d.get("pmid")) == str(pmid_target)] or mem_docs[:1]

        docs_payload = [{
            "pmid": d.get("pmid"), "title": d.get("title"), "journal": d.get("journal"),
            "year": d.get("year"), "url": rag.pmid_url(d.get("pmid")), "scores": {"bm25_group": i+1},
            "abstract": d.get("abstract"),
        } for i, d in enumerate(narrowed[:20])]

        right_pane_initial = _right_pane_from_docs(docs_payload, tried=[], timings={"retrieve_ms": 0, "summarize_ms": 0})
        references_initial = _mk_references_from_rightpane(right_pane_initial)

        user_msg = {"id": str(uuid4()), "role": "user", "content": question, "ts": _now_ms()}
        assistant_msg = {"id": str(uuid4()), "role": "assistant", "content": "", "ts": _now_ms(), "references": references_initial}
        messages = (sess.get("messages") or []) + [user_msg, assistant_msg]
        update_session(session_id, user_id, {"messages": messages, "rightPane": right_pane_initial})

        def _finish_summary_from_mem():
            t1 = time.perf_counter()
            sum_obj = rag.summarize_from_memory(question, {"last_pack": {"docs": narrowed}}, role=role)
            t2 = time.perf_counter()
            text = _assistant_text_from_summary(sum_obj, docs_payload)

            doc = get_session_by_id(session_id, user_id) or {}
            patched = []
            for m in (doc.get("messages") or []):
                if m.get("id") == assistant_msg["id"]:
                    nm = dict(m); nm["content"] = text; nm["references"] = references_initial; patched.append(nm)
                else:
                    patched.append(m)
            rp = doc.get("rightPane") or right_pane_initial
            rp["overview"] = {
                "conclusion": (sum_obj.get("answer") or {}).get("conclusion",""),
                "key_findings": (sum_obj.get("answer") or {}).get("key_findings", []),
                "quality_and_limits": (sum_obj.get("answer") or {}).get("quality_and_limits", []),
            }
            rp["timings"] = {"retrieve_ms": 0, "summarize_ms": int((t2 - t1)*1000)}
            update_session(session_id, user_id, {"messages": patched, "rightPane": rp})
            try:
                create_log({
                    "user_id": user_id, "session_id": session_id, "query": question,
                    "response": text[:8000], "fetch_ms": 0, "summarize_ms": int((t2 - t1)*1000),
                    "n_results": len(docs_payload), "ts": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                print(f"[LOGS] memory path log fail: {e!r}")

        background_tasks.add_task(_finish_summary_from_mem)
        return {"session_id": session_id, "message": assistant_msg, "rightPane": right_pane_initial}

    # ---------- NORMAL STANDALONE PATH ----------
    t0 = time.perf_counter()
    q = rag.norm(question); lo, hi = rag.time_tags(q); ex = rag.parse_not(q)

    try:
        plan = rag.plan_tokens(q)
    except Exception:
        plan = {"chunks":[t for t in rag.tok(q) if t.lower() not in rag.STOP][:4],"anchors":{},"domain_relevant":True}

    variants = rag.compose(plan, lo, hi, ex, qlen=len(rag.tok(q)))
    tried_booleans, merged = [], {}

    for v in variants:
        tried_booleans.append({"label": v["label"], "query": v["query"]})
        try:
            if not rag.esearch(v["query"], rag.CFG.pubmed.retmax_probe):
                continue
            for d in rag.retrieve(v["query"], q, v["label"]):
                merged[d.pmid] = d
            if len(merged) >= rag.CFG.retrieval.topk_titles:
                break
        except Exception:
            continue

    if len(merged) < rag.CFG.retrieval.min_final:
        for lv in rag.widen_variants(plan, q, lo, hi, ex):
            tried_booleans.append({"label": lv["label"], "query": lv["query"]})
            try:
                if not rag.esearch(lv["query"], rag.CFG.pubmed.retmax_probe):
                    continue
                for d in rag.retrieve(lv["query"], q, lv["label"]):
                    if d.pmid not in merged:
                        merged[d.pmid] = d
                if len(merged) >= rag.CFG.retrieval.min_final:
                    break
            except Exception:
                continue

    if not merged:
        try:
            fb = rag.ultra_relaxed(q, lo, hi, ex)
            tried_booleans.append({"label":"fallback_compact","query":fb})
            for d in rag.retrieve(fb, q, "fallback_compact"):
                merged[d.pmid] = d
        except Exception:
            pass

    all_docs = list(merged.values())
    final_docs = rag.fuse_mmr(q, all_docs) if all_docs else []
    used_docs = final_docs if final_docs else all_docs

    t_retrieve_end = time.perf_counter()
    timings_now = {"retrieve_ms": int((t_retrieve_end - t0)*1000), "summarize_ms": 0}

    docs_payload = [{
        "pmid": d.pmid, "title": d.title, "journal": d.journal, "year": d.year,
        "url": rag.pmid_url(d.pmid), "scores": d.scores, "abstract": d.abstract
    } for d in used_docs]

    right_pane_initial = _right_pane_from_docs(docs_payload, tried_booleans, timings_now, overview=None)
    references_initial = _mk_references_from_rightpane(right_pane_initial)

    user_msg = {"id": str(uuid4()), "role": "user", "content": question, "ts": _now_ms()}
    assistant_msg = {"id": str(uuid4()), "role": "assistant", "content": "", "ts": _now_ms(), "references": references_initial}
    messages = (sess.get("messages") or []) + [user_msg, assistant_msg]
    update_session(session_id, user_id, {"messages": messages, "meta": sess.get("meta") or {}, "rightPane": right_pane_initial})

    exact_single = (len(used_docs) == 1)

    def _finish_summary():
        t_sum_start = time.perf_counter()
        try:
            summary_obj = rag.summarize(q, used_docs, exact_flag=exact_single, role=role)
        except Exception as e:
            summary_obj = {"answer":{"conclusion": f"(summary error) {str(e)[:200]}",
                                     "key_findings": [], "quality_and_limits": [], "evidence_citations": []},
                           "citation_links": {}}
        t_sum_end = time.perf_counter()

        # --- write tiny session memory blob ---
        try:
            mem_blob = rag.build_session_memory(used_docs, summary_obj)
            doc = get_session_by_id(session_id, user_id) or {}
            old_mem = doc.get("memory") or {}
            merged_concepts = (old_mem.get("concepts") or [])[:5] + (mem_blob.get("concepts") or [])[:5]
            new_mem = {
                "thread_summary": mem_blob.get("thread_summary") or old_mem.get("thread_summary",""),
                "concepts": merged_concepts[:10],
                "last_pack": mem_blob.get("last_pack") or old_mem.get("last_pack") or {},
            }
            update_session(session_id, user_id, {"memory": new_mem})
        except Exception as e:
            print(f"[MEM] failed to update memory: {e!r}")

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
