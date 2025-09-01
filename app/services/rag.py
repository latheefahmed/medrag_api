# -*- coding: utf-8 -*-
# app/services/rag.py

import os, re, json, time, textwrap, requests, numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from rank_bm25 import BM25Okapi
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---- optional embedding cache (in-memory or Qdrant via app.services.embed_cache) ----
try:
    from app.services.embed_cache import (
        cache_query_vec, cache_doc_vec,
        try_get_query_vec, try_get_doc_vecs,
    )
except Exception:
    def cache_query_vec(*a, **k): pass
    def cache_doc_vec(*a, **k): pass
    def try_get_query_vec(*a, **k): return None
    def try_get_doc_vecs(*a, **k): return {}

# ================= config =================
@dataclass
class PPLXCfg:
    url: str = "https://api.perplexity.ai/chat/completions"
    model_plan: str = os.getenv("PPLX_MODEL_PLAN", "sonar-pro")
    model_summary: str = os.getenv("PPLX_MODEL_SUMMARY", "sonar-pro")
    model_guard: str = os.getenv("PPLX_MODEL_GUARD", "sonar-pro")
    model_intent: str = os.getenv("PPLX_MODEL_INTENT", "sonar-mini")
    temperature: float = float(os.getenv("PPLX_TEMPERATURE", "0.15"))
    max_tokens: int = int(os.getenv("PPLX_MAX_TOKENS", "900"))
    search_mode_plan: Optional[str] = "academic"
    search_mode_summary: Optional[str] = None
    search_mode_intent: Optional[str] = None
    timeout: float = float(os.getenv("PPLX_TIMEOUT", "60"))

@dataclass
class GeminiCfg:
    # Flash for speed; configurable
    url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
    max_output_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "900"))
    timeout: float = float(os.getenv("GEMINI_TIMEOUT", "60"))

@dataclass
class PubMedCfg:
    base: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    email: str = os.getenv("CONTACT_EMAIL", "you@example.com")
    tool: str = "medrag_rag_fast"
    sleep_no_key: float = 0.36
    sleep_with_key: float = 0.12
    retmax_probe: int = 15
    retmax_main: int = 40
    timeout: float = 60.0

@dataclass
class RankCfg:
    embedder: str = "pritamdeka/S-PubMedBERT-MS-MARCO"
    embed_batch: int = 48
    w_cos: float = 0.55
    w_bm25: float = 0.45
    w_bonus: float = 0.05
    mmr_lambda: float = 0.70
    mmr_take: int = 10

@dataclass
class RetrievalCfg:
    topk_titles: int = 12
    final_take: int = 10
    min_final: int = 5

@dataclass
class SummaryCfg:
    top_docs_for_pack: int = 5
    abstract_chars: int = 650
    prior_docs_for_pack: int = 3   # carry a few prior docs

@dataclass
class BooleanCfg:
    max_terms_per_block: int = 6
    sim_threshold: float = 0.55
    long_sim_threshold: float = 0.45

@dataclass
class Cfg:
    pplx: PPLXCfg = field(default_factory=PPLXCfg)
    gemini: GeminiCfg = field(default_factory=GeminiCfg)
    pubmed: PubMedCfg = field(default_factory=PubMedCfg)
    rank: RankCfg = field(default_factory=RankCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    summary: SummaryCfg = field(default_factory=SummaryCfg)
    boolean: BooleanCfg = field(default_factory=BooleanCfg)
    humans_filter: bool = True

CFG = Cfg()

# env keys
PPLX_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
PUBMED_KEY = os.getenv("PUBMED_API_KEY", "").strip()

# ================= http =================
def _session():
    s = requests.Session()
    ad = HTTPAdapter(
        max_retries=Retry(
            total=5, connect=3, read=3, backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods={"GET", "POST"}
        ),
        pool_connections=20, pool_maxsize=40
    )
    s.mount("https://", ad); s.mount("http://", ad)
    s.headers.update({"User-Agent": f'{CFG.pubmed.tool}/1.0 ({CFG.pubmed.email})'})
    return s

S = _session()
_sleep = lambda: time.sleep(CFG.pubmed.sleep_with_key if PUBMED_KEY else CFG.pubmed.sleep_no_key)

# ================= utils =================
STOP = set(("a an and or but the is are was were be been being of in on at to for from with without vs versus compared "
            "as by into about across after before between within than which who whom whose what when where why how do does did "
            "done doing not no nor exclude excluding except only more most less least over under this these those such same other "
            "another each any all some many much several few per via if then else since study studies trial trials effect effects "
            "impact impacts role outcomes results methods patients subjects adults children men women male female").split())

norm = lambda s: re.sub(r"\s+", " ", (s or "").replace("–", "-").replace("—", "-")
                        .replace("“", "\"").replace("”", "\"").replace("’", "'").strip())
tok = lambda s: re.findall(r"[A-Za-z0-9\-\+]+", (s or ""))
q_ta = lambda p: f"\"{norm(p)}\"[Title/Abstract]" if " " in norm(p) else f"{norm(p)}[Title/Abstract]"
pmid_url = lambda p: f"https://pubmed.ncbi.nlm.nih.gov/{p}/"

def minmax(a: np.ndarray) -> np.ndarray:
    if a.size == 0: return a
    mn, mx = float(a.min()), float(a.max())
    return np.ones_like(a) * 0.5 if mx - mn < 1e-12 else (a - mn) / (mx - mn)

# --- citation coercion helper ---
def _coerce_citations(raw, max_idx: int) -> List[int]:
    out: List[int] = []
    for x in (raw or []):
        if isinstance(x, (int, float)):
            idx = int(x)
        elif isinstance(x, str):
            m = re.search(r"\d+", x)
            if not m:
                continue
            idx = int(m.group(0))
        else:
            continue
        if 1 <= idx <= max_idx and idx not in out:
            out.append(idx)
    return out

# ================= embeddings =================
class Embedder:
    def __init__(self, name): self._li = HuggingFaceEmbedding(model_name=name)
    def encode(self, texts, batch):
        if not texts: return np.zeros((0, 768), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch):
            out += [self._li.get_text_embedding(t) for t in texts[i:i+batch]]
        V = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        return V / norms

_EMB = Embedder(CFG.rank.embedder)

def embed(texts: List[str]) -> np.ndarray:
    if not texts: return np.zeros((0, 768), dtype=np.float32)
    return _EMB.encode(texts, CFG.rank.embed_batch)

# ================= LLM Core (PPLX first, Gemini fallback) =================
def _pplx(messages, model, search_mode=None, temp=None, maxtok=None):
    if not PPLX_KEY:
        raise RuntimeError("Perplexity API key missing.")
    h = {"Authorization": f"Bearer {PPLX_KEY}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "stream": False,
            "temperature": CFG.pplx.temperature if temp is None else temp,
            "max_tokens": CFG.pplx.max_tokens if maxtok is None else maxtok}
    if search_mode: body["search_mode"] = search_mode
    r = S.post(CFG.pplx.url, headers=h, json=body, timeout=CFG.pplx.timeout)
    if r.status_code == 401: raise RuntimeError("Perplexity API key invalid.")
    r.raise_for_status()
    return (r.json().get("choices", [{}])[0].get("message", {}) or {}).get("content", "")

def _gemini(system_text: str, user_text: str, temp=None, maxtok=None):
    if not GEMINI_KEY:
        raise RuntimeError("Gemini API key missing.")
    q = {
        "contents": [
            {"role": "user", "parts": [{"text": (system_text.strip() + "\n\n" + user_text.strip()).strip()}]}
        ],
        "generationConfig": {
            "temperature": CFG.gemini.temperature if temp is None else temp,
            "maxOutputTokens": CFG.gemini.max_output_tokens if maxtok is None else maxtok,
        }
    }
    r = S.post(f"{CFG.gemini.url}?key={GEMINI_KEY}", json=q, timeout=CFG.gemini.timeout)
    if r.status_code == 401: raise RuntimeError("Gemini API key invalid.")
    r.raise_for_status()
    data = r.json() or {}
    # best-effort content extraction
    try:
        return (data["candidates"][0]["content"]["parts"][0]["text"] or "").strip()
    except Exception:
        return ""

def llm_json(system_text: str, user_text: str, *, model_kind: str, search_mode: Optional[str]=None, temp: Optional[float]=None, maxtok: Optional[int]=None) -> Optional[Dict[str,Any]]:
    """
    Uniform JSON call:
    - First try Perplexity (fast/accurate).
    - On error or empty/invalid -> try Gemini.
    - If still invalid -> None (caller uses heuristics).
    """
    txt = ""
    try:
        txt = _pplx(
            [{"role":"system","content":system_text},
             {"role":"user","content":user_text}],
            {
                "plan": CFG.pplx.model_plan,
                "guard": CFG.pplx.model_guard,
                "intent": CFG.pplx.model_intent,
                "summary": CFG.pplx.model_summary
            }[model_kind],
            search_mode, temp=temp, maxtok=maxtok
        )
    except Exception:
        # PPLX failed; try Gemini
        try:
            txt = _gemini(system_text, user_text, temp=temp, maxtok=maxtok)
        except Exception:
            return None
    js = json_from_text(txt)
    if js is None:
        # final attempt: if PPLX succeeded but text was chatty, try reparse soft
        js = json_from_text(txt)
    return js

def json_from_text(t):
    t = re.sub(r"```(?:json)?\s*|\s*```", "", t or "").strip()
    o = [m.start() for m in re.finditer(r"\{", t)]
    c = [m.start() for m in re.finditer(r"\}", t)]
    if not o or not c: return None
    chunk = t[o[0]:c[-1]+1]
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
    except Exception:
        try: return json.loads(chunk)
        except Exception: return None

# ================= guardrails =================
def guardrail_domain(query: str) -> Dict[str, Any]:
    sys = textwrap.dedent("""
    You are a strict gatekeeper for a PubMed-backed biomedical QA system.
    Output STRICT JSON only.
    Consider a query "domain_relevant" ONLY if it primarily concerns human biomedical/clinical research,
    diseases, diagnostics, interventions, pharmacology, physiology, epidemiology, or public health.
    JSON schema: {"domain_relevant": true, "reason": "short justification"}
    """).strip()
    user = json.dumps({"query": norm(query)}, ensure_ascii=False)
    js = llm_json(sys, user, model_kind="guard", search_mode=CFG.pplx.search_mode_plan, temp=0.0, maxtok=120)
    if not js:
        words = [t for t in tok(query) if t.lower() not in STOP]
        return {"domain_relevant": len(words) >= 2, "reason": "Heuristic fallback"}
    return {"domain_relevant": bool(js.get("domain_relevant", False)), "reason": str(js.get("reason",""))}

# ================= planner =================
def plan_tokens(query):
    sys = textwrap.dedent("""
    You are a biomedical retrieval planner. Output STRICT JSON only.
    GOAL: minimal PubMed tokens (≤3 words). Group: Disease, Intervention, Comparator, Outcome, Population, Context.
    If Intervention & Comparator exist → must_pair=true. Include boolean "domain_relevant".
    JSON: {"chunks":[],"anchors":{"Disease":[],"Intervention":[],"Comparator":[],"Outcome":[],"Population":[],"Context":[]},"mesh_terms":{"Disease":[],"Intervention":[],"Comparator":[],"Outcome":[]},"must_pair":{"require_intervention_vs_comparator":false},"domain_relevant":true}
    """).strip()
    user = json.dumps({"query": norm(query)}, ensure_ascii=False)
    js = llm_json(sys, user, model_kind="plan", search_mode=CFG.pplx.search_mode_plan, temp=0.0, maxtok=500)
    if not js:
        toks = [t for t in tok(query) if t.lower() not in STOP]
        return {"chunks": list(dict.fromkeys(toks[:6])),
                "anchors": {k: [] for k in ["Disease","Intervention","Comparator","Outcome","Population","Context"]},
                "mesh_terms": {k: [] for k in ["Disease","Intervention","Comparator","Outcome"]},
                "must_pair": {"require_intervention_vs_comparator": False},
                "domain_relevant": True}
    nl = lambda xs, n: list(dict.fromkeys([norm(str(v)).lower() for v in (xs or []) if v and len(str(v).split()) <= 3]))[:n]
    js["chunks"] = nl(js.get("chunks", []), 6)
    for k in ["Disease","Intervention","Comparator","Outcome","Population","Context"]:
        js.setdefault("anchors", {}).setdefault(k, [])
        js["anchors"][k] = nl(js["anchors"][k], 8)
    for k in ["Disease","Intervention","Comparator","Outcome"]:
        js.setdefault("mesh_terms", {}).setdefault(k, [])
        js["mesh_terms"][k] = nl(js["mesh_terms"][k], 8)
    mp = js.get("must_pair") or {}
    js["must_pair"] = {"require_intervention_vs_comparator": bool(mp.get("require_intervention_vs_comparator"))}
    js["domain_relevant"] = bool(js.get("domain_relevant", True))
    return js

# ================= boolean composer =================
def _cluster(terms, k, thr):
    items = list(dict.fromkeys([norm(t).lower() for t in terms if t and t.lower() not in STOP]))
    if not items: return []
    V = embed(items)
    if V.shape[0] <= 1: return [items]
    cents = [V[0]]; groups = [[items[0]]]
    for _ in range(1, min(k, len(items))):
        d = [1 - max(float(np.dot(V[i], c)) for c in cents) for i in range(len(items))]
        if max(d) < (1 - thr): break
        j = int(np.argmax(d)); cents.append(V[j]); groups.append([items[j]])
    for i, v in enumerate(V):
        j = max(range(len(cents)), key=lambda c: float(np.dot(v, cents[c])))
        if items[i] not in groups[j]: groups[j].append(items[i])
    return groups

def _block(ts): return "(" + " OR ".join(q_ta(t) for t in ts[:CFG.boolean.max_terms_per_block]) + ")" if ts else ""

def _guards(boolean, lo, hi, ex):
    b = boolean.strip()
    if (lo or hi) and "PDAT" not in b:
        b += f' AND ("{lo or "0001"}"[PDAT] : "{hi or "3000"}"[PDAT])'
    if ex and " NOT " not in b:
        b += " NOT (" + " OR ".join(q_ta(e) for e in ex[:6]) + ")"
    if CFG.humans_filter and "Humans[Mesh]" not in b:
        b += " AND Humans[Mesh] NOT (Animals[Mesh] NOT Humans[Mesh])"
    return b

def compose(plan, lo, hi, ex, qlen: int):
    A = plan.get("anchors", {})
    dz, it, cp, oc, cx = A.get("Disease", []), A.get("Intervention", []), A.get("Comparator", []), A.get("Outcome", []), A.get("Context", [])
    long_query = qlen >= 50
    thr = CFG.boolean.long_sim_threshold if long_query else CFG.boolean.sim_threshold
    k_dz, k_int, k_cmp, k_out, k_ctx = (3,3,3,2,2) if long_query else (2,2,2,2,1)
    dzc, itc, cpc, occ, cxc = _cluster(dz, k_dz, thr), _cluster(it, k_int, thr), _cluster(cp, k_cmp, thr), _cluster(oc, k_out, thr), _cluster(cx, k_ctx, thr)
    vs = []
    if itc and cpc:
        inter_all = _block(sorted(set(it))); comp_all = _block(sorted(set(cp)))
        strict = " AND ".join([x for x in [_block(dzc[0] if dzc else []), inter_all, comp_all] if x])
        vs.append({"label": "inter_vs_comp_strict", "query": _guards(strict or q_ta("clinical"), lo, hi, ex)})
    relaxed = " AND ".join([x for x in [_block(dzc[0] if dzc else []), _block(itc[0] if itc else []), _block(occ[0] if occ else [])] if x])
    vs.append({"label": "dz_inter_out_relaxed", "query": _guards(relaxed or q_ta("clinical"), lo, hi, ex)})
    broad = " AND ".join([x for x in [_block(dzc[0] if dzc else []), _block(itc[0] if itc else []), _block(cxc[0] if cxc else [])] if x])
    vs.append({"label": "dz_inter_broad", "query": _guards(broad or q_ta("clinical"), lo, hi, ex)})
    if not dz and not it and plan.get("chunks"):
        ch = [c for c in plan["chunks"] if c][:3]
        fb = " AND ".join(q_ta(c) for c in ch) or q_ta("clinical")
        vs.append({"label": "chunks_compact", "query": _guards(fb, lo, hi, ex)})
    return vs

# ================= PubMed I/O =================
def _eutils(path, params, as_json):
    p = {**params, "tool": CFG.pubmed.tool, "email": CFG.pubmed.email}
    if PUBMED_KEY: p["api_key"] = PUBMED_KEY
    _sleep()
    r = S.get(CFG.pubmed.base + path, params=p, timeout=CFG.pubmed.timeout)
    if r.status_code in (429, 500, 502, 503, 504): raise RuntimeError("NCBI busy; try again later.")
    r.raise_for_status()
    return r.json() if as_json else r.text

def esearch(term, retmax): return _eutils("esearch.fcgi", {"db":"pubmed","retmode":"json","term":term,"retmax":str(retmax)}, True).get("esearchresult", {}).get("idlist", [])

def esummary(ids):
    if not ids: return {}
    return _eutils("esummary.fcgi", {"db":"pubmed","retmode":"json","id":",".join(ids)}, True).get("result", {})

def efetch(ids):
    if not ids: return {}
    xml = _eutils("efetch.fcgi", {"db":"pubmed","retmode":"xml","rettype":"abstract","id":",".join(ids)}, False)
    out = {}
    for blk in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, flags=re.S):
        m = re.search(r"<PMID[^>]*>(\d+)</PMID>", blk)
        if not m: continue
        pid = m.group(1)
        abst = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", blk, flags=re.S))
        abst = norm(re.sub(r"<[^>]+>", " ", abst))
        ptypes = [re.sub(r"<[^>]+>", "", p).strip() for p in re.findall(r"<PublicationType>(.*?)</PublicationType>", blk)]
        out[pid] = {"abstract": abst, "pubtypes": ptypes}
    return out

# ================= ranking =================
@dataclass
class Doc:
    pmid: str
    title: str
    journal: str
    year: int
    abstract: str
    pubtypes: List[str]
    scores: Dict[str, float]
    from_label: str = ""

def bm25_scores_local(query, docs):
    tokenized = [tok(d.lower()) for d in docs]
    bm = BM25Okapi(tokenized)
    return np.array(bm.get_scores(tok(query.lower())), dtype=np.float32)

def bonus_score(pt):
    p = [x.lower() for x in (pt or [])]; b = 0.0
    if any("random" in x for x in p): b += 0.6
    if any(("meta" in x) or ("systematic" in x) for x in p): b += 0.7
    if any("guideline" in x for x in p): b += 0.3
    return b

def mmr(qv, D, lam, k):
    if D.shape[0] == 0: return []
    s = (D @ qv.reshape(-1,1)).ravel()
    sel = [int(np.argmax(s))]
    rem = set(range(D.shape[0])) - set(sel)
    DD = D @ D.T
    while rem and len(sel) < k:
        best = max(rem, key=lambda i: lam*s[i] - (1-lam)*max(DD[i,j] for j in sel))
        sel.append(best); rem.remove(best)
    return sel

def retrieve(boolean, nl, label):
    try:
        ids = esearch(boolean, CFG.pubmed.retmax_main)
    except Exception:
        return []
    if not ids: return []
    sm = esummary(ids)
    order, titles, meta = [], [], {}
    for pid in ids:
        e = sm.get(pid, {})
        ttl = (e.get("title") or "").strip()
        if not ttl: continue
        jr = (e.get("fulljournalname") or e.get("source") or "").strip()
        pd = e.get("pubdate", "")
        m = re.search(r"\b(19|20)\d{2}\b", pd)
        yr = int(m.group(0)) if m else 0
        order.append(pid); titles.append(ttl); meta[pid] = {"title": ttl, "journal": jr, "year": yr}
    if not titles: return []
    b = bm25_scores_local(nl, titles)
    idx = sorted(range(len(titles)), key=lambda i: -b[i])[:CFG.retrieval.topk_titles]
    try:
        abs_map = efetch([order[i] for i in idx])
    except Exception:
        abs_map = {}
    out = []
    for pid in [order[i] for i in idx]:
        m = meta[pid]; am = abs_map.get(pid, {})
        out.append(Doc(
            pid, m["title"], m["journal"], m["year"],
            am.get("abstract", ""), am.get("pubtypes", []),
            {"bm25_group": float(b[order.index(pid)])}, label
        ))
    return out

def fuse_mmr(query, docs):
    if not docs: return []
    texts = [(d.title + " " + (d.abstract or "")).strip() for d in docs]

    qv = try_get_query_vec(query)
    if qv is None:
        qv = _EMB.encode([query], CFG.rank.embed_batch)[0]
        cache_query_vec(query, qv)

    D = _EMB.encode(texts, CFG.rank.embed_batch)
    cos = (D @ qv.reshape(-1,1)).ravel().astype(np.float32)
    bon = np.array([bonus_score(d.pubtypes) for d in docs], dtype=np.float32)
    bm25 = np.array([(d.scores or {}).get("bm25_group", 0.0) for d in docs], dtype=np.float32)

    fused = (
        CFG.rank.w_cos  * minmax(cos)  +
        CFG.rank.w_bm25 * minmax(bm25) +
        CFG.rank.w_bonus* minmax(bon)
    )

    top = min(CFG.rank.mmr_take, len(docs))
    order = np.argsort(-fused)[:top]
    Dtop = D[order]; dtop = [docs[i] for i in order]

    try:
        for i, d in enumerate(dtop):
            cache_doc_vec(d.pmid, d.title, d.journal, d.year, d.abstract, Dtop[i])
    except Exception:
        pass

    picks = mmr(qv, Dtop, CFG.rank.mmr_lambda, min(CFG.retrieval.final_take, top))
    for i, d in enumerate(docs):
        d.scores = {**(d.scores or {}), "cos": float(cos[i]), "bonus": float(bon[i]), "fused_raw": float(fused[i])}
    return [dtop[i] for i in picks]

# ==================== history/context ====================
@dataclass
class ResolvedContext:
    follow_up: bool = False
    reason: str = ""
    augmented_query: str = ""
    brief: str = ""             # compact history synopsis
    prior_docs: List[Doc] = field(default_factory=list)

def _intent_heuristic(query: str, recent_msgs: List[Dict[str,str]]) -> ResolvedContext:
    ql = query.lower().strip()
    referential = any(w in ql.split() for w in ("it","that","those","they","this","these","him","her","them")) \
                  or ql.startswith(("and ","also ","what about","how about","same","ok","hmm","continue","more","next"))
    summarize_like = any(k in ql for k in ("summarize","summary","tl;dr","abstract","explain briefly","explain in brief"))
    follow = referential or summarize_like or len(ql.split()) <= 4
    prev_topic = ""
    for m in reversed(recent_msgs):
        if m.get("role") == "user":
            prev_topic = norm(m.get("content",""))[:120]
            break
    aug = (prev_topic + " — " + query)[:180] if follow and prev_topic else query
    return ResolvedContext(follow_up=follow, reason="heuristic", augmented_query=aug, brief=prev_topic)

def _minify_messages(messages: List[Dict[str,str]], max_chars: int = 320) -> str:
    bits = []
    for m in messages[-6:]:  # ~3 turns (~2 prompts history emphasized)
        r = m.get("role","")[:1].upper()
        c = norm(m.get("content",""))
        if not c: continue
        bits.append(f"{r}: {c}")
        if sum(len(b) for b in bits) > max_chars:
            break
    return " | ".join(bits)[:max_chars]

def _to_doc(d: Dict[str,Any]) -> Optional[Doc]:
    if not d: return None
    pid = str(d.get("pmid") or d.get("id") or "").strip()
    if not pid: return None
    return Doc(
        pmid=pid,
        title=d.get("title") or "",
        journal=d.get("journal") or "",
        year=int(d.get("year") or 0),
        abstract=d.get("abstract") or "",
        pubtypes=d.get("pubtypes") or [],
        scores=d.get("scores") or {"bm25_group": float(d.get("score") or 0.0)},
        from_label=d.get("from_label") or "history"
    )

def _enrich_prior_docs(prior_docs: List[Doc]) -> List[Doc]:
    missing = [d.pmid for d in prior_docs if not d.abstract][:8]
    if missing:
        try:
            em = efetch(missing)
            for d in prior_docs:
                if d.pmid in em:
                    d.abstract = d.abstract or em[d.pmid].get("abstract","")
                    d.pubtypes = d.pubtypes or em[d.pmid].get("pubtypes",[])
        except Exception:
            pass
    return prior_docs

def resolve_context(query: str, history: Optional[Dict[str,Any]]) -> ResolvedContext:
    if not history:
        return ResolvedContext(follow_up=False, reason="no-history", augmented_query=query, brief="", prior_docs=[])
    msgs = history.get("recent_messages") or []
    prior_docs_in = history.get("prior_docs") or []

    pd: List[Doc] = []
    for dd in prior_docs_in:
        d = _to_doc(dd)
        if d: pd.append(d)
    pd = _enrich_prior_docs(pd)

    brief = _minify_messages(msgs)

    # Try intent via LLM (strict JSON). If that fails, heuristic.
    sys = textwrap.dedent("""
    You detect whether the latest user question is a FOLLOW-UP to the prior chat or a NEW standalone query.
    Output STRICT JSON only with keys:
    {"follow_up": true|false, "augmented_query": "≤15 words that resolves pronouns, else empty", "reason": "≤12 words"}
    Consider pronouns, ellipsis ("and what about dosage?"), "summarize that", etc.
    Only for summarization/explanation behavior; do not change retrieval rules.
    """).strip()
    user = json.dumps({
        "current_query": norm(query),
        "chat_brief": brief
    }, ensure_ascii=False)

    js = llm_json(sys, user, model_kind="intent", search_mode=CFG.pplx.search_mode_intent, temp=0.0, maxtok=140)
    if js:
        follow = bool(js.get("follow_up", False))
        aug = norm(js.get("augmented_query","")) or query
        reason = js.get("reason","")
        return ResolvedContext(follow_up=follow, reason=reason, augmented_query=aug, brief=brief, prior_docs=pd)

    fb = _intent_heuristic(query, msgs)
    fb.prior_docs = pd
    fb.brief = brief or fb.brief
    return fb

# ================= evidence + summary =================
def evidence_pack(docs, cap=5):
    chosen = docs[:cap]; idx2url = {}; lines = []
    for i, d in enumerate(chosen, 1):
        idx2url[i] = pmid_url(d.pmid)
        head = f"[{i}] PMID {d.pmid} ({d.year}) {d.journal} — {d.title} ({idx2url[i]})".strip(" —")
        body = f"Abstract: {d.abstract[:CFG.summary.abstract_chars]}" if d.abstract else "Abstract: (not available)"
        lines.append(f"{head}\n{body}")
    return "EVIDENCE PACK\n" + "\n\n".join(lines), idx2url

def _role_flavor(role: Optional[str]) -> str:
    r = (role or "").lower()
    if r in ("doctor","clinician","physician","md"):
        return "Write clinically and concisely with effect sizes and guideline context; avoid advice."
    if r in ("researcher","scientist"):
        return "Use a technical tone; emphasize design, endpoints, estimates, and limits."
    return "Use clear, precise prose without bullets or markdown."

def _summarize_llm(system_text: str, user_text: str) -> Optional[Dict[str,Any]]:
    # First PPLX, then Gemini
    js = llm_json(system_text + " Return MINIFIED JSON only.", user_text, model_kind="summary", search_mode=CFG.pplx.search_mode_summary, temp=0.0, maxtok=900)
    return js

def summarize(query, docs, exact_flag=False, role: Optional[str]=None):
    if not docs:
        return {"question":query,"answer":{"conclusion":"No eligible PubMed items found.","key_findings":[],"quality_and_limits":["Try broadening filters or removing exclusions."],"evidence_citations":[]}, "citation_links":{}}

    cap = CFG.summary.top_docs_for_pack
    pack, links = evidence_pack(docs, cap=cap)

    flavor = "Every factual sentence must include bracket citations like [1][2]. Do not use asterisks, dashes, bullets, or markdown anywhere. "
    if exact_flag: flavor += "Treat the single paper as an exact match. "
    flavor += _role_flavor(role)

    sys = ("Use ONLY the EVIDENCE PACK; do not browse. Prefer RCTs/meta-analyses/guidelines. Respond with JSON only. " + flavor)
    schema = {"question":"string","answer":{"conclusion":"string","key_findings":"array","quality_and_limits":"array","evidence_citations":"array","evidence_notes":"object"}}
    user = f"QUESTION\n{query}\n\n{pack}\n\nTASK\nReturn MINIFIED JSON exactly in this schema (no prose):\n{json.dumps(schema,indent=2)}\nIf a field is missing in EVIDENCE PACK, write 'not reported'. Ensure every claim has bracket citations."

    js = _summarize_llm(sys, user)

    if js:
        ans = js.get("answer") or {}
        raw_cites = ans.get("evidence_citations") or js.get("evidence_citations") or []
        ans["evidence_citations"] = _coerce_citations(raw_cites, max_idx=min(cap, len(docs)))
        ans["key_findings"] = [str(x) for x in (ans.get("key_findings") or [])]
        ans["quality_and_limits"] = [str(x) for x in (ans.get("quality_and_limits") or [])]
        js["answer"] = ans
        js["citation_links"] = links
        return js

    cites = list(range(1, min(cap, len(docs)) + 1))
    return {"question":query,
            "answer":{"conclusion":"See synthesized findings from the evidence pack [1].",
                      "key_findings":[f"See items {cites} for key outcomes."],
                      "quality_and_limits":["Automatic fallback summary; verify primary sources."],
                      "evidence_citations": cites},
            "citation_links": links,
            "note":"Fallback summary used."}

def summarize_with_context(query: str, docs: List[Doc], exact_flag: bool, role: Optional[str],
                           history_brief: str = "", prior_docs: Optional[List[Doc]] = None):
    prior_docs = prior_docs or []
    if not prior_docs:
        return summarize(query, docs, exact_flag=exact_flag, role=role)

    # Build prior pack A1.. and main pack 1..
    pcap = CFG.summary.prior_docs_for_pack
    chosen = prior_docs[:pcap]
    lines = []
    links: Dict[str,str] = {}
    for i, d in enumerate(chosen, 1):
        tag = f"A{i}"
        links[tag] = pmid_url(d.pmid)
        head = f"[{tag}] PMID {d.pmid} ({d.year}) {d.journal} — {d.title} ({links[tag]})"
        body = f"Abstract: {d.abstract[:CFG.summary.abstract_chars]}" if d.abstract else "Abstract: (not available)"
        lines.append(f"{head}\n{body}")
    prior_pack = ("PRIOR CONTEXT PACK\n" + "\n\n".join(lines)) if lines else ""

    cap = CFG.summary.top_docs_for_pack
    main_pack, main_links = evidence_pack(docs, cap=cap)

    flavor = _role_flavor(role)
    sys = ("You answer biomedical questions grounded to the provided packs only. "
           "Prefer the current EVIDENCE PACK; use PRIOR CONTEXT PACK only to maintain continuity (e.g., compare, clarify, or summarize across turns). "
           "Respond with JSON only; every factual sentence must include bracket citations like [1][2] or [A1]. " + flavor)

    schema = {"question":"string","answer":{"conclusion":"string","key_findings":"array","quality_and_limits":"array","evidence_citations":"array","evidence_notes":"object"}}

    user = f"""CHAT BRIEF
{history_brief or "(none)"}

QUESTION
{query}

{main_pack}

{prior_pack if prior_pack else ""}

TASK
Return MINIFIED JSON exactly in this schema (no prose):
{json.dumps(schema,indent=2)}
Cite from [1..{min(cap,len(docs))}] for current evidence; if you must reference prior context, use [A1..A{len(chosen)}].
If a field is missing, write 'not reported'. Ensure every claim has bracket citations.
"""

    js = _summarize_llm(sys, user)

    if js:
        ans = js.get("answer") or {}
        # Normalize numeric citations for FE; prior A* are allowed for text but we keep numeric for links
        numeric_only = []
        for x in (ans.get("evidence_citations") or []):
            m = re.search(r"\d+", str(x))
            if m:
                idx = int(m.group(0))
                if 1 <= idx <= min(cap, len(docs)) and idx not in numeric_only:
                    numeric_only.append(idx)
        ans["evidence_citations"] = numeric_only
        ans["key_findings"] = [str(x) for x in (ans.get("key_findings") or [])]
        ans["quality_and_limits"] = [str(x) for x in (ans.get("quality_and_limits") or [])]
        js["answer"] = ans
        js["citation_links"] = main_links | links
        return js

    return summarize(query, docs, exact_flag=exact_flag, role=role)

# ================= helpers =================
def time_tags(q):
    ql = q.lower()
    m_b = re.search(r"\bbetween\s+((?:19|20)\d{2})\s+(?:and|-|to)\s+((?:19|20)\d{2})\b", ql)
    if m_b: return m_b.group(1), m_b.group(2)
    m_ge = re.search(r"\bsince\s+((?:19|20)\d{2})\b", ql)
    m_le = re.search(r"\bbefore\s+((?:19|20)\d{2})\b", ql)
    return (m_ge.group(1) if m_ge else "", m_le.group(1) if m_le else "")

def parse_not(q):
    out = []
    for kw in ("exclude","excluding","without","not"):
        out += [norm(m.group(1)) for m in re.finditer(rf"\b{kw}\s+([^.;,]+)", q, flags=re.I)]
    seen, setret = set(), []
    for e in out:
        el = e.lower()
        if el and el not in seen:
            seen.add(el); setret.append(e)
    return setret[:6]

def ultra_relaxed(q, lo, hi, ex):
    base = " AND ".join(q_ta(t) for t in [t for t in tok(q) if t.lower() not in STOP][:4]) or q_ta("clinical")
    return _guards(base, lo, hi, ex)

def widen_variants(plan, q, lo, hi, ex):
    A = plan.get("anchors", {})
    dz = (A.get("Disease") or [])[:3]; it = (A.get("Intervention") or [])[:3]; cp = (A.get("Comparator") or [])[:3]; oc = (A.get("Outcome") or [])[:3]
    chans = [{"label":"lenient_no_not","query":_guards(" AND ".join([q_ta(t) for t in (dz[:1]+it[:1]+cp[:1]+oc[:1]) if t]), lo, hi, ex=[])},
             {"label":"lenient_no_date","query":_guards(" AND ".join([q_ta(t) for t in (dz[:1]+it[:1]+cp[:1]) if t]) or q_ta("clinical"), "", "", ex)},
             {"label":"lenient_or_chunks","query":"("+" OR ".join(q_ta(t) for t in (plan.get("chunks") or [])[:4])+")" if (plan.get("chunks") or []) else q_ta("clinical")}]
    if dz: chans.append({"label":"lenient_dz_only","query":_guards("("+" OR ".join(q_ta(t) for t in dz)+")", lo, hi, [])})
    if it: chans.append({"label":"lenient_it_only","query":_guards("("+" OR ".join(q_ta(t) for t in it)+")", lo, hi, [])})
    chans.append({"label":"ultra_relaxed","query": ultra_relaxed(q, lo, hi, [])})
    return chans

# ================= pipeline =================
def _guardrail_payload(chunks, lo, hi, ex, tried, reason: str = ""):
    base = "We can’t search relevant keywords from our PubMed-backed database. Please try another query using healthcare or biomedical terminology."
    msg = base if not reason else f"{base} {reason}".strip()
    return {
        "docs": [],
        "summary": {"answer": {"conclusion": msg, "key_findings": [], "quality_and_limits": [], "evidence_citations": []},
                    "citation_links": {}},
        "plan": {"chunks": chunks},
        "time_tags": [lo, hi],
        "exclusions": ex,
        "booleans": tried,
        "timings": {"retrieve_ms": 0, "summarize_ms": 0},
    }

def _extract_pmids_from_query(q: str) -> List[str]:
    pmids = set()
    for m in re.finditer(r"\bPMID[:\s]*([0-9]{5,8})\b", q, flags=re.I):
        pmids.add(m.group(1))
    for m in re.finditer(r"pubmed\.ncbi\.nlm\.nih\.gov/([0-9]{5,8})", q, flags=re.I):
        pmids.add(m.group(1))
    return list(pmids)

def _fetch_docs_by_pmids(pmids: List[str]) -> List[Doc]:
    if not pmids: return []
    sm = esummary(pmids) or {}
    meta = {}
    for pid in pmids:
        e = sm.get(pid, {})
        ttl = (e.get("title") or "").strip() or ""
        jr = (e.get("fulljournalname") or e.get("source") or "").strip()
        pd = e.get("pubdate", "")
        m = re.search(r"\b(19|20)\d{2}\b", pd)
        yr = int(m.group(0)) if m else 0
        meta[pid] = {"title": ttl, "journal": jr, "year": yr}
    ab = efetch(pmids) or {}
    out = []
    for pid in pmids:
        mm = meta.get(pid, {"title":"", "journal":"", "year":0})
        aa = ab.get(pid, {})
        out.append(Doc(
            pmid=pid, title=mm["title"], journal=mm["journal"], year=mm["year"],
            abstract=aa.get("abstract",""), pubtypes=aa.get("pubtypes",[]),
            scores={"bm25_group": 0.0}, from_label="pmid_direct"
        ))
    return out

def run_pipeline(query, role: Optional[str] = None, history: Optional[Dict[str,Any]] = None):
    t0 = time.perf_counter()
    q = norm(query); lo, hi = time_tags(q); ex = parse_not(q)

    # Guardrail
    guard = guardrail_domain(q)
    if not guard.get("domain_relevant", False):
        return _guardrail_payload([], lo, hi, ex, [], guard.get("reason",""))

    # Direct PMID?
    direct_pmids = _extract_pmids_from_query(q)

    # Resolve context (follow-up only influences summary/explanation)
    ctx = resolve_context(q, history)
    q_eff = ctx.augmented_query if ctx.follow_up else q

    # Planner
    try:
        plan = plan_tokens(q_eff)
    except Exception:
        plan = {"chunks":[t for t in tok(q_eff) if t.lower() not in STOP][:4],"anchors":{},"domain_relevant":True}

    if not plan.get("domain_relevant", True):
        return _guardrail_payload(plan.get("chunks",[]), lo, hi, ex, [], "The query did not appear biomedical.")

    q_tokens = len(tok(q_eff))
    variants = compose(plan, lo, hi, ex, qlen=q_tokens)
    tried_booleans, merged = [], {}

    # If explicit PMID(s), short-circuit retrieval
    if direct_pmids:
        for d in _fetch_docs_by_pmids(direct_pmids):
            merged[d.pmid] = d

    for v in variants:
        if len(merged) >= CFG.retrieval.topk_titles:
            break
        tried_booleans.append({"label": v["label"], "query": v["query"]})
        try:
            if not esearch(v["query"], CFG.pubmed.retmax_probe): continue
            for d in retrieve(v["query"], q_eff, v["label"]): merged[d.pmid] = d
        except Exception:
            continue

    if len(merged) < CFG.retrieval.min_final:
        for lv in widen_variants(plan, q_eff, lo, hi, ex):
            if len(merged) >= CFG.retrieval.min_final: break
            tried_booleans.append({"label": lv["label"], "query": lv["query"]})
            try:
                if not esearch(lv["query"], CFG.pubmed.retmax_probe): continue
                for d in retrieve(lv["query"], q_eff, lv["label"]):
                    if d.pmid not in merged: merged[d.pmid] = d
            except Exception:
                continue

    if not merged:
        try:
            fb = ultra_relaxed(q_eff, lo, hi, ex)
            tried_booleans.append({"label":"fallback_compact","query":fb})
            for d in retrieve(fb, q_eff, "fallback_compact"): merged[d.pmid] = d
        except Exception:
            pass

    # Merge prior docs if follow-up
    if ctx.follow_up and ctx.prior_docs:
        for d in ctx.prior_docs:
            if d.pmid not in merged:
                merged[d.pmid] = d

    all_docs = list(merged.values())
    final_docs = fuse_mmr(q_eff, all_docs) if all_docs else []
    used_docs = final_docs if final_docs else all_docs

    t_retrieve_end = time.perf_counter()

    if not used_docs:
        out = _guardrail_payload(plan.get("chunks",[]), lo, hi, ex, tried_booleans, "No PubMed items were found.")
        out["timings"] = {"retrieve_ms": int((t_retrieve_end - t0)*1000), "summarize_ms": 0}
        return out

    t_sum_start = time.perf_counter()
    if ctx.follow_up and ctx.prior_docs:
        summary_obj = summarize_with_context(q, used_docs, exact_flag=(len(used_docs)==1 and not direct_pmids), role=role,
                                             history_brief=ctx.brief, prior_docs=ctx.prior_docs)
    else:
        summary_obj = summarize(q, used_docs, exact_flag=(len(used_docs)==1 and not direct_pmids), role=role)
    t_sum_end = time.perf_counter()

    return {
        "docs": [{**asdict(d), "url": pmid_url(d.pmid)} for d in used_docs],
        "summary": summary_obj,
        "plan": {"chunks": plan.get("chunks", [])},
        "time_tags": [lo, hi],
        "exclusions": ex,
        "booleans": tried_booleans,
        "intent": {"follow_up": ctx.follow_up, "reason": ctx.reason, "augmented_query": ctx.augmented_query, "chat_brief": ctx.brief},
        "timings": {
            "retrieve_ms": int((t_retrieve_end - t0)*1000),
            "summarize_ms": int((t_sum_end - t_sum_start)*1000),
        },
    }

# ================= FE compatibility shim (unchanged API) =================
def run_rag_pipeline(question: str, role: Optional[str] = None, verbose: bool = False, history: Optional[Dict[str,Any]] = None) -> Tuple[str, Dict[str, Any]]:
    pipe = run_pipeline(question, role=role, history=history)
    summary = pipe.get("summary") or {}; docs = pipe.get("docs") or []

    def _list_lines(items: List[str]) -> List[str]:
        out = []
        for i, x in enumerate(items or [], start=1):
            if x: out.append(f"{i}. {x.strip()}")
        return out

    if isinstance(summary, dict) and "answer" in summary:
        a = summary["answer"] or {}
        conclusion = (a.get("conclusion") or "").strip()
        kf = _list_lines(a.get("key_findings") or [])
        ql = _list_lines(a.get("quality_and_limits") or [])
        cites = a.get("evidence_citations") or []
        links = {str(k): v for k, v in (summary.get("citation_links") or {}).items()}

        lines = []
        if conclusion: lines += [conclusion, ""]
        if kf: lines += ["Key findings:", *kf, ""]
        if ql: lines += ["Quality and limitations:", *ql, ""]
        if cites:
            lines.append("Citations:")
            for n in cites:
                try:
                    idx = int(n)
                except Exception:
                    continue
                title = docs[idx-1]["title"] if 1 <= idx <= len(docs) else f"Reference {idx}"
                url = links.get(str(idx)) or (docs[idx-1].get("url") if 1 <= idx <= len(docs) else "")
                lines.append(f"{idx}. {title} {('- ' + url) if url else ''}")
        assistant_text = "\n".join([s for s in lines if s is not None])
    else:
        assistant_text = "No summary generated."

    results = [{"pmid": d.get("pmid"), "title": d.get("title"), "journal": d.get("journal"),
                "year": d.get("year"), "url": d.get("url"),
                "score": (d.get("scores") or {}).get("fused_raw") if d.get("scores") else None,
                "abstract": d.get("abstract")} for d in docs]

    right_pane = {
        "results": results,
        "booleans": [{"group": b.get("label",""), "query": b.get("query",""), "note": ""} for b in (pipe.get("booleans") or [])],
        "plan": {"chunks": (pipe.get("plan") or {}).get("chunks", []),
                 "time_tags": pipe.get("time_tags", []),
                 "exclusions": pipe.get("exclusions", [])},
        "overview": None,
        "timings": pipe.get("timings") or {"retrieve_ms": 0, "summarize_ms": 0},
        "intent": pipe.get("intent") or {},
    }
    return assistant_text, right_pane
