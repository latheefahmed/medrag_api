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
    temperature: float = float(os.getenv("PPLX_TEMPERATURE", "0.15"))
    max_tokens: int = int(os.getenv("PPLX_MAX_TOKENS", "900"))
    search_mode_plan: Optional[str] = "academic"
    search_mode_summary: Optional[str] = None

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

@dataclass
class BooleanCfg:
    max_terms_per_block: int = 6
    sim_threshold: float = 0.55
    long_sim_threshold: float = 0.45

@dataclass
class Cfg:
    pplx: PPLXCfg = field(default_factory=PPLXCfg)
    pubmed: PubMedCfg = field(default_factory=PubMedCfg)
    rank: RankCfg = field(default_factory=RankCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    summary: SummaryCfg = field(default_factory=SummaryCfg)
    boolean: BooleanCfg = field(default_factory=BooleanCfg)
    humans_filter: bool = True

CFG = Cfg()

# env keys
PPLX_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
PUBMED_KEY = os.getenv("PUBMED_API_KEY", "").strip()

# ================= http =================
def _session():
    s = requests.Session()
    ad = HTTPAdapter(
        max_retries=Retry(
            total=5, connect=3, read=3, backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods={"GET", "POST"},
        ),
        pool_connections=20, pool_maxsize=40,
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

# --- citation coercion helper (robust) ---
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

# ================= Perplexity =================
def _need_pplx():
    if not PPLX_KEY:
        raise RuntimeError("Perplexity API key missing or invalid (set PERPLEXITY_API_KEY).")

def pplx(messages, model, search_mode=None, temp=None, maxtok=None):
    _need_pplx()
    h = {"Authorization": f"Bearer {PPLX_KEY}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "stream": False,
            "temperature": CFG.pplx.temperature if temp is None else temp,
            "max_tokens": CFG.pplx.max_tokens if maxtok is None else maxtok}
    if search_mode: body["search_mode"] = search_mode
    r = S.post(CFG.pplx.url, headers=h, json=body, timeout=CFG.pubmed.timeout)
    if r.status_code == 401: raise RuntimeError("Perplexity API key invalid.")
    r.raise_for_status()
    return (r.json().get("choices", [{}])[0].get("message", {}) or {}).get("content", "")

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
def guardrail_domain_is_biomed(query: str) -> Dict[str, Any]:
    """
    Return domain_relevant=True if the query is biomedical/clinical.
    We'll use this to explicitly BLOCK biomedical content.
    """
    sys = textwrap.dedent("""
    You are a strict gatekeeper. Output STRICT JSON only.
    Mark "domain_relevant": true if and only if the query is biomedical/clinical/public health,
    including diseases, diagnostics, interventions, pharmacology, physiology, epidemiology,
    or any request that would reasonably require biomedical knowledge or literature.
    JSON schema: {"domain_relevant": true, "reason": "short justification"}
    """).strip()
    user = json.dumps({"query": norm(query)}, ensure_ascii=False)
    try:
        content = pplx(
            [{"role":"system","content":sys},
             {"role":"user","content":user}],
            CFG.pplx.model_guard,
            CFG.pplx.search_mode_plan,
            temp=0.0, maxtok=120
        )
        js = json_from_text(content)
    except Exception:
        js = None
    if not js:
        # fallback heuristic: look for common medical tokens
        words = [t.lower() for t in tok(query)]
        medish = {"patient","randomized","drug","dose","trial","disease","therapy","symptom","clinical","risk",
                  "mortality","morbidity","cohort","prospective","retrospective","meta","guideline","screening"}
        return {"domain_relevant": any(w in medish for w in words), "reason": "Heuristic fallback"}
    return {"domain_relevant": bool(js.get("domain_relevant", False)), "reason": str(js.get("reason",""))}

# ================= planner =================
def plan_tokens(query):
    sys = textwrap.dedent("""
    Output STRICT JSON only. Goal: minimal PubMed tokens (≤3 words).
    Group terms under Disease/Intervention/Comparator/Outcome/Population/Context.
    JSON: {"chunks":[],"anchors":{"Disease":[],"Intervention":[],"Comparator":[],"Outcome":[],"Population":[],"Context":[]}}
    """).strip()
    content = pplx(
        [{"role":"system","content":sys},
         {"role":"user","content":json.dumps({"query": norm(query)}, ensure_ascii=False)}],
        "sonar-pro", "academic", temp=0.0, maxtok=400
    )
    js = json_from_text(content)
    if not js:
        toks = [t for t in tok(query) if t.lower() not in STOP]
        return {"chunks": list(dict.fromkeys(toks[:6])),
                "anchors": {k: [] for k in ["Disease","Intervention","Comparator","Outcome","Population","Context"]}}
    nl = lambda xs, n: list(dict.fromkeys([norm(str(v)).lower() for v in (xs or []) if v and len(str(v).split()) <= 3]))[:n]
    js["chunks"] = nl(js.get("chunks", []), 6)
    for k in ["Disease","Intervention","Comparator","Outcome","Population","Context"]:
        js.setdefault("anchors", {}).setdefault(k, [])
        js["anchors"][k] = nl(js["anchors"][k], 8)
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

# ================= summaries =================
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

def _structured_block_summary(message: str) -> Dict[str, Any]:
    """
    Produce a fully-structured summary object that FE can render even when blocked.
    """
    return {
        "question": "",
        "answer": {
            "simple_summary": message,
            "what_was_studied": "",
            "conclusion": message,
            "key_findings": [],
            "quality_and_limits": ["Blocked by policy: biomedical queries are not allowed."],
            "evidence_citations": [],
        },
        "citation_links": {},
    }

def summarize(query, docs, exact_flag=False, role: Optional[str]=None):
    if not docs:
        return {
            "question": query,
            "answer": {
                "simple_summary": "No eligible literature found.",
                "what_was_studied": "not reported",
                "conclusion": "No eligible PubMed items found.",
                "key_findings": [],
                "quality_and_limits": ["Try broadening terms or removing exclusions."],
                "evidence_citations": [],
            },
            "citation_links": {}
        }

    cap = 5
    pack, links = evidence_pack(docs, cap=cap)

    flavor = (
        "Every factual sentence must include bracket citations like [1][2]. "
        "Start with 'simple_summary' (plain-English TL;DR). "
        "Include 'what_was_studied' describing designs/populations/endpoints. "
        "Do not use bullets or markdown. "
    )
    if exact_flag: flavor += "Treat the single paper as an exact match. "
    flavor += _role_flavor(role)

    sys = ("Use ONLY the EVIDENCE PACK; do not browse. Prefer RCTs/meta-analyses/guidelines. Respond with JSON only. " + flavor)
    schema = {
        "question":"string",
        "answer":{
            "simple_summary":"string",
            "what_was_studied":"string",
            "conclusion":"string",
            "key_findings":"array",
            "quality_and_limits":"array",
            "evidence_citations":"array",
            "evidence_notes":"object"
        }
    }
    user = f"QUESTION\n{query}\n\n{pack}\n\nTASK\nReturn MINIFIED JSON exactly in this schema (no prose):\n{json.dumps(schema,indent=2)}\nIf a field is missing, write 'not reported'. Ensure every claim has bracket citations."

    try:
        content = pplx(
            [{"role":"system","content":sys+" Return MINIFIED JSON only."},
             {"role":"user","content":user}],
            CFG.pplx.model_summary, CFG.pplx.search_mode_summary, temp=0.0, maxtok=900
        )
        js = json_from_text(content)
        if not js:
            content2 = pplx(
                [{"role":"system","content":sys},
                 {"role":"user","content":user}],
                CFG.pplx.model_summary, CFG.pplx.search_mode_summary, temp=0.0, maxtok=900
            )
            js = json_from_text(content2)
    except Exception:
        js = None

    if js:
        ans = js.get("answer") or {}
        raw_cites = ans.get("evidence_citations") or js.get("evidence_citations") or []
        ans["evidence_citations"] = _coerce_citations(raw_cites, max_idx=min(cap, len(docs)))
        ans["key_findings"] = [str(x) for x in (ans.get("key_findings") or [])]
        ans["quality_and_limits"] = [str(x) for x in (ans.get("quality_and_limits") or [])]
        ans["simple_summary"] = str(ans.get("simple_summary") or "")
        ans["what_was_studied"] = str(ans.get("what_was_studied") or "")
        js["answer"] = ans
        js["citation_links"] = links
        return js

    cites = list(range(1, min(cap, len(docs)) + 1))
    return {
        "question": query,
        "answer": {
            "simple_summary": "Automatic fallback summary.",
            "what_was_studied": "not reported",
            "conclusion": "See synthesized findings from the evidence pack [1].",
            "key_findings": [f"See items {cites} for key outcomes."],
            "quality_and_limits": ["Automatic fallback summary; verify primary sources."],
            "evidence_citations": cites
        },
        "citation_links": links,
        "note": "Fallback summary used."
    }

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
def _blocked_biomed_payload(reason: str = "") -> Dict[str, Any]:
    msg = "This assistant does not process biomedical/clinical queries."
    if reason:
        msg = f"{msg} {reason}".strip()
    return {
        "docs": [],
        "summary": _structured_block_summary(msg),
        "plan": {"chunks": []},
        "time_tags": ["", ""],
        "exclusions": [],
        "booleans": [],
        "timings": {"retrieve_ms": 0, "summarize_ms": 0},
    }

def _out_of_scope_payload() -> Dict[str, Any]:
    msg = "This assistant is restricted and cannot provide literature answers for this non-biomedical topic."
    return {
        "docs": [],
        "summary": _structured_block_summary(msg),
        "plan": {"chunks": []},
        "time_tags": ["", ""],
        "exclusions": [],
        "booleans": [],
        "timings": {"retrieve_ms": 0, "summarize_ms": 0},
    }

def run_pipeline(query, role: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    q = norm(query); lo, hi = time_tags(q); ex = parse_not(q)

    # HARD BLOCK: any biomedical/clinical/public health query is rejected.
    guard = guardrail_domain_is_biomed(q)
    if guard.get("domain_relevant", False):
        return _blocked_biomed_payload(guard.get("reason",""))

    # For non-biomedical queries, we also avoid PubMed retrieval (out of scope).
    t_retrieve_end = time.perf_counter()
    out = _out_of_scope_payload()
    out["timings"] = {"retrieve_ms": int((t_retrieve_end - t0)*1000), "summarize_ms": 0}
    return out

# =============== FE compatibility shim (unchanged function signature) ===============
def run_rag_pipeline(question: str, role: Optional[str] = None, verbose: bool = False, history: Optional[Dict[str,Any]] = None) -> Tuple[str, Dict[str, Any]]:
    pipe = run_pipeline(question, role=role)
    summary = pipe.get("summary") or {}; docs = pipe.get("docs") or []

    def _list_lines(items: List[str]) -> List[str]:
        out = []
        for i, x in enumerate(items or [], start=1):
            if x: out.append(f"{i}. {x.strip()}")
        return out

    if isinstance(summary, dict) and "answer" in summary:
        a = summary["answer"] or {}
        simple = (a.get("simple_summary") or "").strip()
        studied = (a.get("what_was_studied") or "").strip()
        conclusion = (a.get("conclusion") or "").strip()
        kf = _list_lines(a.get("key_findings") or [])
        ql = _list_lines(a.get("quality_and_limits") or [])
        cites = a.get("evidence_citations") or []
        links = {str(k): v for k, v in (summary.get("citation_links") or {}).items()}

        lines = []
        if simple: lines += [f"Quick take (plain-English): {simple}", ""]
        if conclusion: lines += [f"Brief answer: {conclusion}", ""]
        if studied: lines += ["What these papers studied:", studied, ""]
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
        "overview": {
            "simple_summary": (summary.get("answer") or {}).get("simple_summary", "") if isinstance(summary, dict) else "",
            "conclusion": (summary.get("answer") or {}).get("conclusion", "") if isinstance(summary, dict) else "",
            "key_findings": (summary.get("answer") or {}).get("key_findings", []) if isinstance(summary, dict) else [],
            "quality_and_limits": (summary.get("answer") or {}).get("quality_and_limits", []) if isinstance(summary, dict) else [],
        },
        "timings": pipe.get("timings") or {"retrieve_ms": 0, "summarize_ms": 0},
        "intent": {},   # intentionally empty: no follow-up memory/intent
    }
    return assistant_text, right_pane
