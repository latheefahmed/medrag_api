# app/services/rag.py
# -------------------------------------------------------------------------------------------------
# Your original pipeline (verbatim logic) + a thin wrapper for the /ask endpoint and UI mapping.
# Requirements: requests, rank-bm25, numpy, sentence-transformers
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import os, re, json, time, hashlib, textwrap, platform
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple, Optional

import requests
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ---------- KEYS (ENV or defaults) ----------
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "7830499494a0493a17b33fd9203247925707")
PPLX_API_KEY   = os.getenv("PERPLEXITY_API_KEY", "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")
CONTACT_EMAIL  = os.getenv("CONTACT_EMAIL", "lathad046@rmkcet.ac.in")

# ---------- Config ----------
CFG = {
    "paths": {"cache": os.path.abspath("./cache_pplx_mesh_pubmedrag")},
    "pplx": {
        "url": "https://api.perplexity.ai/chat/completions",
        "model_plan": "sonar-pro",
        "model_summary": "sonar-pro",
        "temperature": 0.15,
        "max_tokens": 900,
        "search_mode_plan": "academic",   # planner can browse academic corpora
        "search_mode_summary": None       # summarizer must NOT browse
    },
    "pubmed": {
        "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        "email": CONTACT_EMAIL,
        "tool": "pplx_mesh_pubmedrag",
        "sleep": 0.28,
        "retmax_main": 120,
        "retmax_backoff": 220
    },
    "retrieval": {
        "per_group_keep": 30,
        "min_docs_wanted": 14,
        "final_take": 25
    },
    "rank": {
        "w_bm25": 0.58,
        "w_emb":  0.37,
        "w_bonus":0.05,
        "embed_batch": 48,
        "mmr_lambda": 0.70,
        "mmr_take": 60
    },
    "embedder": {
        "name": "pritamdeka/S-PubMedBERT-MS-MARCO"  # biomedical sentence embeddings
    }
}
os.makedirs(CFG["paths"]["cache"], exist_ok=True)

# ---------- Helpers ----------
BASIC_STOP = set("""
a an and or but the is are was were be been being of in on at to for from with without vs versus compared
as by into about across after before between within than which who whom whose what when where why how
do does did done doing not no nor exclude excluding except only more most less least over under this these those
such same other another each any all some many much several few per via if then else since
""".split())

def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _tok(s: str) -> List[str]:
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[’']", "'", s)
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"([:;,.?!/])", r" \1 ", s)
    return [t for t in re.split(r"\s+", s) if t]

def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        k = json.dumps(x, sort_keys=True) if isinstance(x,(dict,list)) else str(x).lower().strip()
        if k not in seen:
            out.append(x); seen.add(k)
    return out

def _minmax(a: np.ndarray) -> np.ndarray:
    if a.size == 0: return a
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12: return np.ones_like(a)*0.5
    return (a - mn) / (mx - mn)

def _q_ta(phrase: str) -> str:
    phrase = _norm(phrase)
    return f"\"{phrase}\"[Title/Abstract]" if len(phrase.split()) > 1 else f"{phrase}[Title/Abstract]"

def parse_time(q: str) -> List[str]:
    tags = []
    m = re.search(r"\bsince\s+((?:19|20)\d{2})\b", q, flags=re.I)
    if m: tags.append(f'year>={m.group(1)}')
    m = re.search(r"\bbefore\s+((?:19|20)\d{2})\b", q, flags=re.I)
    if m: tags.append(f'year<={m.group(1)}')
    m = re.search(r"\bbetween\s+((?:19|20)\d{2})\s+(?:and|-|to)\s+((?:19|20)\d{2})\b", q, flags=re.I)
    if m: tags += [f'year>={m.group(1)}', f'year<={m.group(2)}']
    return _dedupe(tags)

def _pdat_clause(time_tags: List[str]) -> str:
    ge = [re.findall(r"\d{4}", t)[0] for t in time_tags if t.startswith("year>=")]
    le = [re.findall(r"\d{4}", t)[0] for t in time_tags if t.startswith("year<=")]
    if ge or le:
        lo = ge[0] if ge else "0001"; hi = le[0] if le else "3000"
        return f' AND ("{lo}"[PDAT] : "{hi}"[PDAT])'
    return ""

# ----- Stronger NOT handling -----
def _not_clause(exclusions: List[str]) -> str:
    if not exclusions: return ""
    toks = []
    for e in exclusions[:6]:
        for part in re.split(r"[\/,&;]|(?:\bor\b)|(?:\band\b)|,|\||:", _norm(e), flags=re.I):
            term = _canon_token(part)
            if term:
                toks.append(f"({_q_ta(term)} OR \"{term}\"[MeSH Terms])")
    toks = _dedupe(toks)
    return (" NOT (" + " OR ".join(toks) + ")") if toks else ""

def parse_excl(q: str) -> List[str]:
    ex = []
    for kw in ("exclude","excluding","without","not"):
        for m in re.finditer(rf"\b{kw}\s+([^.;,]+)", q, flags=re.I):
            ex.append(_norm(m.group(1)))
    return _dedupe(ex)

# ---------- Token canon + grouping ----------
EXCEPT_SINGULAR = {"diabetes","tuberculosis","measles","herpes","mumps","shingles"}  # don't strip 's'

def _canon_token(t: str) -> str:
    t = _norm(t).lower()
    t = t.replace("-", " ")
    if t not in EXCEPT_SINGULAR:
        t = re.sub(r"\b([a-z]+)ies\b", r"\1y", t)
        t = re.sub(r"\b([a-z]+)s\b", r"\1", t)
    return t

def _split_mixed(token: str) -> List[str]:
    parts = re.split(r"[\/;&]|,|(?:\bor\b)|(?:\band\b)|；|、", token, flags=re.I)
    out = []
    for p in parts:
        p = _canon_token(p)
        if p and p not in BASIC_STOP and len(p) >= 2:
            out.append(p)
    return _dedupe(out)

def _token_bigram_set(t: str) -> set:
    t = re.sub(r"[^a-z0-9 ]", " ", _canon_token(t))
    t = re.sub(r"\s+", " ", t).strip()
    return {t[i:i+2] for i in range(len(t)-1)} if len(t) > 1 else {t}

def _near(a: str, b: str, thresh: float = 0.42) -> bool:
    A, B = _token_bigram_set(a), _token_bigram_set(b)
    if not A or not B: return False
    overlap = len(A & B) / max(1, len(A | B))
    return overlap >= thresh

# ---------- Perplexity ----------
def pplx_chat(messages: List[Dict[str,str]], model: str, search_mode: Optional[str]) -> Dict[str, Any]:
    if not PPLX_API_KEY or PPLX_API_KEY.startswith("REPLACE_"):
        raise RuntimeError("Perplexity key missing. Set PPLX_API_KEY.")
    headers = {"Authorization": f"Bearer {PPLX_API_KEY}", "Content-Type":"application/json"}
    body = {"model": model, "messages": messages, "temperature": CFG["pplx"]["temperature"],
            "max_tokens": CFG["pplx"]["max_tokens"], "stream": False}
    if search_mode in ("web","academic","sec"):
        body["search_mode"] = search_mode
    r = requests.post(CFG["pplx"]["url"], headers=headers, data=json.dumps(body), timeout=120)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(f"Perplexity API error: HTTP {r.status_code}\n{r.text[:1200]}")
    return r.json()

def pplx_plan(query: str) -> Dict[str, Any]:
    system = textwrap.dedent("""\
        You are a biomedical retrieval planner. Output STRICT JSON only.

        GOAL: Create minimal, precise PubMed search units that don't mix unrelated concepts.

        WHAT TO RETURN (compact):
        - "chunks": 1–3-word tokens *taken from user text only* (no invented long phrases).
          Return 3–5 chunks when possible; if the query is very focused, 1 is acceptable.
          Lowercase; avoid stopwords/connectors.
        - "anchors": concept-separated tokens: Disease, Intervention, Comparator, Outcome, Population, Context.
        - "mesh_terms": PubMed MeSH/synonym anchors per concept (short tokens).
        - "generic_rules": For generic tokens (therapy, treatment, medicines, management), add must_pair_with_any_of (e.g., ["Disease","Intervention"]) so they NEVER search alone.
        - "close_relatives": OPTIONAL dictionary mapping a token to 1–3 near-synonyms *within the same concept*.

        RULES:
        1) Do NOT mix different diseases/conditions or systems into one token.
        2) Prefer disease names, drug/classes, and outcome words; keep tokens brief.
        3) Split combos like "mi/stroke" into separate tokens.
        4) Keep lists short (≤10 items per list); omit if unsure.

        JSON schema:
        {
          "anchors": { "Disease":[], "Intervention":[], "Comparator":[], "Outcome":[], "Population":[], "Context":[] },
          "mesh_terms": { "Disease":[], "Intervention":[], "Outcome":[], "Other":[] },
          "chunks": ["..."],
          "generic_rules": [ { "token":"therapy", "must_pair_with_any_of":["Disease","Intervention"] } ],
          "close_relatives": { "tokenA": ["syn1","syn2"], "tokenB": ["synX"] }
        }
    """).strip()
    user = json.dumps({"query": query}, ensure_ascii=False)
    jr = pplx_chat(
        [{"role":"system","content":system},{"role":"user","content":user}],
        model=CFG["pplx"]["model_plan"],
        search_mode=CFG["pplx"]["search_mode_plan"]
    )
    content = (jr.get("choices",[{}])[0].get("message",{}) or {}).get("content","")
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        toks = [t for t in _tok(query) if t.lower() not in BASIC_STOP]
        toks = [t for t in toks if len(t) <= 20]
        return {"anchors":{k:[] for k in ["Disease","Intervention","Comparator","Outcome","Population","Context"]},
                "mesh_terms":{k:[] for k in ["Disease","Intervention","Outcome","Other"]},
                "chunks": _dedupe(toks[:5]) or _dedupe(toks[:1]),
                "generic_rules": [], "close_relatives": {}, "notes": "fallback"}
    try:
        js = json.loads(m.group(0))
    except Exception:
        js = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))

    def short_list(x, maxn=10):
        out = []
        for v in (x or []):
            v = _norm(str(v))
            if v and len(v.split()) <= 3:
                out.append(v.lower())
        return _dedupe(out)[:maxn]

    js["chunks"] = short_list(js.get("chunks", []), maxn=5) or short_list(_tok(query), maxn=1)
    for k in ["Disease","Intervention","Comparator","Outcome","Population","Context"]:
        js.setdefault("anchors", {}).setdefault(k, [])
        js["anchors"][k] = short_list(js["anchors"][k], maxn=8)
    for k in ["Disease","Intervention","Outcome","Other"]:
        js.setdefault("mesh_terms", {}).setdefault(k, [])
        js["mesh_terms"][k] = short_list(js["mesh_terms"][k], maxn=12)
    js["generic_rules"] = js.get("generic_rules", []) or []
    js["close_relatives"] = { _canon_token(k): short_list(v, maxn=3)
                              for k, v in (js.get("close_relatives", {}) or {}).items() }
    return js

# ---------- PubMed client (cached) ----------
def _ckey(url: str, params: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(url.encode()); h.update(json.dumps(params, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()

def _read_cache(path: str):
    try:
        with open(path,"r",encoding="utf-8") as f: return f.read()
    except Exception: return None

def _get_json(url: str, params: Dict[str, Any], ttl=86400) -> Dict[str, Any]:
    key = _ckey(url, params); path = os.path.join(CFG["paths"]["cache"], key + ".json")
    if os.path.exists(path):
        try:
            data=json.loads(_read_cache(path))
            if time.time()-data.get("_t",0)<=ttl: return data["payload"]
        except Exception: pass
    time.sleep(CFG["pubmed"]["sleep"])
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    payload = r.json()
    with open(path,"w",encoding="utf-8") as f: json.dump({"_t":time.time(),"payload":payload}, f)
    return payload

def _get_text(url: str, params: Dict[str, Any], ttl=86400) -> str:
    key = _ckey(url, params); path = os.path.join(CFG["paths"]["cache"], key + ".xml")
    if os.path.exists(path):
        txt=_read_cache(path)
        if txt is not None: return txt
    time.sleep(CFG["pubmed"]["sleep"])
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    txt = r.text
    with open(path,"w",encoding="utf-8") as f: f.write(txt)
    return txt

def pubmed_esearch(term: str, retmax: int) -> List[str]:
    url = CFG["pubmed"]["base"] + "esearch.fcgi"
    params = {"db":"pubmed","retmode":"json","term":term,"retmax":str(retmax),
              "tool":CFG["pubmed"]["tool"],"email":CFG["pubmed"]["email"]}
    if PUBMED_API_KEY and not PUBMED_API_KEY.startswith("REPLACE_"):
        params["api_key"]=PUBMED_API_KEY
    js = _get_json(url, params)
    return js.get("esearchresult", {}).get("idlist", [])

def pubmed_esummary(pmids: List[str]) -> Dict[str, Any]:
    if not pmids: return {}
    url = CFG["pubmed"]["base"] + "esummary.fcgi"
    params = {"db":"pubmed","retmode":"json","id":",".join(pmids),
              "tool":CFG["pubmed"]["tool"],"email":CFG["pubmed"]["email"]}
    if PUBMED_API_KEY and not PUBMED_API_KEY.startswith("REPLACE_"):
        params["api_key"]=PUBMED_API_KEY
    js = _get_json(url, params, ttl=7*86400)
    return js.get("result", {})

def pubmed_efetch(pmids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not pmids: return {}
    url = CFG["pubmed"]["base"] + "efetch.fcgi"
    params = {"db":"pubmed","retmode":"xml","rettype":"abstract","id":",".join(pmids),
              "tool":CFG["pubmed"]["tool"],"email":CFG["pubmed"]["email"]}
    if PUBMED_API_KEY and not PUBMED_API_KEY.startswith("REPLACE_"):
        params["api_key"]=PUBMED_API_KEY
    xml = _get_text(url, params, ttl=14*86400)
    out: Dict[str, Dict[str, Any]] = {}
    for block in re.split(r"</PubmedArticle>", xml):
        m = re.search(r"<PMID[^>]*>(\d+)</PMID>", block)
        if not m: continue
        pid = m.group(1)
        abstr = re.findall(r"<AbstractText[^>]>(.*?)</AbstractText>", block, flags=re.S)
        clean = _norm(re.sub(r"<[^>]+>", " ", " ".join(abstr)))
        ptypes = re.findall(r"<PublicationType>(.*?)</PublicationType>", block)
        ptypes = [re.sub(r"<[^>]+>","",p).strip() for p in ptypes]
        out[pid] = {"abstract": clean, "pubtypes": ptypes}
    return out

# ---------- Data classes ----------
@dataclass
class Doc:
    pmid: str
    title: str
    journal: str
    year: Optional[int]
    abstract: str
    pubtypes: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    chunk_from: str = ""

@dataclass
class Bucket:
    chunk: str          # group label (e.g., "statin / hmg coa reductase inhibitor")
    boolean: str
    docs: List[Doc]
    note: str = ""

# ---------- Embeddings ----------
_EMB = None
_EMB_DIM = None
def get_embedder():
    global _EMB, _EMB_DIM
    if _EMB is None:
        _EMB = SentenceTransformer(CFG["embedder"]["name"], device="cpu")
        try:
            _EMB_DIM = _EMB.get_sentence_embedding_dimension()
        except Exception:
            _EMB_DIM = None
        print(f"Loaded embedder: {CFG['embedder']['name']} (CPU)")
    return _EMB

def embed_texts(texts: List[str], batch: int = None) -> np.ndarray:
    if not texts:
        dim = _EMB_DIM if _EMB_DIM else 384
        return np.zeros((0, dim), dtype=np.float32)
    model = get_embedder()
    vecs = []
    b = batch or CFG["rank"]["embed_batch"]
    for i in range(0, len(texts), b):
        v = model.encode(texts[i:i+b], normalize_embeddings=True)
        v = np.asarray(v, dtype=np.float32)
        vecs.append(v)
    out = np.vstack(vecs)
    return out

def bm25_scores(query: str, docs: List[str]) -> np.ndarray:
    tok = lambda x: re.findall(r"\w+", x.lower())
    bm = BM25Okapi([tok(d) for d in docs]) if docs else None
    return np.array(bm.get_scores(tok(query)), dtype=np.float32) if bm else np.zeros((0,),dtype=np.float32)

# ---------- Chunk compression ----------
def compress_chunks_with_relatives(plan: Dict[str,Any]) -> List[List[str]]:
    chunks = plan.get("chunks", []) or []
    rel = plan.get("close_relatives", {}) or {}
    anchors = plan.get("anchors", {}); mesh = plan.get("mesh_terms", {})

    def _concept_of(t: str) -> str:
        for c in ["Disease","Intervention","Outcome","Comparator","Population","Context"]:
            cset = [ _canon_token(x) for x in (anchors.get(c) or []) + (mesh.get(c) or []) ]
            if _canon_token(t) in cset:
                return c
        return "Generic"

    seeds = []
    for ch in chunks:
        for unit in _split_mixed(ch):
            seeds.append(unit)
    seeds = _dedupe([_canon_token(s) for s in seeds])

    groups: List[List[str]] = []
    seen = set()
    for s in seeds:
        if s in seen: continue
        c = _concept_of(s)
        candidates = [s]

        for r in rel.get(s, []):
            r = _canon_token(r)
            if r and (_concept_of(r) == c or c == "Generic") and _near(s, r):
                candidates.append(r)

        for t in seeds:
            if t == s or t in candidates: continue
            if _near(s, t) and (_concept_of(t) == c or c == "Generic"):
                candidates.append(t)

        candidates = _dedupe(candidates)[:6]
        for t in candidates: seen.add(t)
        groups.append(candidates)

    return groups[:12] if groups else [chunks[:1] or ["query"]]

# ---------- Boolean builder ----------
def _block_ta_mesh(items: List[str]) -> str:
    items = [i for i in (items or []) if i]
    if not items: return ""
    ta_b = " OR ".join(_q_ta(i) for i in items[:8])
    mh_b = " OR ".join(f"\"{i}\"[MeSH Terms]" for i in items[:8])
    return "(" + ta_b + (f" OR {mh_b}" if mh_b else "") + ")"

def build_boolean_for_group(group: List[str], plan: Dict[str,Any],
                            time_tags: List[str], exclusions: List[str]) -> str:
    anchors = plan.get("anchors", {}); mesh = plan.get("mesh_terms", {})
    generic_rules = plan.get("generic_rules", [])

    ta = " OR ".join(_q_ta(t) for t in group)
    mh = " OR ".join(f"\"{t}\"[MeSH Terms]" for t in group)
    core = f"(({ta})" + (f" OR ({mh}))" if mh else "") + ")"

    must_pair_with = []
    if len(group) == 1:
        g0 = group[0]
        for rule in generic_rules:
            if _canon_token(rule.get("token","")) == _canon_token(g0):
                must_pair_with = rule.get("must_pair_with_any_of", []) or []
                break

    disease_block = _block_ta_mesh(_dedupe((anchors.get("Disease") or []) + (mesh.get("Disease") or [])))
    interv_block  = _block_ta_mesh(_dedupe((anchors.get("Intervention") or []) + (mesh.get("Intervention") or [])))
    outcome_block = _block_ta_mesh(_dedupe((anchors.get("Outcome") or []) + (mesh.get("Outcome") or [])))

    clauses = [core]
    if must_pair_with:
        pair = []
        if "Disease" in must_pair_with and disease_block: pair.append(disease_block)
        if "Intervention" in must_pair_with and interv_block: pair.append(interv_block)
        if "Outcome" in must_pair_with and outcome_block:   pair.append(outcome_block)
        if pair:
            clauses = [f"({core} AND (" + " OR ".join(pair) + "))"]
    else:
        looks_like_dz = any(_canon_token(x) in group for x in (anchors.get("Disease") or []) + (mesh.get("Disease") or []))
        if not looks_like_dz and disease_block:
            clauses.append(disease_block)

    boolean = " AND ".join([c for c in clauses if c]) + _pdat_clause(time_tags) + _not_clause(exclusions)
    return boolean.strip()

# ---------- Retrieval per group ----------
@dataclass
class RunBucket:
    chunk: str
    boolean: str
    note: str
    docs: List[Dict[str, Any]]

def retrieve_for_group(group: List[str], plan: Dict[str,Any], time_tags: List[str], exclusions: List[str]) -> Bucket:
    boolean = build_boolean_for_group(group, plan, time_tags, exclusions)
    ids = pubmed_esearch(boolean, CFG["pubmed"]["retmax_main"])
    if not ids:
        return Bucket(chunk="/".join(group), boolean=boolean, docs=[], note="0 candidates")

    sm = pubmed_esummary(ids)
    id_order, titles, meta = [], [], {}
    for pid in ids:
        e = sm.get(pid, {}); title = (e.get("title","") or "").strip()
        if not title: continue
        jr = e.get("fulljournalname","") or e.get("source","")
        pubdate = e.get("pubdate",""); m = re.search(r"\b(19|20)\d{2}\b", pubdate)
        yr = int(m.group(0)) if m else None
        id_order.append(pid); titles.append(title); meta[pid]={"title":title,"journal":jr,"year":yr}

    if not titles:
        return Bucket(chunk="/".join(group), boolean=boolean, docs=[], note="no titles")

    bm = bm25_scores(" ".join(group), titles)
    order = np.argsort(-bm)[:max(60, CFG["retrieval"]["per_group_keep"]*3)]
    shortlist = [id_order[i] for i in order]
    abs_map = pubmed_efetch(shortlist)

    docs: List[Doc] = []
    for pid in shortlist:
        if pid not in meta: continue
        m = meta[pid]
        abstr = (abs_map.get(pid, {}) or {}).get("abstract","")
        ptypes = (abs_map.get(pid, {}) or {}).get("pubtypes", [])
        docs.append(Doc(
            pmid=pid, title=m.get("title",""), journal=m.get("journal",""),
            year=m.get("year"), abstract=abstr, pubtypes=ptypes,
            scores={"bm25_chunk": float(bm[list(id_order).index(pid)])},
            chunk_from="/".join(group)
        ))
        if len(docs) >= CFG["retrieval"]["per_group_keep"]:
            break
    return Bucket(chunk="/".join(group), boolean=boolean, docs=docs, note=f"kept={len(docs)} from {len(ids)}")

# ---------- Fusion + MMR ----------
def bonus_pubtypes(ptypes: List[str]) -> float:
    pt = [p.lower() for p in ptypes]
    b = 0.0
    if any("random" in p for p in pt): b += 0.6
    if any(("meta" in p) or ("systematic" in p) for p in pt): b += 0.7
    return b

def bonus_guideline(d: Doc) -> float:
    t = (d.title or "").lower(); j = (d.journal or "").lower()
    if any(k in t for k in ["guideline","consensus","statement"]) or any(k in j for k in ["aha","esc","acc","circulation","european heart journal"]):
        return 0.5
    return 0.0

def mmr_indices(q_vec: np.ndarray, d_vecs: np.ndarray, lam: float, take: int) -> List[int]:
    N = d_vecs.shape[0]
    if N == 0: return []
    sims = (d_vecs @ q_vec.reshape(-1,1)).ravel()
    selected = [int(np.argmax(sims))]
    remaining = set(range(N)) - set(selected)
    dd = d_vecs @ d_vecs.T
    while remaining and len(selected) < take:
        best, best_score = None, -1e9
        for i in list(remaining):
            redundancy = max(dd[i, j] for j in selected) if selected else 0.0
            score = lam*sims[i] - (1-lam)*redundancy
            if score > best_score:
                best, best_score = i, score
        selected.append(best); remaining.remove(best)
    return selected

def fuse_and_rerank(query: str, buckets: List[Bucket], final_take: int) -> List[Doc]:
    pool: Dict[str, Doc] = {}
    for b in buckets:
        for d in b.docs:
            if d.pmid not in pool or d.scores.get("bm25_chunk",0) > pool[d.pmid].scores.get("bm25_chunk",0):
                pool[d.pmid] = d
    docs = list(pool.values())
    if not docs: return []

    texts = [(d.title + " " + d.abstract).strip() for d in docs]
    bm = bm25_scores(query, texts)
    q_vec = embed_texts([query])[0]
    d_vecs = embed_texts(texts)
    emb = (d_vecs @ q_vec.reshape(-1,1)).ravel().astype(np.float32)
    bonuses = np.array([bonus_pubtypes(d.pubtypes)+bonus_guideline(d) for d in docs], dtype=np.float32)

    fused = (CFG["rank"]["w_bm25"]*_minmax(bm) +
             CFG["rank"]["w_emb"] *_minmax(emb) +
             CFG["rank"]["w_bonus"]*_minmax(bonuses))

    topK = min(CFG["rank"]["mmr_take"], len(docs))
    order = np.argsort(-fused)[:topK]
    d_top = [docs[i] for i in order]
    v_top = d_vecs[order]
    picks = mmr_indices(q_vec, v_top, CFG["rank"]["mmr_lambda"], take=min(final_take, topK))
    final = [d_top[i] for i in picks]

    for i, d in enumerate(docs):
        d.scores.update({
            "bm25_intent": float(bm[i]),
            "emb_sim": float(emb[i]),
            "bonus": float(bonuses[i]),
            "fused_raw": float(fused[i])
        })
    return final

# ---------- Perplexity summarization (no browsing) ----------
def build_evidence_pack(docs: List[Doc], max_docs: int = 15, abstract_chars: int = 1400) -> str:
    chosen = docs[:max_docs]; lines=[]
    for i,d in enumerate(chosen,1):
        head=f"[{i}] PMID {d.pmid} ({d.year}) {d.journal} — {d.title}".strip(" —")
        body=f"Abstract: {_norm(d.abstract)[:abstract_chars]}" if d.abstract else "Abstract: (not available)"
        lines.append(f"{head}\n{body}")
    return "EVIDENCE PACK\n" + "\n\n".join(lines)

def pplx_summarize(query: str, docs: List[Doc]) -> Dict[str, Any]:
    if not docs: return {"error":"No docs to summarize"}
    pack = build_evidence_pack(docs, max_docs=CFG["retrieval"]["final_take"], abstract_chars=1400)
    system = ("You are an evidence synthesis assistant. Use ONLY the EVIDENCE PACK below. "
              "Do NOT browse or search. Every factual sentence MUST include bracket citations like [1][2]. "
              "Prefer randomized trials, meta-analyses, and guidelines when present. Be precise and concise.")
    schema = {"question":"string","answer":{"conclusion":"string","key_findings":"array","quality_and_limits":"array","evidence_citations":"array"}}
    user = f"""QUESTION
{query}

{pack}

TASK
Answer ONLY from the EVIDENCE PACK with bracket citations [n]. Return compact JSON:
{json.dumps(schema, indent=2)}"""
    jr = pplx_chat(
        [{"role":"system","content":system},{"role":"user","content":user}],
        model=CFG["pplx"]["model_summary"],
        search_mode=CFG["pplx"]["search_mode_summary"]
    )
    content = (jr.get("choices",[{}])[0].get("message",{}) or {}).get("content","")
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m: return {"raw": content}
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
    except Exception:
        return {"raw": content}

# ---------- Orchestration ----------
def run_pipeline(query: str, verbose: bool = True) -> Dict[str, Any]:
    time_tags = parse_time(query)
    exclusions = parse_excl(query)
    plan = pplx_plan(query)
    groups = compress_chunks_with_relatives(plan)

    buckets: List[RunBucket] = []
    all_docs: List[Doc] = []
    seen = set()
    for grp in groups:
        b = retrieve_for_group(grp, plan, time_tags, exclusions)
        uniq = []
        for d in b.docs:
            if d.pmid in seen: continue
            seen.add(d.pmid); uniq.append(d); all_docs.append(d)
        buckets.append(RunBucket(chunk=" / ".join(grp), boolean=b.boolean, note=b.note, docs=[asdict(x) for x in uniq]))

    if len(all_docs) < CFG["retrieval"]["min_docs_wanted"]:
        disease_terms = _dedupe((plan.get("anchors",{}).get("Disease") or []) + (plan.get("mesh_terms",{}).get("Disease") or []))
        if disease_terms:
            dz_block = _block_ta_mesh(disease_terms[:8])
            neutral = dz_block + _pdat_clause(time_tags) + _not_clause(exclusions)
        else:
            base = " OR ".join(_q_ta(t) for t in _dedupe([t for t in _tok(query) if t.lower() not in BASIC_STOP][:6]))
            neutral = "(" + base + ")" + _pdat_clause(time_tags) + _not_clause(exclusions)

        ids = pubmed_esearch(neutral, CFG["pubmed"]["retmax_backoff"])
        sm = pubmed_esummary(ids)
        id_order, titles, meta = [], [], {}
        for pid in ids:
            e = sm.get(pid,{}); title = (e.get("title","") or "").strip()
            if not title: continue
            j = e.get("fulljournalname","") or e.get("source","")
            pubdate = e.get("pubdate",""); m = re.search(r"\b(19|20)\d{2}\b", pubdate)
            yr = int(m.group(0)) if m else None
            id_order.append(pid); titles.append(title); meta[pid]={"title":title,"journal":j,"year":yr}
        bm = bm25_scores(query, titles) if titles else np.array([])
        order = np.argsort(-bm)[:120] if titles else []
        sel = [id_order[i] for i in order]
        abs_map = pubmed_efetch(sel) if sel else {}
        back_docs=[]
        for pid in sel:
            if pid in seen: continue
            seen.add(pid)
            m = meta.get(pid,{})
            abstr = (abs_map.get(pid, {}) or {}).get("abstract","")
            ptypes = (abs_map.get(pid, {}) or {}).get("pubtypes", [])
            back_docs.append(Doc(pmid=pid, title=m.get("title",""), journal=m.get("journal",""),
                                 year=m.get("year"), abstract=abstr, pubtypes=ptypes,
                                 scores={"bm25_chunk":0.0}, chunk_from="[backoff]"))
        buckets.append(RunBucket(chunk="[backoff]", boolean=neutral, note=f"added {len(back_docs)}", docs=[asdict(x) for x in back_docs]))
        all_docs.extend(back_docs)

    final_docs = fuse_and_rerank(query, [Bucket(b.chunk, b.boolean, [Doc(**d) for d in b.docs], b.note) for b in buckets],
                                 final_take=CFG["retrieval"]["final_take"])

    summary = pplx_summarize(query, final_docs)

    return {
        "plan": plan,
        "time_tags": time_tags,
        "exclusions": exclusions,
        "buckets": [asdict(b) for b in buckets],
        "final_docs": [asdict(d) for d in final_docs],
        "summary": summary
    }


# =================================================================================================
# Thin wrapper for /ask → (assistant_text, right_pane)
# =================================================================================================

def _role_heading(role: Optional[str]) -> str:
    if not role: return "Answer"
    role = str(role).replace("_", " ").title()
    return f"{role} Summary"

def _pmid_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

def _mk_assistant_text(question: str, role: Optional[str], pipe: Dict[str, Any]) -> str:
    """
    Build the chat message text from the pipeline summary without changing the core logic.
    """
    ans = pipe.get("summary") or {}
    plan = pipe.get("plan") or {}
    heading = _role_heading(role)

    # Try to read structured answer
    if isinstance(ans, dict) and "answer" in ans:
        a = ans["answer"] or {}
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
            # keep [n] mapping as-is
            lines.append(", ".join([f"[{c}]" for c in cites]))

        # tiny breadcrumb for transparency
        if plan.get("chunks"):
            lines.append(f"\n_(query chunks: {', '.join(plan['chunks'])})_")

        return "\n".join([s for s in lines if s and s.strip()])

    # Fallback: raw content from model
    raw = ans.get("raw") if isinstance(ans, dict) else ""
    if raw:
        return f"### {heading}\n{raw}"
    return f"### {heading}\n(no summary generated)"

def _mk_right_pane(pipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map pipeline output into the UI-friendly rightPane object.
    """
    final_docs = pipe.get("final_docs") or []
    buckets = pipe.get("buckets") or []
    plan = pipe.get("plan") or {}
    summary = pipe.get("summary") or {}

    # results: RankedDoc[]
    results = []
    for d in final_docs:
        pmid = d.get("pmid")
        results.append({
            "pmid": pmid,
            "title": d.get("title"),
            "journal": d.get("journal"),
            "year": d.get("year"),
            "url": _pmid_url(pmid) if pmid else None,
            "score": float(d.get("scores", {}).get("fused_raw")) if d.get("scores") else None,
        })

    # booleans: BooleanItem[]
    booleans = [{"group": b.get("chunk",""), "query": b.get("boolean",""), "note": b.get("note")} for b in buckets]

    # evidence: EvidenceItem[]
    evidence = []
    for d in final_docs[:15]:
        evidence.append({
            "pmid": d.get("pmid"),
            "year": d.get("year"),
            "journal": d.get("journal"),
            "title": d.get("title"),
            "snippet": (_norm(d.get("abstract",""))[:420] if d.get("abstract") else None),
        })

    # overview: pass through structured JSON if present
    overview = None
    if isinstance(summary, dict) and "answer" in summary:
        overview = {
            "conclusion": (summary["answer"] or {}).get("conclusion", ""),
            "key_findings": (summary["answer"] or {}).get("key_findings", []),
            "quality_and_limits": (summary["answer"] or {}).get("quality_and_limits", []),
        }

    # evidence pack (useful if FE wants to show raw)
    try:
        # rebuild Doc objects to reuse builder
        docs = [Doc(**d) for d in final_docs]
        evidence_pack = build_evidence_pack(docs, max_docs=25, abstract_chars=1200)
    except Exception:
        evidence_pack = None

    return {
        "results": results,
        "booleans": booleans,
        "plan": {
            "chunks": plan.get("chunks", []),
            "time_tags": pipe.get("time_tags", []),
            "exclusions": pipe.get("exclusions", []),
        },
        "evidence": evidence,
        "overview": overview,
        "evidencePack": evidence_pack,
    }

def run_rag_pipeline(question: str, role: Optional[str] = None, verbose: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Public entry used by /ask.
    Returns (assistant_text, right_pane)
    """
    pipe = run_pipeline(question, verbose=verbose)
    assistant_text = _mk_assistant_text(question, role, pipe)
    right_pane = _mk_right_pane(pipe)
    return assistant_text, right_pane
