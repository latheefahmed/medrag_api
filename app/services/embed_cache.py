# app/services/embed_cache.py
from __future__ import annotations
import hashlib, time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .qdrant_client import upsert_vectors, get_vectors_by_ids

def _h(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def query_id(query: str) -> str:
    return f"q:{_h((query or '').strip().lower())}"

def doc_id(pmid: str, abstract: str) -> str:
    return f"{pmid}:{_h((abstract or '').strip())}"

def cache_query_vec(query: str, vec: np.ndarray):
    try:
        pid = query_id(query)
        upsert_vectors([(pid, vec, {"type": "query", "qhash": pid, "ts": int(time.time())})])
    except Exception:
        pass

def cache_doc_vec(pmid: str, title: str, journal: str, year: int, abstract: str, vec: np.ndarray):
    try:
        pid = doc_id(pmid, abstract)
        upsert_vectors([(
            pid,
            vec,
            {
                "type": "doc",
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "abs_hash": pid.split(":",1)[1],
                "ts": int(time.time()),
            },
        )])
    except Exception:
        pass

def try_get_query_vec(query: str) -> Optional[np.ndarray]:
    try:
        pid = query_id(query)
        m = get_vectors_by_ids([pid])
        return m.get(pid)
    except Exception:
        return None

def try_get_doc_vecs(pairs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    """
    pairs: list of (pmid, abstract). Returns a dict keyed by doc_id(pmid, abstract).
    """
    try:
        ids = [doc_id(p, a) for (p, a) in pairs]
        return get_vectors_by_ids(ids)
    except Exception:
        return {}
