
from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .qdrant_client import upsert_vectors, get_vectors_by_ids

log = logging.getLogger(__name__)

_EXPECTED_DIM = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))


def _h(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def query_id(query: str) -> str:
    return f"q:{_h((query or '').strip().lower())}"


def doc_id(pmid: str, abstract: str) -> str:
    return f"{pmid}:{_h((abstract or '').strip())}"


def _sanitize_vec(vec: Iterable[float], expected: Optional[int] = _EXPECTED_DIM) -> List[float]:
    """
    Ensure the vector is JSON-serializable (list[float]) and finite.
    We do NOT hard-fail on length mismatch (to avoid interrupting the request),
    but we log a warning so you can spot schema issues immediately.
    """
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if expected and arr.size != int(expected):
        # Do not raise; Qdrant will 400 if wrong anyway. We log to help debugging.
        log.warning("embed_cache: vector length %s != expected %s", arr.size, expected)
    # Replace NaN/Inf to keep Qdrant happy
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return arr.astype(float).tolist()


def _np_or_none(v: Optional[Iterable[float]]) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        return np.asarray(v, dtype=np.float32).reshape(-1)
    except Exception:
        return None


# ---------------- public API (used by rag.py) ----------------
def cache_query_vec(query: str, vec: np.ndarray) -> None:
    """
    Store query vector. This is typically a small in-memory cache in your wrapper,
    but if your qdrant_client persists it, this is still safe.
    """
    try:
        pid = query_id(query)
        v = _sanitize_vec(vec)
        payload = {"type": "query", "qhash": pid, "ts": int(time.time())}
        upsert_vectors([(pid, v, payload)])  # vector MUST be list[float]
    except Exception as e:
        log.debug("cache_query_vec: upsert skipped due to error: %r", e)


def cache_doc_vec(pmid: str, title: str, journal: str, year: int, abstract: str, vec: np.ndarray) -> None:
    """
    Upsert a document vector with lightweight payload. Payload fields are kept to
    JSON-serializable primitives only.
    """
    try:
        pid = doc_id(pmid, abstract)
        v = _sanitize_vec(vec)
        payload = {
            "type": "doc",
            "pmid": str(pmid),
            "title": title or "",
            "journal": journal or "",
            "year": int(year or 0),
            "abs_hash": pid.split(":", 1)[1],
            "ts": int(time.time()),
        }
        upsert_vectors([(pid, v, payload)])  # vector MUST be list[float]
    except Exception as e:
        log.debug("cache_doc_vec: upsert skipped due to error: %r", e)


def try_get_query_vec(query: str) -> Optional[np.ndarray]:
    """
    Return numpy vector if present, else None.
    """
    try:
        pid = query_id(query)
        m = get_vectors_by_ids([pid]) or {}
        return _np_or_none(m.get(pid))
    except Exception:
        return None


def try_get_doc_vecs(pairs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    """
    pairs: list of (pmid, abstract). Returns a dict keyed by doc_id(pmid, abstract)
    with numpy vectors. Missing ids are simply omitted.
    """
    try:
        ids = [doc_id(p, a) for (p, a) in pairs]
        raw = get_vectors_by_ids(ids) or {}
        out: Dict[str, np.ndarray] = {}
        for k, v in raw.items():
            nv = _np_or_none(v)
            if nv is not None:
                out[k] = nv
        return out
    except Exception:
        return {}
