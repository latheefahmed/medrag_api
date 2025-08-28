# app/services/qdrant_client.py
from __future__ import annotations
import os, time
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np

QDRANT_URL         = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY     = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION", "medrag_pmid_768")
EMBED_DIM          = int(os.getenv("EMBED_DIM", "768") or "768")
USE_EMBED_CACHE    = os.getenv("USE_EMBED_CACHE", "1") == "1"

def _enabled() -> bool:
    return USE_EMBED_CACHE and bool(QDRANT_URL)

@lru_cache()
def _client():
    if not _enabled():
        return None
    try:
        from qdrant_client import QdrantClient
        # prefer HTTP URL (Qdrant Cloud or self-hosted)
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10.0)
        return QdrantClient(url=QDRANT_URL, timeout=10.0)
    except Exception:
        return None

def ensure_collection() -> bool:
    """Create the collection if missing. Returns True on success or if disabled."""
    if not _enabled():
        return True
    client = _client()
    if client is None:
        return False
    try:
        from qdrant_client.http import models as qm
        collections = client.get_collections().collections or []
        if not any(c.name == QDRANT_COLLECTION for c in collections):
            client.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
            )
        return True
    except Exception:
        return False

def upsert_vectors(items: Iterable[Tuple[str, np.ndarray, Dict]]):
    """items: (point_id, vector, payload) best-effort upsert."""
    if not _enabled():
        return
    client = _client()
    if client is None:
        return
    try:
        ensure_collection()
        from qdrant_client.http import models as qm
        pts = []
        for pid, vec, payload in items:
            if vec is None:
                continue
            v = vec.astype(np.float32).tolist()
            pts.append(qm.PointStruct(id=pid, vector=v, payload=payload))
        if pts:
            client.upsert(collection_name=QDRANT_COLLECTION, points=pts, wait=False)
    except Exception:
        # best-effort: ignore all failures
        pass

def get_vectors_by_ids(ids: List[str]) -> Dict[str, np.ndarray]:
    """Return a subset of vectors that exist in the collection. Missing ids are omitted."""
    out: Dict[str, np.ndarray] = {}
    if not _enabled():
        return out
    client = _client()
    if client is None:
        return out
    try:
        ensure_collection()
        # retrieve returns payload+vector
        points = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=ids,
            with_vectors=True,
        ) or []
        for p in points:
            vec = p.vector
            if isinstance(vec, list):
                out[str(p.id)] = np.asarray(vec, dtype=np.float32)
    except Exception:
        pass
    return out
