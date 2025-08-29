# app/db.py
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
from functools import lru_cache

from azure.cosmos import CosmosClient, exceptions, PartitionKey
from dotenv import load_dotenv

load_dotenv()

# ---- Config / Defaults ----
DEFAULT_DB = os.getenv("COSMOS_DB", "medrag")
USERS_CONTAINER = os.getenv("COSMOS_USERS_CONTAINER", "users")
SESS_CONTAINER  = os.getenv("COSMOS_SESSIONS_CONTAINER", "sessions")
LOGS_CONTAINER  = os.getenv("COSMOS_LOGS_CONTAINER", "logs")


def _env_ok() -> Tuple[bool, str, str, str]:
    url = (os.getenv("COSMOS_URL") or "").strip()
    key = (os.getenv("COSMOS_KEY") or "").strip()
    db_name = (os.getenv("COSMOS_DB") or DEFAULT_DB).strip()
    return bool(url) and bool(key), url, key, db_name


@lru_cache()
def _handles():
    ok, url, key, db_name = _env_ok()
    if not ok:
        raise RuntimeError("Cosmos env missing: set COSMOS_URL and COSMOS_KEY in .env")

    client = CosmosClient(url, credential=key)

    # Database
    try:
        db = client.create_database_if_not_exists(db_name)
    except exceptions.CosmosResourceExistsError:
        db = client.get_database_client(db_name)

    # users (pk=/email)
    try:
        users = db.create_container_if_not_exists(
            id=USERS_CONTAINER, partition_key=PartitionKey(path="/email"),
        )
    except exceptions.CosmosResourceExistsError:
        users = db.get_container_client(USERS_CONTAINER)

    # sessions (pk=/user_id)
    try:
        sessions = db.create_container_if_not_exists(
            id=SESS_CONTAINER, partition_key=PartitionKey(path="/user_id"),
        )
    except exceptions.CosmosResourceExistsError:
        sessions = db.get_container_client(SESS_CONTAINER)

    # logs (pk=/user_id)
    try:
        logs = db.create_container_if_not_exists(
            id=LOGS_CONTAINER, partition_key=PartitionKey(path="/user_id"),
        )
    except exceptions.CosmosResourceExistsError:
        logs = db.get_container_client(LOGS_CONTAINER)

    return client, db, users, sessions, logs


# ---------- Exposed handles ----------
@lru_cache()
def get_client() -> CosmosClient:
    client, *_ = _handles()
    return client

@lru_cache()
def get_db():
    _, db, *_ = _handles()
    return db

def get_container(name: str):
    client, db, users, sessions, logs = _handles()
    if name in (USERS_CONTAINER, "users"):   return users
    if name in (SESS_CONTAINER,  "sessions"): return sessions
    if name in (LOGS_CONTAINER,  "logs"):     return logs
    try:
        return db.create_container_if_not_exists(id=name, partition_key=PartitionKey(path="/id"))
    except exceptions.CosmosResourceExistsError:
        return db.get_container_client(name)

def get_users_container():    return _handles()[2]
def get_sessions_container(): return _handles()[3]
def get_logs_container():     return _handles()[4]


def _container_pk_path(container) -> str:
    props = container.read()
    pk = props.get("partitionKey") or {}
    paths = pk.get("paths") or []
    return paths[0] if paths else "/id"


# ---------- Cosmos wrappers ----------
def _safe_create_item(container, body, pk_value):
    try:
        return container.create_item(body, partition_key=pk_value)
    except TypeError:
        return container.create_item(body)

def _safe_read_item(container, item_id, pk_value):
    try:
        return container.read_item(item=item_id, partition_key=pk_value)
    except exceptions.CosmosResourceNotFoundError:
        return None
    except TypeError:
        items = list(container.query_items(
            query="SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": item_id}],
            enable_cross_partition_query=True,
        ))
        return items[0] if items else None

def _safe_replace_item(container, item_id, body, pk_value):
    try:
        return container.replace_item(item=item_id, body=body, partition_key=pk_value)
    except TypeError:
        return container.upsert_item(body)

def _safe_delete_item(container, item_id, pk_value):
    try:
        container.delete_item(item=item_id, partition_key=pk_value)
        return True
    except TypeError:
        doc = _safe_read_item(container, item_id, pk_value)
        if not doc:
            return False
        doc["__deleted"] = True
        container.upsert_item(doc)
        return True


def ensure_cosmos() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "env": {
            "has_url": bool(os.getenv("COSMOS_URL")),
            "has_key": bool(os.getenv("COSMOS_KEY")),
            "db": os.getenv("COSMOS_DB") or DEFAULT_DB,
        },
        "users_pk": None,
        "sessions_pk": None,
        "logs_pk": None,
        "ok": False,
    }
    try:
        _, _, users, sessions, logs = _handles()
        info["users_pk"]    = _container_pk_path(users)
        info["sessions_pk"] = _container_pk_path(sessions)
        info["logs_pk"]     = _container_pk_path(logs)
        info["ok"] = True
    except Exception as e:
        info["error"] = str(e)
    return info


# ============================ Users ============================
def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    if not email:
        return None
    email = email.lower().strip()
    users = get_users_container()
    users_pk = _container_pk_path(users)
    pk_value = email if users_pk in ("/email", "/id") else email

    doc = _safe_read_item(users, email, pk_value)
    if doc:
        return doc

    items = list(users.query_items(
        query="SELECT * FROM c WHERE c.email = @e",
        parameters=[{"name": "@e", "value": email}],
        enable_cross_partition_query=True,
    ))
    return items[0] if items else None


def create_user(user: Dict[str, Any]) -> Dict[str, Any]:
    users = get_users_container()
    email = (user.get("email") or "").lower().strip()
    if not email:
        raise ValueError("create_user: 'email' required")

    body = dict(user)
    body["id"] = email
    body["email"] = email
    body.setdefault("role", "student")
    body.setdefault("verified", False)
    body.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    users_pk = _container_pk_path(users)
    pk_value = email if users_pk == "/email" else body["id"] if users_pk == "/id" else email
    return _safe_create_item(users, body, pk_value)


def update_user(email: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    doc = get_user_by_email(email)
    if not doc:
        return None
    users = get_users_container()
    users_pk = _container_pk_path(users)
    doc.update(patch or {})
    pk_value = doc["email"] if users_pk == "/email" else doc["id"]
    return _safe_replace_item(users, doc["id"], doc, pk_value)


def list_users(limit: int = 200) -> List[Dict[str, Any]]:
    users = get_users_container()
    q = "SELECT c.id, c.email, c.role, c.verified, c.created_at FROM c ORDER BY c.created_at DESC"
    items = list(users.query_items(query=q, enable_cross_partition_query=True, max_item_count=limit))
    return items[:limit]


def list_sessions_all(limit: int = 200) -> List[Dict[str, Any]]:
    sessions = get_sessions_container()
    q = "SELECT c.id, c.user_id, c.title, c.updated_at FROM c ORDER BY c.updated_at DESC"
    items = list(sessions.query_items(query=q, enable_cross_partition_query=True, max_item_count=limit))
    return items[:limit]


# ============================ Sessions ============================
def create_session(sess: Dict[str, Any]) -> Dict[str, Any]:
    sessions = get_sessions_container()
    user_id = (sess.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("create_session: 'user_id' required")

    now = datetime.now(timezone.utc).isoformat()
    body = {
        "id": sess.get("id") or str(uuid4()),
        "user_id": user_id,
        "title": sess.get("title") or "Untitled",
        "messages": sess.get("messages") or [],
        "meta": sess.get("meta") or {},
        "created_at": now,
        "updated_at": now,
    }

    sess_pk = _container_pk_path(sessions)
    pk_value = user_id if sess_pk == "/user_id" else body["id"] if sess_pk == "/id" else user_id
    return _safe_create_item(sessions, body, pk_value)


def get_session_by_id(session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    sessions = get_sessions_container()
    sess_pk = _container_pk_path(sessions)
    pk_value = user_id if sess_pk == "/user_id" else session_id if sess_pk == "/id" else user_id

    doc = _safe_read_item(sessions, session_id, pk_value)
    if doc:
        return doc

    items = list(sessions.query_items(
        query="SELECT * FROM c WHERE c.id = @id AND c.user_id = @u",
        parameters=[{"name": "@id", "value": session_id}, {"name": "@u", "value": user_id}],
        enable_cross_partition_query=True,
    ))
    return items[0] if items else None


def list_sessions_for_user(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    sessions = get_sessions_container()
    items = list(sessions.query_items(
        query="SELECT * FROM c WHERE c.user_id = @u ORDER BY c.updated_at DESC",
        parameters=[{"name": "@u", "value": user_id}],
        enable_cross_partition_query=True,
        max_item_count=limit,
    ))
    return items[:limit]


def update_session(session_id: str, user_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    doc = get_session_by_id(session_id, user_id)
    if not doc:
        return None
    sessions = get_sessions_container()
    doc.update(patch or {})
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()

    sess_pk = _container_pk_path(sessions)
    pk_value = user_id if sess_pk == "/user_id" else session_id if sess_pk == "/id" else user_id
    return _safe_replace_item(sessions, session_id, doc, pk_value)


def delete_session(session_id: str, user_id: str) -> bool:
    sessions = get_sessions_container()
    sess_pk = _container_pk_path(sessions)
    pk_value = user_id if sess_pk == "/user_id" else session_id if sess_pk == "/id" else user_id
    return _safe_delete_item(sessions, session_id, pk_value)


# ---------- Backwards-compat shims ----------
def list_sessions(user_id: str, limit: int = 50):
    return list_sessions_for_user(user_id, limit)

def get_session(session_id: str, user_id: str):
    return get_session_by_id(session_id, user_id)

def upsert_session(session_id: str, user_id: str, patch: dict | None = None):
    patch = patch or {}
    existing = get_session_by_id(session_id, user_id)
    if existing:
        return update_session(session_id, user_id, patch)
    else:
        return create_session({
            "id": session_id,
            "user_id": user_id,
            "title": patch.get("title", "Untitled"),
            "messages": patch.get("messages", []),
            "meta": patch.get("meta", {}),
        })


# ============================ Logs ============================
def create_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stores a Chat log row:
      { id, user_id, session_id, query, response, fetch_ms, summarize_ms, n_results, ts }
    Partition key: /user_id
    """
    logs = get_logs_container()
    user_id = (entry.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("create_log: 'user_id' required")

    body = {
        "id": entry.get("id") or str(uuid4()),
        "user_id": user_id,
        "session_id": entry.get("session_id") or "",
        "query": entry.get("query") or "",
        "response": (entry.get("response") or "")[:16000],  # trim to keep doc small
        "fetch_ms": int(entry.get("fetch_ms") or 0),
        "summarize_ms": int(entry.get("summarize_ms") or 0),
        "n_results": int(entry.get("n_results") or 0),
        "ts": entry.get("ts") or datetime.now(timezone.utc).isoformat(),
    }
    logs_pk = _container_pk_path(logs)
    pk_value = user_id if logs_pk == "/user_id" else body["id"]
    return _safe_create_item(logs, body, pk_value)


def list_logs_for_user(user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    logs = get_logs_container()
    items = list(logs.query_items(
        query="SELECT * FROM c WHERE c.user_id = @u ORDER BY c.ts DESC",
        parameters=[{"name": "@u", "value": user_id}],
        enable_cross_partition_query=True,
        max_item_count=limit,
    ))
    return items[:limit]

def write_chat_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts the fields used by /ask and maps them to create_log’s schema.
    - retrieval_ms -> fetch_ms
    - summary_ms   -> summarize_ms
    - refs         -> n_results
    - answer_preview -> response
    Everything else is passed through.
    """
    # normalize known keys without losing anything else
    e = dict(entry or {})
    if "retrieval_ms" in e and "fetch_ms" not in e:
        e["fetch_ms"] = int(e.get("retrieval_ms") or 0)
    if "summary_ms" in e and "summarize_ms" not in e:
        e["summarize_ms"] = int(e.get("summary_ms") or 0)
    if "refs" in e and "n_results" not in e:
        e["n_results"] = int(e.get("refs") or 0)
    if "answer_preview" in e and "response" not in e:
        e["response"] = str(e.get("answer_preview") or "")[:16000]
    return create_log(e)
