# app/schemas.py
from typing import Literal, List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr

# ---- Auth ----
class SignupInput(BaseModel):
    email: EmailStr
    password: str
    confirm_password: Optional[str] = None
    role: Optional[Literal["student","clinician","doctor","researcher","data_analyst","admin"]] = "student"

class LoginInput(BaseModel):
    email: EmailStr
    password: str

# ---- Sessions ----
class SessionCreate(BaseModel):
    title: Optional[str] = "Untitled"
    messages: Optional[List[Dict[str, Any]]] = []
    meta: Optional[Dict[str, Any]] = {}

class SessionPatch(BaseModel):
    title: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None

# ==================  ASK  ==================
# Accept BOTH the new shape {session_id, text|q} and the legacy {question}
class AskInput(BaseModel):
    session_id: Optional[str] = None
    q: Optional[str] = None
    text: Optional[str] = None
    question: Optional[str] = None  # legacy frontend support

# -------- Types the frontend expects back from /ask --------
MsgRole = Literal["user", "assistant", "system"]

class Reference(BaseModel):
    pmid: Optional[str] = None
    title: str
    journal: Optional[str] = None
    year: Optional[int] = None
    score: Optional[float] = None
    url: Optional[str] = None
    abstract: Optional[str] = None

class Message(BaseModel):
    id: str
    role: MsgRole
    content: str
    ts: int
    references: Optional[List[Reference]] = None

class RetrievedDoc(BaseModel):
    pmid: Optional[str] = None
    title: str
    journal: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    score: Optional[float] = None
    abstract: Optional[str] = None

class RightPane(BaseModel):
    results: List[RetrievedDoc] = []
    overview: Optional[str] = None
    booleans: List[str] = []
    evidencePack: Optional[str] = None

class SessionOut(BaseModel):
    id: str
    title: str
    createdAt: int
    updatedAt: int
    messages: List[Message] = []
    rightPane: Optional[RightPane] = None
