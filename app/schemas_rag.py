# app/schemas_rag.py
from typing import List, Optional
from pydantic import BaseModel

# Mirrors your frontend types (RightPaneData, RankedDoc, BooleanItem, etc.)

class RankedDoc(BaseModel):
    pmid: str
    title: str
    journal: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    score: Optional[float] = None

class BooleanItem(BaseModel):
    group: str
    query: str
    note: Optional[str] = None

class EvidenceItem(BaseModel):
    pmid: str
    year: Optional[int] = None
    journal: Optional[str] = None
    title: str
    snippet: Optional[str] = None

class Overview(BaseModel):
    conclusion: str = ""
    key_findings: List[str] = []
    quality_and_limits: List[str] = []

class PlanLite(BaseModel):
    chunks: List[str] = []
    time_tags: List[str] = []
    exclusions: List[str] = []

class RightPaneData(BaseModel):
    results: List[RankedDoc] = []
    booleans: List[BooleanItem] = []
    evidence: List[EvidenceItem] = []
    overview: Optional[Overview] = None
    plan: Optional[PlanLite] = None
    # keep camelCase to match FE prop name exactly
    evidencePack: Optional[str] = None

class AskOutput(BaseModel):
    message: str               # the assistant chat text we’ll append to the session
    rightPane: RightPaneData   # everything the right-side panel needs
