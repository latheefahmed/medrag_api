
from typing import List, Optional
from pydantic import BaseModel

class RankedDoc(BaseModel):
    pmid: str
    title: str
    journal: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    score: Optional[float] = None
    abstract: Optional[str] = None

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
    evidencePack: Optional[str] = None 

class Reference(BaseModel):
    pmid: Optional[str] = None
    title: str
    journal: Optional[str] = None
    year: Optional[int] = None
    score: Optional[float] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
