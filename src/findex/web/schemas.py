"""Pydantic models at the HTTP boundary; core index models remain dataclasses."""

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=200)
    k: int = Field(default=10, ge=1, le=100)
    scorer: str = Field(default="bm25", pattern="^(bm25|tfidf)$")
    mode: str = Field(default="bm25", pattern="^(bm25|tfidf|semantic|hybrid)$")
    page: int = Field(default=1, ge=1)


class SearchResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    doc_id: int
    title: str
    score: float
    snippet: str
    match_type: str = "term"


class SearchResponse(BaseModel):
    query: str
    mode: str = "bm25"
    total: int
    page: int
    pages: int
    k: int
    took_ms: float
    results: list[SearchResultResponse]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str


class StatsResponse(BaseModel):
    status: str
    index_size_bytes: int
    documents: int
    terms: int
    average_document_length: float
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
