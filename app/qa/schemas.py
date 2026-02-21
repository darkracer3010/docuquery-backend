from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None


class SourceCitation(BaseModel):
    citation_index: int
    document_name: str
    document_id: str
    page_number: Optional[int] = None
    chunk_snippet: str
    vector_similarity_score: float
    relevance_score: float
    relevance_justification: str


class RetrievalMetadata(BaseModel):
    total_candidates: int
    after_reranking: int
    model_used: str
    cache_hit: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    retrieval_metadata: RetrievalMetadata
