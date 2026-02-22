from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[UUID] = None


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
    conversation_id: UUID
    message_id: UUID


class ChatMessage(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    sources: Optional[List[SourceCitation]] = None
    metadata: Optional[dict] = None
    created_at: datetime


class Conversation(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

