from pydantic import BaseModel
from typing import Optional


class ParsedElement(BaseModel):
    """A single structural element extracted from a document."""
    text: str
    element_type: str  # "heading", "paragraph", "table_row", "list_item", "code", "key_value"
    page_number: Optional[int] = None
    metadata: dict = {}


class DocumentUploadResponse(BaseModel):
    id: str
    file_name: str
    status: str
    message: str


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    file_size: Optional[int] = None
    status: str
    indexing_progress: int = 0
    total_chunks: int = 0
    created_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
