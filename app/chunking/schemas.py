from pydantic import BaseModel
from typing import Optional


class Chunk(BaseModel):
    """A semantically coherent chunk ready for embedding and storage."""
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: dict = {}
    token_count: int = 0
