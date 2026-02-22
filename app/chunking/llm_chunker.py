import logging
from typing import List
import json

from openai import OpenAI
from app.documents.schemas import ParsedElement
from app.chunking.schemas import Chunk
from app.embeddings.service import EmbeddingService
from app.config import get_settings

logger = logging.getLogger(__name__)

class LLMChunker:
    """
    Advanced chunker that uses an LLM to identify semantic boundaries.
    
    This approach is more accurate than simple vector similarity but more expensive.
    It works by sending a sequence of element summaries/snippets to the LLM and 
    asking it to identify where one topic ends and another begins.
    """

    def __init__(self, openai_client: OpenAI, embedding_service: EmbeddingService):
        self.openai = openai_client
        self.embedding_service = embedding_service
        self.settings = get_settings()
        self.max_tokens = self.settings.chunk_max_tokens
        self.overlap_tokens = self.settings.chunk_overlap_tokens

    def chunk(self, elements: List[ParsedElement], on_progress: callable = None) -> List[Chunk]:
        """Perform LLM-informed semantic chunking."""
        if not elements:
            return []

        # We first group elements into small 'mini-chunks' to reduce LLM calls
        # then let the LLM decide where to merge them or split them.
        # For simplicity in this version, we'll ask the LLM to identify 
        # indices in the element list where a major topic shift occurs.
        
        # We'll batch elements to stay within context windows
        BATCH_SIZE = 20 
        boundaries = []
        
        for i in range(0, len(elements), BATCH_SIZE):
            batch = elements[i:i + BATCH_SIZE]
            if len(batch) < 2:
                continue
            
            batch_boundaries = self._get_llm_boundaries(batch, i)
            boundaries.extend(batch_boundaries)
            
            if on_progress:
                progress = int(((i + len(batch)) / len(elements)) * 100)
                on_progress(progress)

        # Merge elements into chunks based on boundaries and token limits
        return self._create_chunks(elements, boundaries)

    def _get_llm_boundaries(self, elements: List[ParsedElement], offset: int) -> List[int]:
        """Ask LLM to identify indices of topic shifts in the batch."""
        element_list = []
        for j, e in enumerate(elements):
            snippet = e.text[:150].replace("\n", " ")
            element_list.append(f"[{j + offset}] (Type: {e.element_type}): {snippet}...")

        prompt = f"""Analyze the following sequence of document elements. 
Identify the indices where a major topic shift occurs (a new section, a change in subject, or a new context).

ELEMENTS:
{chr(10).join(element_list)}

Return a JSON object with a key 'boundaries' containing a list of indices where a NEW topic starts. If everything is one topic, return {"boundaries": []}.
Example: {"boundaries": [5, 12]}"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at document structure analysis. Always return JSON with a 'boundaries' key."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            
            # Robust extraction of the list
            boundaries = data.get("boundaries", [])
            if isinstance(boundaries, list):
                return [int(x) for x in boundaries if str(x).isdigit() or isinstance(x, (int, float))]
            
            return []
        except Exception as e:
            logger.warning(f"LLM boundary detection failed: {e}")
            return []

    def _create_chunks(self, elements: List[ParsedElement], boundaries: List[int]) -> List[Chunk]:
        """Group elements based on boundaries while respecting token limits."""
        chunks = []
        boundaries_set = set(boundaries)
        
        current_group = []
        current_tokens = 0
        chunk_idx = 0

        for i, element in enumerate(elements):
            tokens = self.embedding_service.count_tokens(element.text)
            
            # Start new chunk if we hit a boundary or exceed max tokens
            if (i in boundaries_set or current_tokens + tokens > self.max_tokens) and current_group:
                chunks.append(self._build_chunk(current_group, chunk_idx))
                chunk_idx += 1
                # Handle overlap: keep some elements from previous chunk
                current_group = self._get_overlap_elements(current_group)
                current_tokens = sum(self.embedding_service.count_tokens(e.text) for e in current_group)

            current_group.append(element)
            current_tokens += tokens

        if current_group:
            chunks.append(self._build_chunk(current_group, chunk_idx))

        return chunks

    def _get_overlap_elements(self, elements: List[ParsedElement]) -> List[ParsedElement]:
        """Keep last few elements that fit within overlap token limit."""
        overlap = []
        tokens = 0
        for e in reversed(elements):
            e_tokens = self.embedding_service.count_tokens(e.text)
            if tokens + e_tokens > self.overlap_tokens:
                break
            overlap.insert(0, e)
            tokens += e_tokens
        return overlap

    def _build_chunk(self, group: List[ParsedElement], idx: int) -> Chunk:
        content = "\n\n".join(e.text for e in group)
        page_numbers = list(set(e.page_number for e in group if e.page_number is not None))
        return Chunk(
            content=content,
            chunk_index=idx,
            page_number=page_numbers[0] if page_numbers else None,
            token_count=self.embedding_service.count_tokens(content),
            metadata={
                "page_numbers": page_numbers,
                "element_types": list(set(e.element_type for e in group)),
                "llm_semantic_boost": True
            }
        )
