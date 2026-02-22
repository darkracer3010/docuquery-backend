import numpy as np
from typing import List

from app.documents.schemas import ParsedElement
from app.chunking.schemas import Chunk
from app.embeddings.service import EmbeddingService
from app.config import get_settings


class SemanticChunker:
    """
    Semantic chunking engine that groups content by meaning.

    Algorithm:
    1. Receive parsed elements from document parsers
    2. Embed each element to get its semantic vector
    3. Compute cosine similarity between consecutive elements
    4. Detect topic boundaries where similarity drops below threshold
    5. Group elements into chunks respecting token limits (800-1200)
    6. Add overlap between consecutive chunks (200 tokens)
    7. Attach metadata (page numbers, headings, element types)
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.settings = get_settings()
        self.min_tokens = self.settings.chunk_min_tokens
        self.max_tokens = self.settings.chunk_max_tokens
        self.overlap_tokens = self.settings.chunk_overlap_tokens
        self.similarity_threshold = self.settings.semantic_similarity_threshold

    def chunk(self, elements: List[ParsedElement]) -> List[Chunk]:
        """Perform semantic chunking on parsed document elements."""
        if not elements:
            return []

        # Filter out very short elements (relaxed from 10 to 3 to preserve headers)
        valid_elements = [e for e in elements if len(e.text.strip()) > 3]
        if not valid_elements:
            return []

        # Step 1: Get embeddings for all elements
        texts = [e.text for e in valid_elements]
        embeddings = self.embedding_service.embed_texts(texts)

        # Step 2: Compute similarities between consecutive elements
        similarities = self._compute_consecutive_similarities(embeddings)

        # Step 3: Detect boundary points (topic shifts)
        boundaries = self._detect_boundaries(similarities)

        # Step 4: Group elements into semantic groups
        groups = self._group_elements(valid_elements, boundaries)

        # Step 5: Merge/split groups to respect token limits
        sized_groups = self._enforce_token_limits(groups)

        # Step 6: Create chunks with overlap
        chunks = self._create_chunks_with_overlap(sized_groups)

        return chunks

    def _compute_consecutive_similarities(
        self, embeddings: List[List[float]]
    ) -> List[float]:
        """Compute cosine similarity between each pair of consecutive embeddings."""
        if len(embeddings) < 2:
            return []

        similarities = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i])
            b = np.array(embeddings[i + 1])
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                similarities.append(0.0)
            else:
                similarities.append(float(np.dot(a, b) / (norm_a * norm_b)))

        return similarities

    def _detect_boundaries(self, similarities: List[float]) -> List[int]:
        """
        Detect topic shift boundaries where similarity drops below threshold.
        Returns indices where breaks should occur.
        """
        if not similarities:
            return []

        boundaries = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)  # Break AFTER element at index i

        return boundaries

    def _group_elements(
        self, elements: List[ParsedElement], boundaries: List[int]
    ) -> List[List[ParsedElement]]:
        """Group elements by semantic boundaries."""
        groups = []
        prev_boundary = 0

        for boundary in boundaries:
            group = elements[prev_boundary:boundary]
            if group:
                groups.append(group)
            prev_boundary = boundary

        # Add remaining elements
        remaining = elements[prev_boundary:]
        if remaining:
            groups.append(remaining)

        return groups

    def _enforce_token_limits(
        self, groups: List[List[ParsedElement]]
    ) -> List[List[ParsedElement]]:
        """Split or merge groups to respect min/max token limits."""
        final_groups = []

        accumulator = []
        acc_tokens = 0

        for group in groups:
            group_text = "\n\n".join(e.text for e in group)
            group_tokens = self.embedding_service.count_tokens(group_text)

            if group_tokens > self.max_tokens:
                # Flush accumulator first
                if accumulator:
                    final_groups.append(accumulator)
                    accumulator = []
                    acc_tokens = 0

                # Split large group element by element
                sub_group = []
                sub_tokens = 0
                for elem in group:
                    elem_tokens = self.embedding_service.count_tokens(elem.text)
                    if sub_tokens + elem_tokens > self.max_tokens and sub_group:
                        final_groups.append(sub_group)
                        sub_group = []
                        sub_tokens = 0
                    sub_group.append(elem)
                    sub_tokens += elem_tokens
                if sub_group:
                    final_groups.append(sub_group)

            elif acc_tokens + group_tokens < self.min_tokens:
                # Too small — accumulate
                accumulator.extend(group)
                acc_tokens += group_tokens

            else:
                # Flush accumulator if merging would exceed max
                if acc_tokens + group_tokens > self.max_tokens:
                    if accumulator:
                        final_groups.append(accumulator)
                    accumulator = list(group)
                    acc_tokens = group_tokens
                else:
                    accumulator.extend(group)
                    acc_tokens += group_tokens
                    final_groups.append(accumulator)
                    accumulator = []
                    acc_tokens = 0

        # Flush any remaining accumulator
        if accumulator:
            # If too small and we have previous groups, merge with last
            if acc_tokens < self.min_tokens and final_groups:
                final_groups[-1].extend(accumulator)
            else:
                final_groups.append(accumulator)

        return final_groups

    def _create_chunks_with_overlap(
        self, groups: List[List[ParsedElement]]
    ) -> List[Chunk]:
        """Create final chunks with overlap from previous chunk."""
        chunks = []
        prev_tail_text = ""

        for idx, group in enumerate(groups):
            # Build chunk text
            group_text = "\n\n".join(e.text for e in group)

            # Add overlap from previous chunk
            if prev_tail_text and idx > 0:
                content = prev_tail_text + "\n\n" + group_text
            else:
                content = group_text

            # Determine page number (use first element's page)
            page_numbers = [e.page_number for e in group if e.page_number is not None]
            page_number = page_numbers[0] if page_numbers else None

            # Collect metadata
            element_types = list(set(e.element_type for e in group))
            headings = [e.text for e in group if e.element_type == "heading"]
            sources = list(set(e.metadata.get("source", "") for e in group))

            token_count = self.embedding_service.count_tokens(content)

            chunks.append(
                Chunk(
                    content=content,
                    chunk_index=idx,
                    page_number=page_number,
                    token_count=token_count,
                    metadata={
                        "element_types": element_types,
                        "headings": headings,
                        "page_numbers": list(set(page_numbers)) if page_numbers else [],
                        "sources": sources,
                    },
                )
            )

            # Prepare overlap for next chunk — take last ~overlap_tokens worth of text
            self._prepare_overlap_text(group_text)
            prev_tail_text = self._get_overlap_text(group_text)

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Extract the last N tokens worth of text for overlap."""
        tokens = self.embedding_service._tokenizer.encode(text)
        if len(tokens) <= self.overlap_tokens:
            return text
        overlap_tokens = tokens[-self.overlap_tokens :]
        return self.embedding_service._tokenizer.decode(overlap_tokens)

    def _prepare_overlap_text(self, text: str):
        """No-op placeholder for potential future overlap preparation logic."""
        pass
