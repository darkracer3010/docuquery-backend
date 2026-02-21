import tiktoken
from typing import List
from openai import OpenAI

from app.config import get_settings


class EmbeddingService:
    """Wrapper around OpenAI embeddings API with batching and token management."""

    MAX_TOKENS_PER_TEXT = 8191
    MAX_BATCH_SIZE = 2048

    def __init__(self, client: OpenAI):
        self.client = client
        self.settings = get_settings()
        self.model = self.settings.embedding_model
        self._tokenizer = tiktoken.encoding_for_model(self.model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._tokenizer.encode(text))

    def truncate_text(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within token limit."""
        max_tokens = max_tokens or self.MAX_TOKENS_PER_TEXT
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Handles batching and token truncation automatically.
        """
        if not texts:
            return []

        # Truncate texts that exceed token limit
        safe_texts = [self.truncate_text(t) for t in texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(safe_texts), self.MAX_BATCH_SIZE):
            batch = safe_texts[i : i + self.MAX_BATCH_SIZE]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = self.embed_texts([text])
        return result[0] if result else []
