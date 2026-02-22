from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_jwt_secret: str

    # OpenAI
    openai_api_key: str

    # App
    cors_origins: str = "http://localhost:3000"

    # Chunking
    chunk_min_tokens: int = 800
    chunk_max_tokens: int = 1200
    chunk_overlap_tokens: int = 200
    semantic_similarity_threshold: float = 0.5
    use_llm_chunking: bool = False  # Set to True for more accurate LLM-based chunking

    # RAG
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    rerank_relevance_threshold: float = 0.3
    cache_similarity_threshold: float = 0.95

    # Models
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
