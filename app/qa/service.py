import json
import logging
from typing import List, Optional

from openai import OpenAI
from supabase import Client

from app.config import get_settings
from app.embeddings.service import EmbeddingService
from app.qa.schemas import SourceCitation, RetrievalMetadata, AskResponse

logger = logging.getLogger(__name__)


class QAService:
    """RAG Q&A pipeline with semantic caching, LLM re-ranking, and source citations."""

    def __init__(
        self,
        supabase_admin: Client,
        openai_client: OpenAI,
        embedding_service: EmbeddingService,
    ):
        self.admin = supabase_admin
        self.openai = openai_client
        self.embedding_service = embedding_service
        self.settings = get_settings()

    async def ask(
        self,
        question: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> AskResponse:
        """
        Full RAG pipeline:
        1. Embed query
        2. Check semantic cache
        3. Retrieve candidates (vector search)
        4. LLM re-ranking with justifications
        5. Generate answer with citations
        6. Cache & log
        """
        # Step 1: Embed the question
        query_embedding = self.embedding_service.embed_single(question)

        # Step 2: Check semantic cache
        cached = self._check_cache(query_embedding, user_id)
        if cached:
            logger.info(f"Cache hit for question: {question[:50]}...")
            # Update hit count
            self.admin.table("query_cache").update(
                {"hit_count": cached["hit_count"] + 1}
            ).eq("id", cached["id"]).execute()

            response_data = cached["response"]
            response_data["retrieval_metadata"]["cache_hit"] = True
            return AskResponse(**response_data)

        # Step 3: Retrieve candidates via vector search
        candidates = self._vector_search(
            query_embedding, user_id, document_ids, self.settings.retrieval_top_k
        )

        if not candidates:
            return AskResponse(
                answer="I couldn't find any relevant information in your documents to answer this question. Please make sure you've uploaded relevant documents and they have been processed successfully.",
                sources=[],
                retrieval_metadata=RetrievalMetadata(
                    total_candidates=0,
                    after_reranking=0,
                    model_used=self.settings.chat_model,
                    cache_hit=False,
                ),
            )

        # Step 4: LLM Re-ranking
        reranked = self._rerank_chunks(question, candidates)

        # Step 5: Generate answer with citations
        answer, cited_sources = self._generate_answer(question, reranked)

        # Step 6: Build response
        response = AskResponse(
            answer=answer,
            sources=cited_sources,
            retrieval_metadata=RetrievalMetadata(
                total_candidates=len(candidates),
                after_reranking=len(reranked),
                model_used=self.settings.chat_model,
                cache_hit=False,
            ),
        )

        # Step 7: Cache & log
        self._cache_response(question, query_embedding, user_id, document_ids, response)
        self._log_query(question, user_id)

        return response

    # ── Vector Search ──────────────────────────────────────────────

    def _vector_search(
        self,
        query_embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]],
        top_k: int,
    ) -> List[dict]:
        """Search for similar chunks using the match_chunks RPC function."""
        try:
            params = {
                "query_embedding": query_embedding,
                "match_user_id": user_id,
                "match_count": top_k,
            }
            if document_ids:
                params["filter_document_ids"] = document_ids

            result = self.admin.rpc("match_chunks", params).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    # ── LLM Re-Ranking ────────────────────────────────────────────

    def _rerank_chunks(self, question: str, candidates: List[dict]) -> List[dict]:
        """Re-rank retrieved chunks using LLM for relevance scoring."""
        if not candidates:
            return []

        chunks_text = ""
        for i, chunk in enumerate(candidates):
            snippet = chunk["content"][:500]
            chunks_text += f"\n--- Chunk {i} (from: {chunk.get('file_name', 'unknown')}, page: {chunk.get('page_number', 'N/A')}) ---\n{snippet}\n"

        rerank_prompt = f"""You are a relevance scoring assistant. Given a user question and retrieved document chunks, score each chunk's relevance.

QUESTION: "{question}"

RETRIEVED CHUNKS:
{chunks_text}

For each chunk, provide:
1. chunk_index: the chunk number (0-indexed)
2. relevance_score: a float between 0.0 and 1.0 (1.0 = perfectly relevant)
3. justification: a brief explanation of why this score was given

Return ONLY valid JSON array, no other text:
[{{"chunk_index": 0, "relevance_score": 0.95, "justification": "Directly answers the question about..."}}]
"""

        try:
            response = self.openai.chat.completions.create(
                model=self.settings.chat_model,
                messages=[
                    {"role": "system", "content": "You are a precise relevance scoring assistant. Return only valid JSON."},
                    {"role": "user", "content": rerank_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            # Handle both {"results": [...]} and [...] formats
            rankings = parsed if isinstance(parsed, list) else parsed.get("results", parsed.get("chunks", []))

            # Enrich candidates with relevance scores
            score_map = {}
            for r in rankings:
                idx = r.get("chunk_index", -1)
                if 0 <= idx < len(candidates):
                    score_map[idx] = {
                        "relevance_score": float(r.get("relevance_score", 0)),
                        "justification": r.get("justification", "No justification provided."),
                    }

            for i, chunk in enumerate(candidates):
                if i in score_map:
                    chunk["relevance_score"] = score_map[i]["relevance_score"]
                    chunk["relevance_justification"] = score_map[i]["justification"]
                else:
                    chunk["relevance_score"] = 0.0
                    chunk["relevance_justification"] = "Not scored by LLM."

            # Sort by relevance and filter
            candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            filtered = [
                c
                for c in candidates
                if c.get("relevance_score", 0) >= self.settings.rerank_relevance_threshold
            ]

            return filtered[: self.settings.rerank_top_k]

        except Exception as e:
            logger.error(f"Re-ranking failed: {e}. Using original order.")
            # Fallback: use vector similarity scores as relevance
            for chunk in candidates:
                chunk["relevance_score"] = chunk.get("similarity", 0)
                chunk["relevance_justification"] = "Fallback: using vector similarity score."
            return candidates[: self.settings.rerank_top_k]

    # ── Answer Generation ──────────────────────────────────────────

    def _generate_answer(
        self, question: str, chunks: List[dict]
    ) -> tuple[str, List[SourceCitation]]:
        """Generate an answer using the re-ranked chunks as context."""
        if not chunks:
            return "No relevant information found.", []

        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_info = f"[Source {i + 1}] Document: {chunk.get('file_name', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')}"
            context_parts.append(f"{source_info}\n{chunk['content']}")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are a helpful document Q&A assistant. Answer the user's question based ONLY on the provided document context.

RULES:
1. Use ONLY information from the provided sources to answer
2. Cite your sources using [1], [2], etc. markers corresponding to the source numbers
3. If the context doesn't contain enough information, say so clearly
4. Be accurate and concise
5. Do not fabricate information not present in the sources"""

        user_prompt = f"""CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive answer with source citations [1], [2], etc."""

        response = self.openai.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content

        # Build source citations
        sources = []
        for i, chunk in enumerate(chunks):
            sources.append(
                SourceCitation(
                    citation_index=i + 1,
                    document_name=chunk.get("file_name", "Unknown"),
                    document_id=str(chunk.get("document_id", "")),
                    page_number=chunk.get("page_number"),
                    chunk_snippet=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    vector_similarity_score=round(float(chunk.get("similarity", 0)), 4),
                    relevance_score=round(float(chunk.get("relevance_score", 0)), 4),
                    relevance_justification=chunk.get("relevance_justification", ""),
                )
            )

        return answer, sources

    # ── Semantic Cache ─────────────────────────────────────────────

    def _check_cache(self, query_embedding: List[float], user_id: str) -> Optional[dict]:
        """Check if a semantically similar question was asked before."""
        try:
            result = self.admin.rpc(
                "check_query_cache",
                {
                    "query_embedding": query_embedding,
                    "match_user_id": user_id,
                    "similarity_threshold": self.settings.cache_similarity_threshold,
                },
            ).execute()

            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
            return None

    def _cache_response(
        self,
        question: str,
        embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]],
        response: AskResponse,
    ):
        """Store the Q&A result in semantic cache."""
        try:
            self.admin.table("query_cache").insert(
                {
                    "user_id": user_id,
                    "question": question,
                    "question_embedding": embedding,
                    "document_ids": document_ids,
                    "response": response.model_dump(),
                    "hit_count": 0,
                }
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    # ── Query Logging ──────────────────────────────────────────────

    def _log_query(self, question: str, user_id: str):
        """Log the query for analytics."""
        try:
            self.admin.table("query_logs").insert(
                {"user_id": user_id, "question": question}
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
