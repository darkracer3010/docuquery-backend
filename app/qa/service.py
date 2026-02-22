import json
import logging
from typing import AsyncIterator, List, Optional
from datetime import datetime
from uuid import UUID, uuid4

from openai import OpenAI
from supabase import Client

from app.config import get_settings
from app.embeddings.service import EmbeddingService
from app.qa.schemas import (
    SourceCitation, 
    RetrievalMetadata, 
    AskResponse, 
    ChatMessage, 
    Conversation
)

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
            # No document context found — respond as a general assistant
            general_answer = self._general_chat(question)
            return AskResponse(
                answer=general_answer,
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
        # For synchronous ask, we just use a default or dummy conversation ID for now
        dummy_conv_id = uuid4()
        response.conversation_id = dummy_conv_id
        response.message_id = uuid4()

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
            candidates = result.data or []
            
            if candidates:
                top_sim = candidates[0].get("similarity", 0)
                logger.info(f"Vector search: found {len(candidates)} candidates. Top similarity: {top_sim:.4f}")
            else:
                logger.info("Vector search: no candidates found.")
                
            return candidates
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

            # Handle both {"results": [...]}, {"result": [...]}, and [...] formats
            rankings = parsed if isinstance(parsed, list) else parsed.get("results", parsed.get("result", parsed.get("chunks", [])))
            
            logger.info(f"LLM Re-ranking: scored {len(rankings)} chunks.")

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
        self, question: str, chunks: List[dict], history: List[dict] = []
    ) -> tuple[str, List[SourceCitation]]:
        """Generate an answer using the re-ranked chunks as context."""
        if not chunks:
            return self._general_chat(question), []

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
                *history,
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

    # ── General Chat (No RAG) ─────────────────────────────────────

    def _general_chat(self, question: str) -> str:
        """Handle general conversation without document context."""
        try:
            response = self.openai.chat.completions.create(
                model=self.settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are DocuQuery, a helpful AI assistant for document intelligence. "
                            "You can answer general questions, have normal conversations, and help users "
                            "with their uploaded documents. If the user asks about specific document content "
                            "but no documents were found, let them know they can upload documents and ask "
                            "questions about them. Be friendly and concise."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"General chat failed: {e}")
            return "Sorry, I'm having trouble responding right now. Please try again."

    # ── Streaming Pipeline ────────────────────────────────────────

    _GENERAL_SYSTEM_PROMPT = (
        "You are DocuQuery, a helpful AI assistant for document intelligence. "
        "You can answer general questions, have normal conversations, and help users "
        "with their uploaded documents. If the user asks about specific document content "
        "but no documents were found, let them know they can upload documents and ask "
        "questions about them. Be friendly and concise."
    )

    _RAG_SYSTEM_PROMPT = """You are a helpful document Q&A assistant. Answer the user's question based ONLY on the provided document context.

RULES:
1. Use ONLY information from the provided sources to answer
2. Cite your sources using [1], [2], etc. markers corresponding to the source numbers
3. If the context doesn't contain enough information, say so clearly
4. Be accurate and concise
5. Do not fabricate information not present in the sources"""

    async def ask_stream(
        self,
        question: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        conversation_id: Optional[UUID] = None,
    ) -> AsyncIterator[str]:
        """
          event: done\ndata: {}\n\n
        """
        # Step 0: Ensure conversation exists and log user message
        conv_id = await self._get_or_create_conversation(user_id, conversation_id, question)
        user_msg_id = await self._save_message(conv_id, "user", question)

        # Load conversation history FIRST (before any caching logic)
        history = []
        if conv_id:
            raw_history = await self.get_messages(str(conv_id), user_id)
            # Take last 10 messages (excluding the current user message we just saved)
            for msg in raw_history[-11:-1]:  # -11:-1 to exclude the last message (current question)
                history.append({"role": msg.role, "content": msg.content})
            logger.info(f"Loaded {len(history)} messages from conversation history for context")

        # --- GREETING FAST PATH (Moved before caching/embedding) ---
        if self._is_general_query(question):
            full_answer = ""
            async for event in self._stream_llm(question, self._GENERAL_SYSTEM_PROMPT, history=history):
                yield event
                if event.startswith("event: token\n"):
                    data_line = event.split("data: ", 1)[1].split("\n")[0]
                    full_answer += json.loads(data_line)

            metadata = RetrievalMetadata(
                total_candidates=0, after_reranking=0,
                model_used=self.settings.chat_model, cache_hit=False,
            )
            yield f"event: sources\ndata: []\n\n"
            
            # Save message (Note: We do NOT cache general queries)
            assistant_msg_id = await self._save_message(
                conv_id, "assistant", full_answer, [], metadata.model_dump()
            )
            
            meta_json = metadata.model_dump()
            meta_json["conversation_id"] = str(conv_id)
            meta_json["message_id"] = str(assistant_msg_id)
            
            yield f"event: metadata\ndata: {json.dumps(meta_json)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Step 1: Embed the question
        query_embedding = self.embedding_service.embed_single(question)

        # Step 2: Check semantic cache
        # We check cache for all questions, but we're smart about when to use it:
        # - Use cache for identical/very similar questions (similarity > 0.95)
        # - Skip cache for contextual follow-ups that need conversation history
        cached = self._check_cache(query_embedding, user_id)
        
        # Determine if this is a contextual follow-up question
        is_contextual_followup = self._is_contextual_question(question) and len(history) > 0
        
        if cached and not is_contextual_followup:
            logger.info(f"Cache hit for question (similarity above threshold)")
            self.admin.table("query_cache").update(
                {"hit_count": cached["hit_count"] + 1}
            ).eq("id", cached["id"]).execute()

            response_data = cached["response"]
            # Stream the cached answer token-by-token for consistent UX
            answer = response_data.get("answer", "")
            for word in answer.split(" "):
                yield f"event: token\ndata: {json.dumps(word + ' ')}\n\n"

            sources = response_data.get("sources", [])
            metadata_dict = response_data.get("retrieval_metadata", {})
            metadata_dict["cache_hit"] = True
            
            # Save cached response as a new message in history
            assistant_msg_id = await self._save_message(
                conv_id, "assistant", answer, sources, metadata_dict
            )
            
            metadata_dict["conversation_id"] = str(conv_id)
            metadata_dict["message_id"] = str(assistant_msg_id)

            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            yield f"event: metadata\ndata: {json.dumps(metadata_dict)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        elif cached and is_contextual_followup:
            logger.info(f"Cache found but skipping - this is a contextual follow-up question")
        elif not cached:
            logger.info(f"No cache hit found")

        # Step 3: Retrieve candidates
        candidates = self._vector_search(
            query_embedding, user_id, document_ids, self.settings.retrieval_top_k
        )

        if not candidates:
            # General chat — stream directly
            full_answer = ""
            async for event in self._stream_llm(question, self._GENERAL_SYSTEM_PROMPT, history=history):
                yield event
                if event.startswith("event: token\n"):
                    data_line = event.split("data: ", 1)[1].split("\n")[0]
                    full_answer += json.loads(data_line)

            metadata = RetrievalMetadata(
                total_candidates=0, after_reranking=0,
                model_used=self.settings.chat_model, cache_hit=False,
            )
            yield f"event: sources\ndata: []\n\n"
            
            assistant_msg_id = await self._save_message(
                conv_id, "assistant", full_answer, [], metadata.model_dump()
            )
            
            meta_json = metadata.model_dump()
            meta_json["conversation_id"] = str(conv_id)
            meta_json["message_id"] = str(assistant_msg_id)

            yield f"event: metadata\ndata: {json.dumps(meta_json)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Step 4: Re-rank
        reranked = self._rerank_chunks(question, candidates)

        if not reranked:
            # Re-ranking filtered everything — fall back to general chat
            full_answer = ""
            async for event in self._stream_llm(question, self._GENERAL_SYSTEM_PROMPT, history=history):
                yield event
                if event.startswith("event: token\n"):
                    data_line = event.split("data: ", 1)[1].split("\n")[0]
                    full_answer += json.loads(data_line)

            metadata = RetrievalMetadata(
                total_candidates=len(candidates), after_reranking=0,
                model_used=self.settings.chat_model, cache_hit=False,
            )
            yield f"event: sources\ndata: []\n\n"
            
            assistant_msg_id = await self._save_message(
                conv_id, "assistant", full_answer, [], metadata.model_dump()
            )
            
            meta_json = metadata.model_dump()
            meta_json["conversation_id"] = str(conv_id)
            meta_json["message_id"] = str(assistant_msg_id)

            yield f"event: metadata\ndata: {json.dumps(meta_json)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Step 5: Build context and stream the RAG answer
        context_parts = []
        for i, chunk in enumerate(reranked):
            source_info = f"[Source {i + 1}] Document: {chunk.get('file_name', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')}"
            context_parts.append(f"{source_info}\n{chunk['content']}")
        context = "\n\n---\n\n".join(context_parts)

        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nProvide a comprehensive answer with source citations [1], [2], etc."

        full_answer = ""
        async for event in self._stream_llm(user_prompt, self._RAG_SYSTEM_PROMPT, temperature=0.1, history=history):
            yield event
            # Extract the token text from the event to accumulate the full answer
            if event.startswith("event: token\n"):
                data_line = event.split("data: ", 1)[1].split("\n")[0]
                full_answer += json.loads(data_line)

        # Build source citations
        sources = []
        for i, chunk in enumerate(reranked):
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
                ).model_dump()
            )

        metadata = RetrievalMetadata(
            total_candidates=len(candidates),
            after_reranking=len(reranked),
            model_used=self.settings.chat_model,
            cache_hit=False,
        )

        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        
        # Step 6: Save assistant message and update cache
        assistant_msg_id = await self._save_message(
            conv_id, 
            "assistant", 
            full_answer, 
            sources, 
            metadata.model_dump()
        )
        
        meta_json = metadata.model_dump()
        meta_json["conversation_id"] = str(conv_id)
        meta_json["message_id"] = str(assistant_msg_id)

        yield f"event: metadata\ndata: {json.dumps(meta_json)}\n\n"
        yield "event: done\ndata: {}\n\n"

        # Update semantic cache (ONLY for RAG results with sources)
        try:
            response = AskResponse(
                answer=full_answer,
                sources=[SourceCitation(**s) for s in sources],
                retrieval_metadata=metadata,
                conversation_id=conv_id,
                message_id=assistant_msg_id
            )
            # Only cache if we have sources (RAG response, not general chat)
            if sources and len(sources) > 0:
                self._cache_response(question, query_embedding, user_id, document_ids, response)
                logger.info(f"Cached RAG response with {len(sources)} sources")
        except Exception as e:
            logger.warning(f"Post-stream cache failed: {e}")

    def _is_general_query(self, question: str) -> bool:
        """Heuristic to detect if a query is general conversation vs document-specific."""
        q = question.lower().strip()
        
        # Very short queries are likely general
        if len(q.split()) < 3:
            return True
            
        # Common greetings/general intents
        general_keywords = {
            "hi", "hello", "hey", "how are you", "who are you", 
            "what can you do", "help", "thanks", "thank you",
            "tell me a joke", "joke", "weather", "good morning",
            "good afternoon", "good evening", "how's it going",
            "goodbye", "bye"
        }
        
        # Check if query starts with or is a general greeting
        if any(q.startswith(kw) for kw in general_keywords):
            return True
            
        return False

    def _is_contextual_question(self, question: str) -> bool:
        """Detect if a question requires conversational context (references previous messages)."""
        q = question.lower().strip()
        
        # Contextual reference words/phrases
        contextual_indicators = [
            "it", "this", "that", "these", "those",
            "tell me more", "more about", "elaborate", "explain that",
            "what about", "how about", "and what", "also",
            "the same", "similar", "like that", "as well",
            "you mentioned", "you said", "earlier", "before",
            "continue", "go on", "keep going",
            "what else", "anything else", "more details"
        ]
        
        # Check if question contains contextual indicators
        for indicator in contextual_indicators:
            if indicator in q:
                return True
        
        # Questions starting with "and", "but", "so" are usually contextual
        if q.startswith(("and ", "but ", "so ", "also ")):
            return True
            
        return False

    async def _stream_llm(
        self, user_content: str, system_prompt: str, temperature: float = 0.7, history: List[dict] = []
    ) -> AsyncIterator[str]:
        """Stream OpenAI completion token-by-token as SSE events."""
        try:
            stream = self.openai.chat.completions.create(
                model=self.settings.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history,
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"event: token\ndata: {json.dumps(delta.content)}\n\n"
        except Exception as e:
            logger.error(f"Streaming LLM failed: {e}")
            yield f"event: token\ndata: {json.dumps('Sorry, I encountered an error. Please try again.')}\n\n"

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

            if result.data and isinstance(result.data, list) and len(result.data) > 0:
                entry = result.data[0]
                # Validate the expected keys exist
                if isinstance(entry, dict) and "hit_count" in entry and "response" in entry:
                    return entry
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
        """Store the Q&A result in semantic cache. Only cache RAG responses with sources."""
        try:
            # IMPORTANT: Only cache responses that have sources (RAG results)
            # Don't cache general conversation as it loses context
            if not response.sources or len(response.sources) == 0:
                logger.info("Skipping cache for general conversation (no sources)")
                return
                
            self.admin.table("query_cache").insert(
                {
                    "user_id": user_id,
                    "question": question,
                    "question_embedding": embedding,
                    "document_ids": document_ids,
                    "response": response.model_dump(mode='json'),
                    "hit_count": 0,
                }
            ).execute()
            logger.info(f"Cached response for question: {question[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    # ── Query Logging ──────────────────────────────────────────────

    # ── Conversation History ──────────────────────────────────────

    async def get_conversations(self, user_id: str) -> List[Conversation]:
        """Fetch all conversations for a user, sorted by recency."""
        try:
            result = self.admin.table("conversations").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
            return [Conversation(**c) for c in result.data]
        except Exception as e:
            logger.error(f"Failed to fetch conversations: {e}")
            return []

    async def get_messages(self, conversation_id: str, user_id: str) -> List[ChatMessage]:
        """Fetch all messages for a specific conversation."""
        # Verify ownership via query (RLS should handle this, but let's be explicit if needed)
        try:
            result = self.admin.table("messages").select("*").eq("conversation_id", conversation_id).order("created_at").execute()
            return [ChatMessage(**m) for m in result.data]
        except Exception as e:
            logger.error(f"Failed to fetch messages: {e}")
            return []

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation and its messages."""
        try:
            self.admin.table("conversations").delete().eq("id", conversation_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return False

    async def _get_or_create_conversation(
        self, user_id: str, conversation_id: Optional[UUID], first_message: str
    ) -> UUID:
        """Get existing conversation or create a new one with a snippet of the first message as title."""
        if conversation_id:
            return conversation_id

        # Create new conversation
        title = (first_message[:40] + "...") if len(first_message) > 40 else first_message
        try:
            result = self.admin.table("conversations").insert({
                "user_id": user_id,
                "title": title
            }).execute()
            return UUID(result.data[0]["id"])
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            return uuid4() # Fallback to avoid breaking the stream

    async def _save_message(
        self, 
        conversation_id: UUID, 
        role: str, 
        content: str, 
        sources: List[dict] = None, 
        metadata: dict = None
    ) -> UUID:
        """Save a message to the database and update conversation timestamp."""
        print(f"💾 SAVING MESSAGE: conv={conversation_id}, role={role}")
        logger.info(f"💾 Saving message: conv={conversation_id}, role={role}")
        try:
            result = self.admin.table("messages").insert({
                "conversation_id": str(conversation_id),
                "role": role,
                "content": content,
                "sources": sources or [],
                "metadata": metadata or {}
            }).execute()
            
            # Update conversation timestamp
            self.admin.table("conversations").update({
                "updated_at": datetime.now().isoformat()
            }).eq("id", str(conversation_id)).execute()
            
            print(f"📝 MESSAGE SAVED, checking title...")
            logger.info(f"📝 Message saved, checking if title needs update...")
            # Check if we should update the conversation title
            await self._maybe_update_conversation_title(conversation_id)
            
            return UUID(result.data[0]["id"])
        except Exception as e:
            print(f"❌ ERROR SAVING MESSAGE: {e}")
            logger.error(f"Failed to save message: {e}", exc_info=True)
            return uuid4()

    async def _maybe_update_conversation_title(self, conversation_id: UUID):
        """Update conversation title after 4 messages if it's still the default."""
        print(f"🔍 Checking title for conversation {conversation_id}")
        try:
            # Get conversation
            conv_result = self.admin.table("conversations").select("*").eq("id", str(conversation_id)).single().execute()
            if not conv_result.data:
                print(f"⚠️ Conversation {conversation_id} not found")
                logger.warning(f"Conversation {conversation_id} not found for title update")
                return
            
            conversation = conv_result.data
            current_title = conversation.get("title", "")
            title_generated = conversation.get("title_generated", False)
            
            # Get message count
            messages_result = self.admin.table("messages").select("*").eq("conversation_id", str(conversation_id)).execute()
            messages = messages_result.data or []
            
            print(f"📊 Conversation has {len(messages)} messages, title='{current_title}', title_generated={title_generated}")
            logger.info(f"Conversation {conversation_id}: {len(messages)} messages, title='{current_title}', title_generated={title_generated}")
            
            # If title was already AI-generated, never change it
            if title_generated:
                print(f"✅ Title already AI-generated, never changing it")
                logger.info(f"Title already AI-generated, skipping update")
                return
            
            # Generate title after exactly 4 messages (2 Q&A turns)
            if len(messages) >= 4:
                print(f"🎯 Generating title after {len(messages)} messages...")
                logger.info(f"Generating new title for conversation {conversation_id}")
                # Generate a smart title from the conversation
                new_title = await self._generate_conversation_title(messages[:4])  # Use first 2 turns
                
                if new_title and new_title != current_title:
                    self.admin.table("conversations").update({
                        "title": new_title,
                        "title_generated": True  # Mark that this title was AI-generated
                    }).eq("id", str(conversation_id)).execute()
                    print(f"✅ Updated title to: '{new_title}'")
                    logger.info(f"✅ Updated conversation title to: '{new_title}'")
                else:
                    print(f"⚠️ Failed to generate new title or unchanged")
                    logger.warning(f"Failed to generate new title or title unchanged")
            else:
                print(f"⏳ Not enough messages yet ({len(messages)}/4)")
                logger.info(f"Not enough messages yet ({len(messages)}/4), keeping current title")
        except Exception as e:
            print(f"❌ Error updating title: {e}")
            logger.error(f"Failed to update conversation title: {e}", exc_info=True)

    async def _generate_conversation_title(self, messages: List[dict]) -> str:
        """Generate a concise title summarizing the conversation."""
        try:
            # Build conversation summary from first 2 turns (4 messages)
            conversation_text = ""
            for msg in messages[:4]:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]  # Truncate long messages
                conversation_text += f"{role.upper()}: {content}\n"
            
            prompt = f"""Based on this conversation, generate a short, descriptive title (max 40 characters).
The title should capture the main topic or question being discussed.

CONVERSATION:
{conversation_text}

Return ONLY the title, nothing else. Keep it concise and clear.
Examples: "Python async patterns", "Resume formatting tips", "SQL query optimization"
"""
            
            response = self.openai.chat.completions.create(
                model=self.settings.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise conversation titles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=20
            )
            
            title = response.choices[0].message.content.strip()
            # Remove quotes if present
            title = title.strip('"\'')
            # Truncate to 40 chars if needed
            if len(title) > 40:
                title = title[:37] + "..."
            
            return title
        except Exception as e:
            logger.error(f"Failed to generate conversation title: {e}")
            return ""

    def _log_query(self, question: str, user_id: str):
        """Log the query for analytics."""
        try:
            self.admin.table("query_logs").insert(
                {"user_id": user_id, "question": question}
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
