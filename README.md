# DocuQuery Backend

RAG-based Smart Document Q&A API built with FastAPI, Supabase, and OpenAI.

## Features

- **User Authentication** — Supabase Auth with JWT verification (ES256/HS256)
- **Document Management** — Upload, list, view, delete (PDF, DOCX, Excel, CSV, JSON, Markdown, Text)
- **Smart Chunking** — Two strategies available:
  - Semantic chunking (default) — Embedding-based topic boundary detection (800-1200 tokens, 200 overlap)
  - LLM chunking — GPT-powered semantic boundary detection (more accurate, higher cost)
- **RAG Q&A** — Vector retrieval → LLM re-ranking → answer generation with source citations
- **Intelligent Caching** — Context-aware semantic caching:
  - Caches identical questions for fast responses
  - Skips cache for contextual follow-ups to preserve conversation flow
  - Only caches RAG responses with sources
- **Streaming Responses** — Real-time SSE streaming for chat-like experience
- **Conversation History** — Persistent chat sessions with message tracking
- **Background Indexing** — Async document processing with real-time progress tracking

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI (Python 3.11+) |
| Database | Supabase PostgreSQL + pgvector |
| Auth | Supabase Auth (JWT with JWKS) |
| Storage | Supabase Storage |
| AI | OpenAI (text-embedding-3-small + gpt-4o-mini) |
| Deployment | Railway (Nixpacks) |

## Setup

### 1. Clone & Install

```bash
git clone <repo-url>
cd docuquery-backend
uv sync
```

Or with pip:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env.example .env
# Fill in your Supabase and OpenAI credentials
```

Required variables:
- `SUPABASE_URL` — Your Supabase project URL
- `SUPABASE_ANON_KEY` — Supabase anon/public key
- `SUPABASE_SERVICE_ROLE_KEY` — Supabase service role key (for admin operations)
- `SUPABASE_JWT_SECRET` — JWT secret for token verification
- `OPENAI_API_KEY` — OpenAI API key
- `CORS_ORIGINS` — Comma-separated list of allowed origins (e.g., `http://localhost:3000`)

### 3. Supabase Setup

Run the following in Supabase SQL Editor:
1. Main schema (tables, RLS policies) — see `schema.sql`
2. `match_chunks` RPC function — for vector similarity search
3. `query_cache` table + `check_query_cache` RPC — for semantic caching
4. `conversations` and `messages` tables — for chat history
5. Create a Storage bucket named `documents` (public: off)

### 4. Run Locally

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Or with activated venv:
```bash
uvicorn app.main:app --reload --port 8000
```

API docs at: http://localhost:8000/docs

## API Endpoints

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/signup` | Register new user |
| POST | `/auth/login` | Sign in |
| GET | `/auth/me` | Get current user profile |

### Documents
| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents/upload` | Upload single document |
| POST | `/documents/upload-bulk` | Upload multiple documents |
| GET | `/documents/` | List all documents |
| GET | `/documents/{id}` | Get document details |
| GET | `/documents/{id}/progress` | Get processing progress |
| DELETE | `/documents/{id}` | Delete document + chunks |

### Q&A
| Method | Endpoint | Description |
|---|---|---|
| POST | `/qa/ask` | Ask a question with RAG (non-streaming) |
| POST | `/qa/ask/stream` | Ask a question with streaming response (SSE) |
| GET | `/qa/conversations` | List all conversations |
| GET | `/qa/conversations/{id}/messages` | Get conversation message history |
| DELETE | `/qa/conversations/{id}` | Delete a conversation |

## Architecture

### Document Processing Pipeline
```
Upload → Supabase Storage → Background Task:
  1. Parse (7 formats: PDF, DOCX, XLSX, CSV, JSON, MD, TXT)
  2. Semantic/LLM Chunking (800-1200 tokens, 200 overlap)
  3. Embed chunks (text-embedding-3-small)
  4. Store in pgvector with metadata
```

### RAG Query Pipeline
```
Question → Embed → Smart Cache Check → 
  Vector Search (top-10) → LLM Re-Rank (top-5, scored + justified) → 
  Generate Answer with History → Source Citations → Cache Result
```

### Intelligent Caching Strategy
- **Cache Hit**: Identical questions (>0.95 similarity) return cached responses instantly
- **Cache Skip**: Contextual follow-ups ("tell me more", "what about it") use full RAG with conversation history
- **Cache Storage**: Only RAG responses with sources are cached (general chat is not cached)

### Conversation Management
- Each chat session creates a conversation with auto-generated title
- Messages are stored with sources and metadata
- Conversation list sorted by most recent activity
- Full message history available for context

## Configuration

Key settings in `app/config.py` (can be overridden via environment variables):

### Chunking Strategy
```python
use_llm_chunking: bool = False  # Enable LLM-based chunking for better accuracy
chunk_min_tokens: int = 800
chunk_max_tokens: int = 1200
chunk_overlap_tokens: int = 200
semantic_similarity_threshold: float = 0.5
```

**Semantic Chunking (default):**
- Fast and cost-effective
- Uses embedding similarity to detect topic boundaries
- Best for most documents
- Cost: ~$0.0002 per document

**LLM Chunking (optional):**
- More accurate semantic boundaries
- Uses GPT-4o-mini to identify topic shifts
- Better for complex documents with subtle topic changes
- Cost: ~$0.0002 per document (similar to semantic!)
- Enable by setting `USE_LLM_CHUNKING=true` in `.env`

### RAG Settings
```python
retrieval_top_k: int = 10
rerank_top_k: int = 5
rerank_relevance_threshold: float = 0.3
cache_similarity_threshold: float = 0.95
```

### Models
```python
embedding_model: str = "text-embedding-3-small"
chat_model: str = "gpt-4o-mini"
```

## Deployment (Railway)

1. Push to GitHub
2. Connect repo to Railway
3. Add environment variables in Railway dashboard
4. Deploy — Nixpacks auto-detects Python & installs dependencies

## Development

### Running Tests
```bash
pytest
```

### Code Quality
```bash
# Format code
black .

# Lint
ruff check .
```

## Troubleshooting

### JWT Verification Issues
The system supports both ES256 (via JWKS) and HS256 (via secret) for JWT verification. If you encounter auth issues:
1. Ensure `SUPABASE_JWT_SECRET` is set correctly
2. Check that your Supabase project uses the expected signing algorithm

### Cache Not Working
- Verify `query_cache` table exists in Supabase
- Check `cache_similarity_threshold` setting (default: 0.95)
- Ensure pgvector extension is enabled

### Document Processing Stuck
- Check backend logs for errors
- Verify OpenAI API key is valid and has credits
- Ensure Supabase Storage bucket exists and is accessible
