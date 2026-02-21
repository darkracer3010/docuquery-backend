# DocuQuery Backend

RAG-based Smart Document Q&A API built with FastAPI, Supabase, and OpenAI.

## Features

- **User Authentication** — Supabase Auth with JWT verification
- **Document Management** — Upload, list, view, delete (PDF, DOCX, Excel, CSV, JSON, Markdown, Text)
- **Semantic Chunking** — Embedding-based topic boundary detection (800-1200 tokens, 200 overlap)
- **RAG Q&A** — Vector retrieval → LLM re-ranking → answer generation with source citations
- **Semantic Caching** — Postgres-based query cache using pgvector similarity
- **Background Indexing** — Async document processing with progress tracking

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI (Python 3.11) |
| Database | Supabase PostgreSQL + pgvector |
| Auth | Supabase Auth (JWT) |
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

### 2. Environment Variables

```bash
cp .env.example .env
# Fill in your Supabase and OpenAI credentials
```

### 3. Supabase Setup

Run the following in Supabase SQL Editor:
1. Main schema (tables, RLS policies) — see `schema.sql`
2. `match_chunks` RPC function — for vector similarity search
3. `query_cache` table + `check_query_cache` RPC — for semantic caching
4. Create a Storage bucket named `documents` (public: off)

### 4. Run Locally

```bash
uv run uvicorn app.main:app --reload --port 8000
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
| GET | `/documents/{id}` | Get document status |
| DELETE | `/documents/{id}` | Delete document + chunks |

### Q&A
| Method | Endpoint | Description |
|---|---|---|
| POST | `/qa/ask` | Ask a question with RAG |

## Architecture

```
Upload → Supabase Storage → Background Task:
  Parse (7 formats) → Semantic Chunk → Embed → Store in pgvector

Question → Embed → Cache Check → Vector Search (top-10) →
  LLM Re-Rank (scored + justified) → Generate Answer → Citations
```

## Deployment (Railway)

1. Push to GitHub
2. Connect repo to Railway
3. Add env vars in Railway dashboard
4. Deploy — Nixpacks auto-detects Python & installs deps
