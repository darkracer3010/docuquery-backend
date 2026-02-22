import logging
import json
from typing import List, Optional

from supabase import Client

from app.config import get_settings
from app.documents.parsers import parse_document, SUPPORTED_EXTENSIONS
from app.documents.schemas import ParsedElement
from app.chunking.semantic import SemanticChunker
from app.chunking.llm_chunker import LLMChunker
from app.chunking.schemas import Chunk
from app.embeddings.service import EmbeddingService
from openai import OpenAI

logger = logging.getLogger(__name__)


class DocumentService:
    """Handles document upload, processing, and management."""

    STORAGE_BUCKET = "documents"

    def __init__(
        self,
        supabase_client: Client,
        admin_client: Client,
        embedding_service: EmbeddingService,
        openai_client: OpenAI,
    ):
        self.client = supabase_client
        self.admin = admin_client
        self.embedding_service = embedding_service
        self.openai = openai_client
        self.settings = get_settings()

    # ── Upload ─────────────────────────────────────────────────────

    def upload_document(
        self, user_id: str, file_bytes: bytes, filename: str, file_size: int
    ) -> dict:
        """Upload a file to Supabase Storage and create a document record."""
        # Validate file type
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Upload to Supabase Storage
        storage_path = f"{user_id}/{filename}"
        self.admin.storage.from_(self.STORAGE_BUCKET).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": "application/octet-stream", "upsert": "true"},
        )

        # Create document record
        doc_data = {
            "user_id": user_id,
            "file_name": filename,
            "file_path": storage_path,
            "file_size": file_size,
            "status": "uploaded",
            "indexing_progress": 0,
            "total_chunks": 0,
        }
        result = (
            self.admin.table("documents").insert(doc_data).execute()
        )

        return result.data[0]

    # ── Background Processing ──────────────────────────────────────

    def process_document(self, document_id: str, user_id: str):
        """
        Background task: parse, chunk, embed, and index a document.
        Updates status and progress throughout.
        """
        try:
            # Update status to processing
            if not self._document_exists(document_id):
                logger.info(f"Document {document_id} no longer exists. Aborting.")
                return
            self._update_status(document_id, "processing", 0)

            # 1. Download file from storage
            doc = self._get_document(document_id)
            file_bytes = self.admin.storage.from_(self.STORAGE_BUCKET).download(
                doc["file_path"]
            )
            self._update_status(document_id, "processing", 10)

            # 2. Parse document
            if not self._document_exists(document_id):
                return
            logger.info(f"Parsing document: {doc['file_name']}")
            elements = parse_document(file_bytes, doc["file_name"])
            if not elements:
                self._update_status(document_id, "ready", 100, total_chunks=0)
                return
            self._update_status(document_id, "processing", 30)

            # 3. Choose chunker based on settings
            if not self._document_exists(document_id):
                return
            
            if self.settings.use_llm_chunking:
                logger.info("Using LLM-based chunking for better accuracy")
                chunker = LLMChunker(self.openai, self.embedding_service)
                def chunk_progress(p):
                    self._update_status(document_id, "processing", 30 + int(p * 0.2))
                chunks = chunker.chunk(elements, on_progress=chunk_progress)
            else:
                logger.info("Using semantic chunking (default)")
                chunker = SemanticChunker(self.embedding_service)
                chunks = chunker.chunk(elements)

            if not chunks:
                self._update_status(document_id, "ready", 100, total_chunks=0)
                return
            self._update_status(document_id, "processing", 50)

            # 4. Embed chunks
            if not self._document_exists(document_id):
                return
            logger.info(f"Embedding {len(chunks)} chunks")
            texts = [c.content for c in chunks]
            
            def embed_progress(p):
                self._update_status(document_id, "processing", 50 + int(p * 0.3))
            
            embeddings = self.embedding_service.embed_texts(texts, on_progress=embed_progress)
            self._update_status(document_id, "processing", 80)

            # 5. Store in vector DB
            if not self._document_exists(document_id):
                return
            logger.info(f"Indexing {len(chunks)} chunks")
            BATCH_SIZE = 50
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i : i + BATCH_SIZE]
                batch_embeddings = embeddings[i : i + BATCH_SIZE]

                # ... (rest of the logic remains same, but I'll add progress update)
                progress = 80 + int((i / len(chunks)) * 15)
                self._update_status(document_id, "processing", progress)
                
                rows = []
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    if i % 100 == 0 and not self._document_exists(document_id):
                        logger.info(f"Document {document_id} deleted during indexing. stopping.")
                        return
                    rows.append(
                        {
                            "document_id": document_id,
                            "user_id": user_id,
                            "content": chunk.content,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            "embedding": embedding,
                            "metadata": chunk.metadata,
                        }
                    )
                self.admin.table("chunks").insert(rows).execute()

            self._update_status(
                document_id, "ready", 100, total_chunks=len(chunks)
            )

            logger.info(
                f"Document {document_id} processed successfully: {len(chunks)} chunks"
            )

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            self._update_status(document_id, "failed", 0)
            raise

    def _store_chunks(
        self,
        document_id: str,
        user_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ):
        """Insert chunks with embeddings into the database."""
        BATCH_SIZE = 50
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i : i + BATCH_SIZE]
            batch_embeddings = embeddings[i : i + BATCH_SIZE]

            rows = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                rows.append(
                    {
                        "document_id": document_id,
                        "user_id": user_id,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "embedding": embedding,
                        "metadata": chunk.metadata,
                    }
                )

            self.admin.table("chunks").insert(rows).execute()

    # ── Status Management ──────────────────────────────────────────

    def _update_status(
        self,
        document_id: str,
        status: str,
        progress: int,
        total_chunks: int = None,
    ):
        """Update document processing status and progress."""
        update_data = {"status": status, "indexing_progress": progress}
        if total_chunks is not None:
            update_data["total_chunks"] = total_chunks

        self.admin.table("documents").update(update_data).eq(
            "id", document_id
        ).execute()

    def _document_exists(self, document_id: str) -> bool:
        """Check if a document still exists in the database."""
        result = self.admin.table("documents").select("id").eq("id", document_id).execute()
        return len(result.data) > 0

    def _get_document(self, document_id: str) -> dict:
        """Get a document by ID."""
        result = (
            self.admin.table("documents")
            .select("*")
            .eq("id", document_id)
            .single()
            .execute()
        )
        return result.data

    # ── CRUD ───────────────────────────────────────────────────────

    def list_documents(self, user_id: str) -> List[dict]:
        """List all documents for a user."""
        result = (
            self.admin.table("documents")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    def get_document(self, document_id: str, user_id: str) -> Optional[dict]:
        """Get a single document, verifying ownership."""
        result = (
            self.admin.table("documents")
            .select("*")
            .eq("id", document_id)
            .eq("user_id", user_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete a document, its chunks, and its storage file."""
        doc = self.get_document(document_id, user_id)
        if not doc:
            return False

        # Delete from storage
        try:
            self.admin.storage.from_(self.STORAGE_BUCKET).remove([doc["file_path"]])
        except Exception as e:
            logger.warning(f"Failed to delete storage file: {e}")

        # Delete chunks (cascade should handle this, but be explicit)
        self.admin.table("chunks").delete().eq(
            "document_id", document_id
        ).execute()

        # Delete document record
        self.admin.table("documents").delete().eq("id", document_id).execute()

        return True
