from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from typing import List

from app.auth.dependencies import get_current_user
from app.dependencies import get_supabase_client, get_supabase_admin_client, get_openai_client
from app.documents.service import DocumentService
from app.documents.schemas import DocumentUploadResponse, DocumentResponse, DocumentListResponse
from app.embeddings.service import EmbeddingService

router = APIRouter()


def _get_document_service() -> DocumentService:
    return DocumentService(
        supabase_client=get_supabase_client(),
        admin_client=get_supabase_admin_client(),
        embedding_service=EmbeddingService(get_openai_client()),
    )


# ── Upload ─────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    """Upload a document for processing."""
    service = _get_document_service()

    file_bytes = await file.read()
    file_size = len(file_bytes)

    try:
        doc = service.upload_document(user_id, file_bytes, file.filename, file_size)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Trigger background indexing
    background_tasks.add_task(service.process_document, doc["id"], user_id)

    return DocumentUploadResponse(
        id=doc["id"],
        file_name=doc["file_name"],
        status=doc["status"],
        message="Document uploaded. Processing started in background.",
    )


@router.post("/upload-bulk", response_model=List[DocumentUploadResponse])
async def upload_documents_bulk(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
):
    """Upload multiple documents at once."""
    service = _get_document_service()
    results = []

    for file in files:
        file_bytes = await file.read()
        file_size = len(file_bytes)

        try:
            doc = service.upload_document(user_id, file_bytes, file.filename, file_size)
            background_tasks.add_task(service.process_document, doc["id"], user_id)
            results.append(
                DocumentUploadResponse(
                    id=doc["id"],
                    file_name=doc["file_name"],
                    status=doc["status"],
                    message="Processing started.",
                )
            )
        except ValueError as e:
            results.append(
                DocumentUploadResponse(
                    id="",
                    file_name=file.filename,
                    status="failed",
                    message=str(e),
                )
            )

    return results


# ── List / Get / Delete ────────────────────────────────────────────

@router.get("/", response_model=DocumentListResponse)
async def list_documents(user_id: str = Depends(get_current_user)):
    """List all documents for the current user."""
    service = _get_document_service()
    docs = service.list_documents(user_id)
    return DocumentListResponse(
        documents=[DocumentResponse(**d) for d in docs],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get a single document's details and processing status."""
    service = _get_document_service()
    doc = service.get_document(document_id, user_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return DocumentResponse(**doc)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
):
    """Delete a document and all its chunks."""
    service = _get_document_service()
    success = service.delete_document(document_id, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return {"message": "Document deleted successfully"}
