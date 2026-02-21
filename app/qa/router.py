from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.dependencies import get_supabase_admin_client, get_openai_client
from app.embeddings.service import EmbeddingService
from app.qa.service import QAService
from app.qa.schemas import AskRequest, AskResponse

router = APIRouter()


def _get_qa_service() -> QAService:
    openai_client = get_openai_client()
    return QAService(
        supabase_admin=get_supabase_admin_client(),
        openai_client=openai_client,
        embedding_service=EmbeddingService(openai_client),
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Ask a question about your uploaded documents.

    The system will:
    1. Check semantic cache for similar past questions
    2. Retrieve relevant chunks via vector similarity search
    3. Re-rank chunks using LLM for better relevance
    4. Generate an answer with source citations
    """
    if not body.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )

    service = _get_qa_service()

    try:
        result = await service.ask(
            question=body.question,
            user_id=user_id,
            document_ids=body.document_ids,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}",
        )
