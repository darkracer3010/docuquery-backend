from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.auth.dependencies import get_current_user
from app.dependencies import get_supabase_admin_client, get_openai_client
from app.embeddings.service import EmbeddingService
from app.qa.service import QAService
from app.qa.schemas import AskRequest, AskResponse, ChatMessage, Conversation

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


@router.post("/ask/stream")
async def ask_question_stream(
    body: AskRequest,
    user_id: str = Depends(get_current_user),
):
    """Stream a response to a question about your documents (SSE)."""
    if not body.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )

    service = _get_qa_service()

    return StreamingResponse(
        service.ask_stream(
            question=body.question,
            user_id=user_id,
            document_ids=body.document_ids,
            conversation_id=body.conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=List[Conversation])
async def get_conversations(
    user_id: str = Depends(get_current_user),
):
    """Get all conversations for the current user."""
    service = _get_qa_service()
    return await service.get_conversations(user_id)


@router.get("/conversations/{conversation_id}/messages", response_model=List[ChatMessage])
async def get_messages(
    conversation_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get message history for a conversation."""
    service = _get_qa_service()
    return await service.get_messages(conversation_id, user_id)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user),
):
    """Delete a conversation."""
    print(f"!!! RECEIVED DELETE REQUEST FOR CONVO {conversation_id} from user {user_id} !!!")
    service = _get_qa_service()
    success = await service.delete_conversation(conversation_id, user_id)
    print(f"!!! DELETE RESULT: {success} !!!")
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation",
        )
    return {"status": "success"}
