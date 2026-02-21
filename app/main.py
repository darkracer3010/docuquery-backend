from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.auth.router import router as auth_router
from app.documents.router import router as documents_router
from app.qa.router import router as qa_router

settings = get_settings()

app = FastAPI(
    title="DocuQuery API",
    description="RAG-based Smart Document Q&A System",
    version="1.0.0",
)

# CORS
origins = [origin.strip() for origin in settings.cors_origins.split(",")]
allow_all = "*" in origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else origins,
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(documents_router, prefix="/documents", tags=["Documents"])
app.include_router(qa_router, prefix="/qa", tags=["Q&A"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "docuquery-api"}
