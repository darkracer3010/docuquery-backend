from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.auth.dependencies import get_current_user
from app.auth.service import AuthService
from app.dependencies import get_supabase_client

router = APIRouter()


# ── Request / Response Models ──────────────────────────────────────

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    user_id: str | None
    email: str | None
    access_token: str | None
    refresh_token: str | None


class ProfileResponse(BaseModel):
    id: str
    full_name: str | None
    created_at: str


# ── Endpoints ──────────────────────────────────────────────────────

@router.post("/signup", response_model=AuthResponse)
async def signup(body: SignUpRequest):
    """Register a new user."""
    service = AuthService(get_supabase_client())
    try:
        result = service.sign_up(body.email, body.password, body.full_name)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=AuthResponse)
async def login(body: SignInRequest):
    """Sign in with email and password."""
    service = AuthService(get_supabase_client())
    try:
        result = service.sign_in(body.email, body.password)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.get("/me", response_model=ProfileResponse)
async def get_me(user_id: str = Depends(get_current_user)):
    """Get the current user's profile."""
    service = AuthService(get_supabase_client())
    profile = service.get_profile(user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found",
        )
    return profile
