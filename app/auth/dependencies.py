import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import lru_cache

from app.config import get_settings

security = HTTPBearer()

def get_jwks_client():
    """Get PyJWKClient for Supabase."""
    settings = get_settings()
    jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
    return jwt.PyJWKClient(jwks_url)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Verify Supabase JWT and return the user ID.
    Supports both ES256 (via JWKS) and HS256 (via secret).
    """
    settings = get_settings()
    token = credentials.credentials
    
    try:
        # 1. First attempt: Verify using JWKS (handles ES256/RS256)
        # This is more robust as it fetches public keys from Supabase
        jwks_client = get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={"verify_aud": False}
        )
        return payload.get("sub")
    except Exception as e:
        # 2. Fallback: Try HS256 with the secret
        # This is for projects still using symmetric keys
        try:
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                options={"verify_aud": False}
            )
            user_id: str = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing user ID",
                )
            return user_id
        except jwt.PyJWTError as hs_e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(hs_e)} (fallback failed after JWKS error: {str(e)})",
            )
