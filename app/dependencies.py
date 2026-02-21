from functools import lru_cache
from supabase import create_client, Client
from openai import OpenAI

from app.config import get_settings


@lru_cache()
def get_supabase_client() -> Client:
    """Get Supabase client using the anon key (for user-scoped operations)."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_anon_key)


@lru_cache()
def get_supabase_admin_client() -> Client:
    """Get Supabase client using the service role key (bypasses RLS)."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@lru_cache()
def get_openai_client() -> OpenAI:
    """Get OpenAI client."""
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)
