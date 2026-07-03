from __future__ import annotations

from fastapi import Depends
from supabase import create_client
from supabase.client import Client

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError


def get_supabase_client(settings: Settings = Depends(get_settings)) -> Client | None:
    supabase_url = settings.supabase_url
    service_role_key = settings.supabase_service_role_key

    if supabase_url is None or not service_role_key:
        return None
    return create_client(str(supabase_url), service_role_key)


def require_supabase_client(
    supabase: Client | None = Depends(get_supabase_client),
) -> Client:
    if supabase is None:
        raise AppError("Supabase is not configured", status_code=503)
    return supabase
