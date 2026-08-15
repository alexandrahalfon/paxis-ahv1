"""
Cache stats endpoints.
"""

from fastapi import APIRouter, Depends

from src.api.services.auth_dependencies import get_current_user
from src.api.services.cache_service import get_cache_service


router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/hits")
async def get_cache_hits(limit: int = 20, current_user: dict = Depends(get_current_user)):
    cache_service = get_cache_service()
    return await cache_service.get_top_hits(current_user["id"], limit=limit)
