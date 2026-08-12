"""
Community Routes (Phase 7)

Kept structurally separate from every clinical/evidence route: nothing
here calls into comprehensive_retrieval.py, enhanced_rag_service.py, or
the evidence/ package, and nothing in those calls into this file or the
community/ service package. See architecture review sections 32-34.

Open to both patient and physician accounts (get_current_user, not
require_patient) — a caregiver or clinician participating as a person is
a legitimate use of a support community, and access is gated by having a
pseudonymous community_profile, not by account role.

One router with explicit full paths (rather than several prefixed
sub-routers) since the resource paths don't share a common prefix:
/communities/*, /posts/*, /comments/*, /community/* all live at the API
root, mirroring how saved_studies/saved_cases already sit alongside
patients/portal without a shared parent.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.services.auth_dependencies import get_current_user, user_role

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Communities"])

_SERVER_ERROR = "Something went wrong on our end. Please try again in a moment."


async def get_own_community_profile(current_user: dict = Depends(get_current_user)) -> dict:
    from src.api.services.community.community_service import get_community_service
    try:
        return await get_community_service().ensure_profile(current_user["id"])
    except Exception:
        logger.exception("[communities] profile resolution failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if user_role(current_user) != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")
    return current_user


class CreateCommunityBody(BaseModel):
    slug: str = Field(min_length=2, max_length=80)
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    category: Optional[str] = Field(default=None, max_length=80)


class CreatePostBody(BaseModel):
    title: Optional[str] = Field(default=None, max_length=300)
    body: str = Field(min_length=1, max_length=10000)


class CreateCommentBody(BaseModel):
    body: str = Field(min_length=1, max_length=5000)


class ReactBody(BaseModel):
    reaction: str = Field(default="support", max_length=40)


class ReportBody(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=1000)


class BlockBody(BaseModel):
    # community_profile_id, not a real account user_id — the only
    # identifier of another person a client ever legitimately has is the
    # one returned on posts/comments (community_profile_id / handle).
    # Accepting a raw user_id here would have let a client block (and,
    # via ensure_profile, silently create a community_profile for)
    # anyone whose real account id it obtained some other way, breaking
    # the pseudonymity boundary this package's __init__ documents.
    blocked_profile_id: str


class UpdateHandleBody(BaseModel):
    handle: str = Field(min_length=3, max_length=40)


class ResolveReportBody(BaseModel):
    action: str = Field(pattern="^(hide|dismiss)$")
    reason: Optional[str] = Field(default=None, max_length=1000)


# ── Community profile ───────────────────────────────────────────────────

@router.get("/community/profile")
async def get_my_community_profile(profile: dict = Depends(get_own_community_profile)):
    return {"profile": profile}


@router.put("/community/profile")
async def update_my_handle(body: UpdateHandleBody, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.community_service import get_community_service
    try:
        updated = await get_community_service().update_handle(profile["user_id"], body.handle)
        return {"success": True, "profile": updated}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/community/block")
async def block_user(body: BlockBody, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.post_service import get_post_service
    return await get_post_service().block_user(profile["id"], body.blocked_profile_id)


# ── Communities ──────────────────────────────────────────────────────────

@router.get("/communities")
async def list_communities():
    from src.api.services.community.community_service import get_community_service
    return {"communities": await get_community_service().list_communities()}


@router.get("/communities/mine")
async def list_my_communities(profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.community_service import get_community_service
    return {"communities": await get_community_service().list_my_communities(profile["id"])}


@router.post("/communities")
async def create_community(body: CreateCommunityBody, current_user: dict = Depends(get_current_user)):
    from src.api.services.community.community_service import get_community_service
    community = await get_community_service().create_community(
        slug=body.slug, name=body.name, description=body.description, category=body.category,
    )
    return {"success": True, "community": community}


@router.post("/communities/{slug}/join")
async def join_community(slug: str, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.community_service import get_community_service
    try:
        return await get_community_service().join(slug, profile["id"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/communities/{slug}/leave")
async def leave_community(slug: str, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.community_service import get_community_service
    ok = await get_community_service().leave(slug, profile["id"])
    return {"left": ok}


@router.get("/communities/{slug}/posts")
async def list_posts(slug: str, limit: int = 50):
    from src.api.services.community.post_service import get_post_service
    return {"posts": await get_post_service().list_posts(slug, limit=min(max(limit, 1), 100))}


@router.post("/communities/{slug}/posts")
async def create_post(slug: str, body: CreatePostBody, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.post_service import get_post_service
    try:
        post = await get_post_service().create_post(slug, profile["id"], body.title, body.body)
        return {"success": True, "post": post}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Posts / comments ─────────────────────────────────────────────────────

@router.get("/posts/{post_id}")
async def get_post(post_id: str):
    from src.api.services.community.post_service import get_post_service
    post = await get_post_service().get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    comments = await get_post_service().list_comments(post_id)
    return {"post": post, "comments": comments}


@router.post("/posts/{post_id}/comments")
async def add_comment(post_id: str, body: CreateCommentBody, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.post_service import get_post_service
    try:
        comment = await get_post_service().add_comment(post_id, profile["id"], body.body)
        return {"success": True, "comment": comment}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/posts/{post_id}/react")
async def react_to_post(post_id: str, body: ReactBody, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.post_service import get_post_service
    return await get_post_service().react("post", post_id, profile["id"], body.reaction)


@router.post("/posts/{post_id}/report")
async def report_post(post_id: str, body: ReportBody, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.post_service import get_post_service
    return await get_post_service().report("post", post_id, profile["id"], body.reason)


@router.post("/comments/{comment_id}/report")
async def report_comment(comment_id: str, body: ReportBody, profile: dict = Depends(get_own_community_profile)):
    from src.api.services.community.post_service import get_post_service
    return await get_post_service().report("comment", comment_id, profile["id"], body.reason)


# ── Moderation (admin only) ───────────────────────────────────────────────

@router.get("/communities/moderation/reports")
async def list_reports(status: str = "open", current_user: dict = Depends(require_admin)):
    from src.api.services.community.moderation_service import get_moderation_service
    return {"reports": await get_moderation_service().list_reports(status)}


@router.post("/communities/moderation/reports/{report_id}/resolve")
async def resolve_report(
    report_id: str, body: ResolveReportBody, current_user: dict = Depends(require_admin),
):
    from src.api.services.community.moderation_service import get_moderation_service
    try:
        return await get_moderation_service().resolve_report(
            report_id, body.action, current_user["id"], body.reason
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
