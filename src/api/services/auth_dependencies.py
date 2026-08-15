"""
FastAPI auth dependencies for current user.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .account_service import get_account_service
from .auth_service import decode_access_token

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)

# Every authenticated route (all of /patients/*, saved studies/cases, etc.)
# depends on this. If the accounts DB is unreachable, that would otherwise
# surface as a raw, possibly credential-leaking 500 on completely unrelated
# endpoints — e.g. "create patient" failing for accounts-DB reasons that
# have nothing to do with patient creation itself.
_AUTH_SERVICE_ERROR_MSG = "Couldn't verify your session. Please try again in a moment."


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    if credentials is None:
        return None
    user_id = decode_access_token(credentials.credentials)
    if not user_id:
        return None
    account_service = get_account_service()
    try:
        return await account_service.get_user_by_id(user_id)
    except Exception:
        logger.exception("[get_current_user_optional] account lookup failed")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user_id = decode_access_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    account_service = get_account_service()
    try:
        user = await account_service.get_user_by_id(user_id)
    except Exception:
        logger.exception("[get_current_user] account lookup failed")
        raise HTTPException(status_code=503, detail=_AUTH_SERVICE_ERROR_MSG)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ── Role guards ────────────────────────────────────────────────────────────
# Added 2026-08-08 with the patient portal. `role` defaults to 'physician'
# for every pre-existing account, so require_physician is a no-op for all
# current users and no existing route changes behaviour by adopting it.


def user_role(user: dict) -> str:
    """Role of a user dict, defaulting to physician for legacy rows."""
    return (user or {}).get("role") or "physician"


async def require_physician(current_user=Depends(get_current_user)):
    """Allow physician (and admin) accounts only."""
    if user_role(current_user) not in ("physician", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This area is for clinician accounts.",
        )
    return current_user


async def require_patient(current_user=Depends(get_current_user)):
    """Allow patient accounts only."""
    if user_role(current_user) != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This area is for patient accounts.",
        )
    return current_user
