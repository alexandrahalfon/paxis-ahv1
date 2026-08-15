"""
Auth endpoints for account registration and login.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.models.auth_models import (
    AuthToken,
    LoginRequest,
    PatientRegisterRequest,
    RegisterRequest,
    UserResponse,
)
from src.api.services.account_service import get_account_service
from src.api.services.auth_dependencies import get_current_user
from src.api.services.auth_service import create_access_token, hash_password, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Generic message for anything unexpected (DB connectivity, etc.) — the real
# exception is logged server-side but never shown to the client, since it
# can contain connection details (host/user, occasionally the password in
# the exception string) that must not leak to the browser.
_SERVER_ERROR_MSG = "Something went wrong on our end. Please try again in a moment."


@router.post("/register", response_model=UserResponse)
async def register(request: RegisterRequest):
    """Clinician registration.

    Gated on the email domain because clinician accounts appear in a
    patient-facing directory. Patients register via /register/patient,
    which is deliberately not gated.
    """
    # Domain check before anything else, so a rejected signup never
    # touches the database.
    try:
        from src.api.services.clinician_domains import (
            DomainNotAllowed,
            check_clinician_email,
        )
        check_clinician_email(request.email)
    except DomainNotAllowed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        # A fault in the checker must not take registration down.
        logger.exception("[auth/register] domain check errored, allowing through")

    account_service = get_account_service()
    try:
        existing = await account_service.get_user_by_email(request.email)
    except Exception:
        logger.exception("[auth/register] Failed checking existing user")
        raise HTTPException(status_code=500, detail=_SERVER_ERROR_MSG)

    if existing:
        raise HTTPException(status_code=400, detail="An account with that email already exists. Try logging in instead.")

    try:
        password_hash = hash_password(request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        user = await account_service.create_user(
            email=request.email,
            password_hash=password_hash,
            first_name=request.first_name.strip(),
            last_name=request.last_name.strip(),
            institution=request.institution.strip(),
        )
    except Exception:
        logger.exception("[auth/register] Failed creating user")
        raise HTTPException(status_code=500, detail=_SERVER_ERROR_MSG)

    return UserResponse(**user)


@router.post("/register/patient", response_model=UserResponse)
async def register_patient(request: PatientRegisterRequest):
    """Patient self-registration.

    Separate from /register so the physician flow is untouched. Creates a
    role='patient' account, and if an invite code is supplied, links it to
    the physician's patient record immediately. A bad code does not fail
    the signup: the account is created and the patient can connect
    afterwards, which avoids stranding someone who mistyped a code.
    """
    account_service = get_account_service()
    try:
        existing = await account_service.get_user_by_email(request.email)
    except Exception:
        logger.exception("[auth/register/patient] Failed checking existing user")
        raise HTTPException(status_code=500, detail=_SERVER_ERROR_MSG)

    if existing:
        raise HTTPException(
            status_code=400,
            detail="An account with that email already exists. Try logging in instead.",
        )

    try:
        password_hash = hash_password(request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        user = await account_service.create_user(
            email=request.email,
            password_hash=password_hash,
            first_name=request.first_name.strip(),
            last_name=request.last_name.strip(),
            institution=None,
            role="patient",
        )
    except Exception:
        logger.exception("[auth/register/patient] Failed creating user")
        raise HTTPException(status_code=500, detail=_SERVER_ERROR_MSG)

    # Every patient account gets its own longitudinal profile immediately,
    # independent of whether a clinician is ever connected — see
    # patient_db.py's "Phase 0: patient-owned profile" comment for why
    # this replaced the old "profile only exists if a physician created
    # a chart" model. Non-fatal: a failure here must not block signup,
    # since ensure_profile is idempotent and safe to retry from anywhere
    # else in the app that assumes a profile exists.
    try:
        from src.api.services.patient.patient_profile_service import (
            get_patient_profile_service,
        )
        await get_patient_profile_service().ensure_profile(
            user_id=user["id"],
            first_name=user.get("first_name"),
            last_name=user.get("last_name"),
        )
    except Exception as e:
        logger.warning("[auth/register/patient] profile creation failed: %s", e)

    if request.invite_code:
        try:
            from src.api.services.patient_portal.patient_link_service import (
                get_patient_link_service,
            )
            await get_patient_link_service().claim_invite(
                invite_code=request.invite_code,
                patient_user_id=user["id"],
            )
        except Exception as e:
            # Non-fatal by design. The account exists; the patient can
            # enter the code again or request a connection from the app.
            logger.warning("[auth/register/patient] invite claim failed: %s", e)

    return UserResponse(**user)


@router.post("/login", response_model=AuthToken)
async def login(request: LoginRequest):
    account_service = get_account_service()
    try:
        user = await account_service.get_user_by_email(request.email)
    except Exception:
        logger.exception("[auth/login] Failed looking up user")
        raise HTTPException(status_code=500, detail=_SERVER_ERROR_MSG)

    # Distinguishing "no such account" from "wrong password" is a deliberate
    # trade-off: it's more helpful for a small beta user base than the
    # standard "invalid email or password" wording, at the cost of letting
    # someone probe which emails have accounts. Worth revisiting before a
    # wider launch.
    if not user:
        raise HTTPException(status_code=404, detail="No account found with that email. Check the email or create an account.")
    if not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    token = create_access_token(user["id"])
    return AuthToken(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(current_user=Depends(get_current_user)):
    return UserResponse(**current_user)
