"""
Patient portal routes (added 2026-08-08).

Split into two groups:

* ``/api/portal/*``    patient-authenticated (role='patient')
* ``/api/portal/clinician/*``  physician-authenticated (role='physician')

Kept in a separate router from ``patient_cases.py`` so the physician
product's existing endpoints are untouched. Everything here is additive.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.services.auth_dependencies import require_patient, require_physician
from src.api.services.patient_portal.patient_link_service import (
    get_patient_link_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portal", tags=["Patient Portal"])

_SERVER_ERROR = "Something went wrong on our end. Please try again in a moment."


# ── Models ─────────────────────────────────────────────────────────────────

class ClaimInviteRequest(BaseModel):
    invite_code: str = Field(min_length=4, max_length=64)


class LinkRequestBody(BaseModel):
    physician_id: str
    first_name: Optional[str] = Field(default=None, max_length=100)
    last_name: Optional[str] = Field(default=None, max_length=100)
    date_of_birth: Optional[str] = Field(default=None, max_length=32)
    note: Optional[str] = Field(default=None, max_length=500)


class ApproveRequestBody(BaseModel):
    patient_record_id: str


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class EscalateBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    ai_draft_answer: Optional[str] = Field(default=None, max_length=8000)
    conversation_id: Optional[str] = None
    urgency: str = Field(default="routine", pattern="^(routine|soon|urgent)$")


class RespondBody(BaseModel):
    response: str = Field(min_length=1, max_length=8000)


class MedicationBody(BaseModel):
    medication: str = Field(min_length=1, max_length=200)


class ReportBody(BaseModel):
    report_text: str = Field(min_length=20, max_length=20000)


class SymptomBody(BaseModel):
    symptom: str = Field(min_length=1, max_length=200)
    severity: Optional[int] = Field(default=None, ge=1, le=5)
    noted_on: Optional[str] = Field(default=None, max_length=32)
    note: Optional[str] = Field(default=None, max_length=1000)


# ── Patient-facing ─────────────────────────────────────────────────────────

@router.get("/me")
async def portal_me(current_user=Depends(require_patient)):
    """Patient's own view: who they are and whether they're connected."""
    svc = get_patient_link_service()
    try:
        record = await svc.get_linked_record(current_user["id"])
        pending = None if record else await svc.pending_request_for(current_user["id"])
    except Exception:
        logger.exception("[portal/me] lookup failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)

    return {
        "user": {
            "id": current_user["id"],
            "email": current_user.get("email"),
            "first_name": current_user.get("first_name"),
            "last_name": current_user.get("last_name"),
            "role": current_user.get("role", "patient"),
        },
        "linked": record is not None,
        "record": record,
        "pending_request": pending,
    }


@router.post("/claim-invite")
async def claim_invite(body: ClaimInviteRequest, current_user=Depends(require_patient)):
    """Connect this account using a code from the care team."""
    try:
        return await get_patient_link_service().claim_invite(
            invite_code=body.invite_code,
            patient_user_id=current_user["id"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/claim-invite] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.post("/request-link")
async def request_link(body: LinkRequestBody, current_user=Depends(require_patient)):
    """Ask a physician to connect. Creates a pending request only.

    Deliberately does not grant any access. Selecting a clinician from a
    list is a request; only the clinician can approve it.
    """
    try:
        return await get_patient_link_service().request_link(
            patient_user_id=current_user["id"],
            physician_id=body.physician_id,
            first_name=body.first_name or current_user.get("first_name"),
            last_name=body.last_name or current_user.get("last_name"),
            date_of_birth=body.date_of_birth,
            note=body.note,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/request-link] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.get("/physicians")
async def list_physicians(q: Optional[str] = None, current_user=Depends(require_patient)):
    """Searchable clinician directory for the connect flow.

    Returns only name and institution, never contact details or patient
    counts. A patient seeing a name here grants them nothing; it only
    lets them address a request, which that clinician must approve.

    Only clinicians who actually have patient records are listed.
    Registration is currently open, so without this filter anyone could
    sign up and be presented to patients as a clinician. Having a patient
    record is also a precondition for approving a request at all, so this
    excludes nobody who could act on one.

    A real verification step (credential check, or an allowlist) is the
    right long-term answer. See PATIENT_PLATFORM_PLAN.md.
    """
    try:
        # Cross-database: users live in the accounts DB, patients in the
        # patients DB, so this cannot be a single join.
        from src.api.services.patient_db import get_patient_db
        pdb = get_patient_db()
        await pdb.ensure_schema()
        ppool = await pdb.get_pool()
        async with ppool.acquire() as pconn:
            active_rows = await pconn.fetch(
                "SELECT DISTINCT physician_id FROM patients"
            )
        active_ids = [str(r["physician_id"]) for r in active_rows]
        if not active_ids:
            return {"physicians": []}

        from src.api.services.account_db import get_account_db
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            if q and q.strip():
                rows = await conn.fetch(
                    """
                    SELECT id, first_name, last_name, institution
                      FROM users
                     WHERE role = 'physician'
                       AND id = ANY($2::uuid[])
                       AND (first_name ILIKE $1 OR last_name ILIKE $1
                            OR institution ILIKE $1)
                     ORDER BY last_name NULLS LAST, first_name NULLS LAST
                     LIMIT 50
                    """,
                    f"%{q.strip()}%", active_ids,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, first_name, last_name, institution
                      FROM users
                     WHERE role = 'physician'
                       AND id = ANY($1::uuid[])
                     ORDER BY last_name NULLS LAST, first_name NULLS LAST
                     LIMIT 50
                    """,
                    active_ids,
                )
        return {
            "physicians": [
                {
                    "id": str(r["id"]),
                    "name": " ".join(
                        p for p in [r["first_name"], r["last_name"]] if p
                    ) or "Clinician",
                    "institution": r["institution"],
                }
                for r in rows
            ]
        }
    except Exception:
        logger.exception("[portal/physicians] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


# ── Patient chat ───────────────────────────────────────────────────────────

@router.post("/chat")
async def patient_chat(body: ChatBody, current_user=Depends(require_patient)):
    """Ask a question. Safety triage runs before anything is generated."""
    from src.api.services.patient_portal.patient_chat_service import (
        get_patient_chat_service,
    )
    try:
        result = await get_patient_chat_service().answer(
            message=body.message,
            patient_user_id=current_user["id"],
            conversation_history=body.conversation_history,
            conversation_id=body.conversation_id,
        )
        return result.to_dict()
    except RuntimeError:
        # Generation failed upstream. Deliberately generic: this endpoint
        # is patient-facing and must never surface internals.
        raise HTTPException(
            status_code=503,
            detail="I couldn't answer that just now. Please try again in a moment.",
        )
    except Exception:
        logger.exception("[portal/chat] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.get("/conversations")
async def list_conversations(current_user=Depends(require_patient)):
    try:
        from src.api.services.patient_db import get_patient_db
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, created_at, updated_at
                  FROM patient_conversations
                 WHERE patient_user_id = $1
                 ORDER BY updated_at DESC LIMIT 50
                """,
                current_user["id"],
            )
        return {
            "conversations": [
                {
                    "id": str(r["id"]),
                    "title": r["title"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                }
                for r in rows
            ]
        }
    except Exception:
        logger.exception("[portal/conversations] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, current_user=Depends(require_patient)):
    """Messages in one conversation. Ownership-scoped."""
    try:
        from src.api.services.patient_db import get_patient_db
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            owner = await conn.fetchval(
                "SELECT patient_user_id FROM patient_conversations WHERE id = $1",
                conversation_id,
            )
            if owner is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if str(owner) != str(current_user["id"]):
                raise HTTPException(status_code=403, detail="Not your conversation")
            rows = await conn.fetch(
                """
                SELECT role, content, safety_category, sources, created_at
                  FROM patient_messages
                 WHERE conversation_id = $1
                 ORDER BY created_at
                """,
                conversation_id,
            )
        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "safety_category": r["safety_category"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ],
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("[portal/conversation] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.post("/escalate")
async def escalate(body: EscalateBody, current_user=Depends(require_patient)):
    """Send a question to the patient's physician.

    Requires a linked care team: there is nowhere to route it otherwise.
    """
    from src.api.services.patient_portal.patient_chat_service import (
        get_patient_chat_service,
    )
    from src.api.services.patient_portal.escalation_service import (
        get_escalation_service,
    )
    try:
        facts = await get_patient_chat_service().known_facts_for(current_user["id"])
        if not facts.get("physician_id"):
            raise HTTPException(
                status_code=400,
                detail="Connect to your care team first so we know who to send this to.",
            )

        # conversation_id is client-supplied. Verify it belongs to this
        # patient, otherwise the physician's reply would be written into
        # someone else's conversation when they answer.
        if body.conversation_id:
            from src.api.services.patient_db import get_patient_db
            db = get_patient_db()
            await db.ensure_schema()
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                owner = await conn.fetchval(
                    "SELECT patient_user_id FROM patient_conversations WHERE id = $1",
                    body.conversation_id,
                )
            if owner is None or str(owner) != str(current_user["id"]):
                raise HTTPException(status_code=403, detail="Not your conversation")

        return await get_escalation_service().create(
            patient_user_id=current_user["id"],
            physician_id=facts["physician_id"],
            question=body.question,
            ai_draft_answer=body.ai_draft_answer,
            patient_record_id=facts.get("patient_record_id"),
            conversation_id=body.conversation_id,
            facts=facts,
            urgency=body.urgency,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("[portal/escalate] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.get("/my-questions")
async def my_questions(current_user=Depends(require_patient)):
    from src.api.services.patient_portal.escalation_service import get_escalation_service
    try:
        return {"questions": await get_escalation_service().list_for_patient(current_user["id"])}
    except Exception:
        logger.exception("[portal/my-questions] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


# ── Patient tools ──────────────────────────────────────────────────────────

@router.post("/medication")
async def explain_medication(body: MedicationBody, current_user=Depends(require_patient)):
    """Plain-language evidence summary for one medication."""
    from src.api.services.patient_portal.patient_tools_service import (
        get_patient_tools_service,
    )
    try:
        result = await get_patient_tools_service().explain_medication(
            medication=body.medication, patient_user_id=current_user["id"]
        )
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/medication] failed")
        raise HTTPException(
            status_code=503,
            detail="I couldn't look that up just now. Please try again in a moment.",
        )


@router.post("/report")
async def explain_report(body: ReportBody, current_user=Depends(require_patient)):
    """Explain the terminology in a report. Never interprets results."""
    from src.api.services.patient_portal.patient_tools_service import (
        get_patient_tools_service,
    )
    try:
        result = await get_patient_tools_service().explain_report(body.report_text)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/report] failed")
        raise HTTPException(
            status_code=503,
            detail="I couldn't read that just now. Please try again in a moment.",
        )


@router.post("/question-prep")
async def question_prep(current_user=Depends(require_patient)):
    """Turn this patient's recent questions into a list for their next visit."""
    from src.api.services.patient_portal.patient_tools_service import (
        get_patient_tools_service,
    )
    try:
        from src.api.services.patient_db import get_patient_db
        db = get_patient_db()
        await db.ensure_schema()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT m.content
                  FROM patient_messages m
                  JOIN patient_conversations c ON c.id = m.conversation_id
                 WHERE c.patient_user_id = $1 AND m.role = 'patient'
                 ORDER BY m.created_at DESC
                 LIMIT 20
                """,
                current_user["id"],
            )
        topics = [r["content"] for r in rows]
        result = await get_patient_tools_service().prepare_questions(topics)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/question-prep] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


# ── Symptom diary ──────────────────────────────────────────────────────────

@router.post("/symptoms")
async def add_symptom(body: SymptomBody, current_user=Depends(require_patient)):
    from src.api.services.patient_portal.symptom_service import get_symptom_service
    from src.api.services.patient_portal.patient_chat_service import (
        get_patient_chat_service,
    )
    try:
        facts = await get_patient_chat_service().known_facts_for(current_user["id"])
        return await get_symptom_service().add(
            patient_user_id=current_user["id"],
            symptom=body.symptom,
            severity=body.severity,
            noted_on=body.noted_on,
            note=body.note,
            patient_record_id=facts.get("patient_record_id"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/symptoms:add] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.get("/symptoms")
async def list_symptoms(current_user=Depends(require_patient)):
    from src.api.services.patient_portal.symptom_service import get_symptom_service
    try:
        return {"entries": await get_symptom_service().list_entries(current_user["id"])}
    except Exception:
        logger.exception("[portal/symptoms:list] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.delete("/symptoms/{entry_id}")
async def delete_symptom(entry_id: str, current_user=Depends(require_patient)):
    from src.api.services.patient_portal.symptom_service import get_symptom_service
    try:
        return {"deleted": await get_symptom_service().delete(entry_id, current_user["id"])}
    except Exception:
        logger.exception("[portal/symptoms:delete] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.post("/symptoms/share")
async def share_symptoms(current_user=Depends(require_patient)):
    """Send a symptom summary to the care team via the physician inbox."""
    from src.api.services.patient_portal.symptom_service import get_symptom_service
    from src.api.services.patient_portal.patient_chat_service import (
        get_patient_chat_service,
    )
    from src.api.services.patient_portal.escalation_service import (
        get_escalation_service,
    )
    try:
        facts = await get_patient_chat_service().known_facts_for(current_user["id"])
        if not facts.get("physician_id"):
            raise HTTPException(
                status_code=400,
                detail="Connect to your care team first so we know who to send this to.",
            )
        svc = get_symptom_service()
        entries = await svc.list_entries(current_user["id"], limit=200)
        if not entries:
            raise HTTPException(status_code=400, detail="Nothing logged yet.")

        # The summary IS the content the physician needs to see, so it
        # goes in the question field rather than being returned only to
        # the patient. No AI draft: this is a report, not a question.
        summary = svc.build_summary(entries)
        created = await get_escalation_service().create(
            patient_user_id=current_user["id"],
            physician_id=facts["physician_id"],
            question=summary,
            ai_draft_answer=None,
            patient_record_id=facts.get("patient_record_id"),
            facts=facts,
            urgency="routine",
        )
        created["summary"] = summary
        return created
    except HTTPException:
        raise
    except Exception:
        logger.exception("[portal/symptoms:share] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


# ── Clinician-facing ───────────────────────────────────────────────────────

@router.post("/clinician/patients/{patient_id}/invite")
async def create_invite(patient_id: str, current_user=Depends(require_physician)):
    """Generate a single-use invite code for one of your patients."""
    try:
        return await get_patient_link_service().create_invite(
            patient_id=patient_id, physician_id=current_user["id"]
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not your patient")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/clinician/invite] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.get("/clinician/link-requests")
async def link_requests(status: str = "pending", current_user=Depends(require_physician)):
    """Patients asking to connect to you."""
    try:
        return {
            "requests": await get_patient_link_service().list_link_requests(
                physician_id=current_user["id"], status=status
            )
        }
    except Exception:
        logger.exception("[portal/clinician/link-requests] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.post("/clinician/link-requests/{request_id}/approve")
async def approve_request(
    request_id: str,
    body: ApproveRequestBody,
    current_user=Depends(require_physician),
):
    """Approve a request and bind it to one of your patient records."""
    try:
        return await get_patient_link_service().approve_link_request(
            request_id=request_id,
            physician_id=current_user["id"],
            patient_record_id=body.patient_record_id,
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not your request")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/clinician/approve] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.post("/clinician/link-requests/{request_id}/decline")
async def decline_request(request_id: str, current_user=Depends(require_physician)):
    try:
        ok = await get_patient_link_service().decline_link_request(
            request_id=request_id, physician_id=current_user["id"]
        )
        return {"declined": ok}
    except Exception:
        logger.exception("[portal/clinician/decline] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.get("/clinician/escalations")
async def clinician_escalations(
    status: str = "open", current_user=Depends(require_physician)
):
    """Patient questions waiting on you, each with a draft answer."""
    from src.api.services.patient_portal.escalation_service import get_escalation_service
    try:
        return {
            "escalations": await get_escalation_service().list_for_physician(
                physician_id=current_user["id"], status=status
            )
        }
    except Exception:
        logger.exception("[portal/clinician/escalations] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)


@router.post("/clinician/escalations/{escalation_id}/respond")
async def respond_escalation(
    escalation_id: str,
    body: RespondBody,
    current_user=Depends(require_physician),
):
    """Answer a patient question. Goes back into their conversation
    attributed to you, not to Paxis."""
    from src.api.services.patient_portal.escalation_service import get_escalation_service
    try:
        return await get_escalation_service().respond(
            escalation_id=escalation_id,
            physician_id=current_user["id"],
            response_text=body.response,
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Not your escalation")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[portal/clinician/respond] failed")
        raise HTTPException(status_code=503, detail=_SERVER_ERROR)
