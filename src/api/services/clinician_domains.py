"""
Clinician email domain gating.

Applies to clinician registration only. Patients sign up with whatever
email they have, so this must never be applied to the patient path.

What this is and is not
-----------------------
This checks that a clinician's email *looks* institutional. It does NOT
verify the person controls that mailbox, because there is no email
infrastructure yet. Someone determined can still type a plausible
hospital address they do not own and get through.

So: this stops casual and accidental signups, and keeps obviously
personal addresses out of the patient-facing clinician directory. It is
a speed bump, not identity verification. Real verification means a
confirmation email (which also gets you password reset) and ideally a
credential check. See PATIENT_PLATFORM_PLAN.md.

Modes (``settings.clinician_email_mode``)
-----------------------------------------
``blocklist`` (default)
    Reject known consumer and disposable providers, allow anything else.
    Sensible for beta: no maintenance as new hospitals sign up.
``allowlist``
    Reject everything except domains in ``clinician_email_allowlist``.
    Tightest option. Use when you know exactly which institutions are
    onboarding.
``off``
    No checking. Escape hatch if this ever blocks a real user at a bad
    moment.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

from src.core.config import settings


# Consumer mailbox providers. A clinician using one of these for a
# patient-facing clinical account is the case we want to catch.
CONSUMER_DOMAINS: Set[str] = {
    "gmail.com", "googlemail.com",
    "yahoo.com", "yahoo.co.uk", "yahoo.co.in", "ymail.com", "rocketmail.com",
    "hotmail.com", "hotmail.co.uk", "outlook.com", "live.com", "msn.com",
    "aol.com", "icloud.com", "me.com", "mac.com",
    "proton.me", "protonmail.com", "pm.me",
    "gmx.com", "gmx.de", "web.de", "mail.com", "zoho.com",
    "yandex.com", "yandex.ru", "mail.ru",
    "qq.com", "163.com", "126.com", "naver.com", "daum.net",
    "comcast.net", "verizon.net", "att.net", "sbcglobal.net",
    "btinternet.com", "sky.com", "orange.fr", "free.fr", "libero.it",
}

# Throwaway / disposable providers.
DISPOSABLE_DOMAINS: Set[str] = {
    "mailinator.com", "guerrillamail.com", "10minutemail.com",
    "tempmail.com", "temp-mail.org", "throwawaymail.com", "yopmail.com",
    "sharklasers.com", "trashmail.com", "getnada.com", "dispostable.com",
    "maildrop.cc", "fakeinbox.com", "mintemail.com", "spamgourmet.com",
    "moakt.com", "emailondeck.com", "tempr.email", "mohmal.com",
}

BLOCKED_DOMAINS: Set[str] = CONSUMER_DOMAINS | DISPOSABLE_DOMAINS


class DomainNotAllowed(ValueError):
    """Raised when a clinician email domain fails the policy."""


def extract_domain(email: str) -> str:
    """Lowercased domain part of an email, or '' if unparseable."""
    if not email or "@" not in email:
        return ""
    return email.rsplit("@", 1)[1].strip().lower().rstrip(".")


def _domain_and_parents(domain: str) -> Tuple[str, ...]:
    """('mail.gmail.com',) -> ('mail.gmail.com', 'gmail.com', 'com')

    Checking parents means a subdomain like ``mail.gmail.com`` can't slip
    past a blocklist that only lists ``gmail.com``.
    """
    parts = domain.split(".")
    return tuple(".".join(parts[i:]) for i in range(len(parts)))


def _configured_allowlist() -> Set[str]:
    raw = getattr(settings, "clinician_email_allowlist", "") or ""
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def check_clinician_email(email: str) -> None:
    """Raise ``DomainNotAllowed`` if this email may not register as a clinician.

    Returns None on success so callers can simply let it raise.
    """
    mode = (getattr(settings, "clinician_email_mode", "blocklist") or "blocklist").lower()
    if mode == "off":
        return

    domain = extract_domain(email)
    if not domain or "." not in domain:
        raise DomainNotAllowed("Please enter a valid email address.")

    candidates = _domain_and_parents(domain)

    if mode == "allowlist":
        allowed = _configured_allowlist()
        if not allowed:
            # Misconfiguration: an empty allowlist in allowlist mode would
            # lock out every new clinician. Fail open to the blocklist
            # rather than silently blocking all signups.
            mode = "blocklist"
        elif not any(c in allowed for c in candidates):
            raise DomainNotAllowed(
                "Clinician accounts are limited to approved institutions right now. "
                "If you should have access, contact us and we'll add your organisation."
            )
        else:
            return

    # blocklist mode
    if any(c in BLOCKED_DOMAINS for c in candidates):
        raise DomainNotAllowed(
            "Please register with your work or institutional email address. "
            "Personal email accounts can't be used for clinician access."
        )


def is_allowed_clinician_email(email: str) -> bool:
    """Boolean form, for callers that don't want the exception."""
    try:
        check_clinician_email(email)
        return True
    except DomainNotAllowed:
        return False
