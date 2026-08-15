"""
Authentication helpers: password hashing and JWT handling.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt
from passlib.context import CryptContext

from src.core.config import settings


# Use pbkdf2 to avoid bcrypt version issues on some systems.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.auth_access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.auth_secret_key, algorithm=settings.auth_algorithm)


def decode_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            token,
            settings.auth_secret_key,
            algorithms=[settings.auth_algorithm],
        )
        return payload.get("sub")
    except Exception:
        return None
