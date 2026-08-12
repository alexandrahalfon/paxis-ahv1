"""
Account persistence for user registration/login.
"""

import uuid
from typing import Dict, Optional

from .account_db import get_account_db


class AccountService:
    """CRUD for user accounts."""

    async def create_user(
        self,
        email: str,
        password_hash: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        institution: Optional[str] = None,
        role: str = "physician",
    ) -> Dict:
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        user_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (id, email, password_hash, first_name, last_name, institution, role)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                user_id,
                email.lower().strip(),
                password_hash,
                first_name,
                last_name,
                institution,
                role,
            )
        return {
            "id": user_id,
            "email": email.lower().strip(),
            "first_name": first_name,
            "last_name": last_name,
            "institution": institution,
            "role": role,
        }

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, password_hash, first_name, last_name, institution, role
                FROM users
                WHERE email = $1
                """,
                email.lower().strip(),
            )
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "email": row["email"],
            "password_hash": row["password_hash"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "institution": row["institution"],
            "role": row["role"] or "physician",
        }

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        db = get_account_db()
        await db.ensure_schema()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, first_name, last_name, institution, role
                FROM users
                WHERE id = $1
                """,
                user_id,
            )
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "email": row["email"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "institution": row["institution"],
            "role": row["role"] or "physician",
        }


_account_service: Optional[AccountService] = None


def get_account_service() -> AccountService:
    global _account_service
    if _account_service is None:
        _account_service = AccountService()
    return _account_service
