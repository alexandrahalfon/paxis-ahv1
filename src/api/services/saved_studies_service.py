"""
Saved Studies Service

Stores and retrieves saved studies/sources for users.
Allows users to bookmark studies from search results for later reference.
"""

from typing import Dict, Any, Optional, List
from .account_db import get_account_db


class SavedStudiesService:
    """Service for managing saved studies per user"""
    
    async def _ensure_schema(self):
        """Ensure the saved_studies table exists"""
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS saved_studies (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    
                    -- Study identification
                    study_id VARCHAR(255) NOT NULL,
                    title TEXT,
                    doi VARCHAR(255),
                    pmid VARCHAR(50),
                    
                    -- Metadata
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- Unique constraint per user
                    UNIQUE(user_id, study_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_saved_studies_user_id 
                ON saved_studies(user_id);
                
                CREATE INDEX IF NOT EXISTS idx_saved_studies_user_created 
                ON saved_studies(user_id, created_at DESC);
            """)
    
    async def save_study(
        self,
        user_id: str,
        study_id: str,
        title: Optional[str] = None,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a study for a user.
        
        Args:
            user_id: User's UUID
            study_id: Unique study identifier (doc_id, pmid, or doi)
            title: Study title
            doi: DOI if available
            pmid: PMID if available
            
        Returns:
            The saved study record
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Use INSERT ... ON CONFLICT to handle duplicates
            row = await conn.fetchrow("""
                INSERT INTO saved_studies (user_id, study_id, title, doi, pmid)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, study_id) DO UPDATE 
                SET title = COALESCE(EXCLUDED.title, saved_studies.title),
                    doi = COALESCE(EXCLUDED.doi, saved_studies.doi),
                    pmid = COALESCE(EXCLUDED.pmid, saved_studies.pmid)
                RETURNING id, study_id, title, doi, pmid, created_at
            """, user_id, study_id, title, doi, pmid)
            
            return {
                "id": row["id"],
                "study_id": row["study_id"],
                "title": row["title"],
                "doi": row["doi"],
                "pmid": row["pmid"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
    
    async def get_user_studies(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all saved studies for a user.
        
        Args:
            user_id: User's UUID
            limit: Maximum number of studies to return
            
        Returns:
            List of saved studies
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, study_id, title, doi, pmid, created_at
                FROM saved_studies
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, user_id, limit)
            
            return [
                {
                    "id": row["id"],
                    "study_id": row["study_id"],
                    "title": row["title"],
                    "doi": row["doi"],
                    "pmid": row["pmid"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in rows
            ]
    
    async def is_study_saved(self, user_id: str, study_id: str) -> bool:
        """Check if a study is saved by a user."""
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 1 FROM saved_studies
                WHERE user_id = $1 AND study_id = $2
            """, user_id, study_id)
            
            return row is not None
    
    async def delete_study(self, user_id: str, study_id: str) -> bool:
        """
        Delete a saved study.

        Args:
            user_id: User's UUID
            study_id: Study identifier

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_schema()

        db = get_account_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM saved_studies
                WHERE user_id = $1 AND study_id = $2
            """, user_id, study_id)

            return "DELETE 1" in result

    # ------------------------------------------------------------------
    # Patient-scoped extension (patient-centric pivot)
    # ------------------------------------------------------------------
    #
    # Additive only — existing columns/constraints/methods above are
    # untouched. The original UNIQUE(user_id, study_id) constraint means a
    # physician saving the same study for two different patients would
    # collide if routed through save_study()/ON CONFLICT above, so
    # patient-scoped rows go through a separate manual upsert here instead
    # of reusing that constraint. patient_id is nullable and NULL for every
    # pre-existing row, so none of the methods above change behavior.

    async def _ensure_patient_columns(self):
        db = get_account_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                ALTER TABLE saved_studies ADD COLUMN IF NOT EXISTS patient_id UUID;
                ALTER TABLE saved_studies ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'manual';
                ALTER TABLE saved_studies ADD COLUMN IF NOT EXISTS auto_seeded BOOLEAN DEFAULT FALSE;

                CREATE INDEX IF NOT EXISTS idx_saved_studies_patient_id
                ON saved_studies(patient_id) WHERE patient_id IS NOT NULL;
            """)

    async def save_study_for_patient(
        self,
        patient_id: str,
        user_id: str,
        study_id: str,
        title: Optional[str] = None,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
        source: str = "core",
        auto_seeded: bool = True,
    ) -> Dict[str, Any]:
        """
        Save (or refresh) a study for a specific patient. Source is one of
        'core' (main KB), 'personal' (physician's uploaded library), or
        'web'. auto_seeded distinguishes matches from patient_collection_seeder
        vs. studies a physician manually attached to the patient.

        Args:
            patient_id: Patient's UUID (from the exueed-patients database;
                no DB-level FK since it's a different database)
            user_id: Physician's UUID — who this save is attributed to
            study_id: Unique study identifier (doc_id, pmid, or doi)
            title, doi, pmid: Study metadata
            source: 'core' | 'personal' | 'web'
            auto_seeded: True if written by the seeder, False if manually attached

        Returns:
            The saved study record
        """
        await self._ensure_schema()
        await self._ensure_patient_columns()

        db = get_account_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT id FROM saved_studies
                WHERE patient_id = $1 AND study_id = $2
                """,
                patient_id, study_id,
            )
            if existing:
                row = await conn.fetchrow(
                    """
                    UPDATE saved_studies
                    SET title = COALESCE($3, title),
                        doi = COALESCE($4, doi),
                        pmid = COALESCE($5, pmid),
                        source = $6,
                        auto_seeded = $7
                    WHERE patient_id = $1 AND study_id = $2
                    RETURNING id, patient_id, study_id, title, doi, pmid,
                              source, auto_seeded, created_at
                    """,
                    patient_id, study_id, title, doi, pmid, source, auto_seeded,
                )
            else:
                row = await conn.fetchrow(
                    """
                    INSERT INTO saved_studies
                        (user_id, patient_id, study_id, title, doi, pmid, source, auto_seeded)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id, patient_id, study_id, title, doi, pmid,
                              source, auto_seeded, created_at
                    """,
                    user_id, patient_id, study_id, title, doi, pmid, source, auto_seeded,
                )

            return {
                "id": row["id"],
                "patient_id": str(row["patient_id"]) if row["patient_id"] else None,
                "study_id": row["study_id"],
                "title": row["title"],
                "doi": row["doi"],
                "pmid": row["pmid"],
                "source": row["source"],
                "auto_seeded": row["auto_seeded"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }

    async def get_patient_studies(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all studies (auto-seeded + manually attached) for a patient."""
        await self._ensure_schema()
        await self._ensure_patient_columns()

        db = get_account_db()
        pool = await db.get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, patient_id, study_id, title, doi, pmid,
                       source, auto_seeded, created_at
                FROM saved_studies
                WHERE patient_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                patient_id, limit,
            )
            return [
                {
                    "id": row["id"],
                    "patient_id": str(row["patient_id"]) if row["patient_id"] else None,
                    "study_id": row["study_id"],
                    "title": row["title"],
                    "doi": row["doi"],
                    "pmid": row["pmid"],
                    "source": row["source"],
                    "auto_seeded": row["auto_seeded"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in rows
            ]


# Singleton instance
_saved_studies_service: Optional[SavedStudiesService] = None


def get_saved_studies_service() -> SavedStudiesService:
    """Get or create the saved studies service singleton"""
    global _saved_studies_service
    if _saved_studies_service is None:
        _saved_studies_service = SavedStudiesService()
    return _saved_studies_service
