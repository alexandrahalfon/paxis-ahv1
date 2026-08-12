"""
Saved Cases Service

Stores and retrieves classified patient queries (cases) for users.
Allows users to save complex clinical cases and reuse them for future queries.
Also supports saving full query responses with sources for reference.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .account_db import get_account_db
from .query_classifier_service import StructuredQuery


class SavedCasesService:
    """Service for managing saved patient cases per user"""
    
    async def _ensure_schema(self):
        """Ensure the saved_cases and case_alerts tables exist"""
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Main saved_cases table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS saved_cases (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    
                    -- Case identification
                    case_name VARCHAR(255),
                    original_query TEXT NOT NULL,
                    query_summary VARCHAR(500),
                    
                    -- Structured data (stored as JSONB for flexibility)
                    structured_data JSONB NOT NULL,
                    
                    -- Categorized fields for quick display
                    demographics JSONB,
                    diagnosis JSONB,
                    staging JSONB,
                    pathology JSONB,
                    treatment_history JSONB,
                    risk_factors JSONB,
                    
                    -- Full response data (for saved searches)
                    response_answer TEXT,
                    response_sources JSONB,
                    response_metadata JSONB,
                    
                    -- Usage tracking
                    use_count INTEGER DEFAULT 0,
                    last_used_at TIMESTAMPTZ,
                    
                    -- Metadata
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- Soft delete
                    is_archived BOOLEAN DEFAULT FALSE
                );
                
                CREATE INDEX IF NOT EXISTS idx_saved_cases_user_id 
                ON saved_cases(user_id);
                
                CREATE INDEX IF NOT EXISTS idx_saved_cases_user_created 
                ON saved_cases(user_id, created_at DESC);
            """)
            
            # Add columns if they don't exist (for existing tables)
            await conn.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name = 'saved_cases' AND column_name = 'response_answer') THEN
                        ALTER TABLE saved_cases ADD COLUMN response_answer TEXT;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name = 'saved_cases' AND column_name = 'response_sources') THEN
                        ALTER TABLE saved_cases ADD COLUMN response_sources JSONB;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name = 'saved_cases' AND column_name = 'response_metadata') THEN
                        ALTER TABLE saved_cases ADD COLUMN response_metadata JSONB;
                    END IF;
                END $$;
            """)
            
            # Case alerts table for new trial notifications
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS case_alerts (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    case_id INTEGER NOT NULL REFERENCES saved_cases(id) ON DELETE CASCADE,
                    
                    -- Alert settings
                    alerts_enabled BOOLEAN DEFAULT TRUE,
                    email_notifications BOOLEAN DEFAULT TRUE,
                    
                    -- Search criteria for matching new trials
                    search_criteria JSONB NOT NULL,
                    
                    -- Tracking
                    last_checked_at TIMESTAMPTZ,
                    last_alert_sent_at TIMESTAMPTZ,
                    new_matches_count INTEGER DEFAULT 0,
                    
                    -- Metadata
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    UNIQUE(user_id, case_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_case_alerts_user_id 
                ON case_alerts(user_id);
                
                CREATE INDEX IF NOT EXISTS idx_case_alerts_enabled 
                ON case_alerts(alerts_enabled) WHERE alerts_enabled = TRUE;
            """)
            
            # Alert matches table to track which trials have been notified
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_matches (
                    id SERIAL PRIMARY KEY,
                    alert_id INTEGER NOT NULL REFERENCES case_alerts(id) ON DELETE CASCADE,
                    
                    -- Matched trial info
                    trial_doc_id VARCHAR(255),
                    trial_title TEXT,
                    trial_doi VARCHAR(255),
                    match_score FLOAT,
                    match_reasons JSONB,
                    
                    -- Status
                    notified_at TIMESTAMPTZ,
                    viewed_at TIMESTAMPTZ,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    UNIQUE(alert_id, trial_doc_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_alert_matches_alert_id 
                ON alert_matches(alert_id);
            """)
    
    async def save_case(
        self,
        user_id: str,
        original_query: str,
        structured_query: StructuredQuery,
        case_name: Optional[str] = None,
        demographics: Optional[Dict] = None,
        diagnosis: Optional[Dict] = None,
        staging: Optional[Dict] = None,
        pathology: Optional[Dict] = None,
        treatment_history: Optional[Dict] = None,
        risk_factors: Optional[Dict] = None,
        response_answer: Optional[str] = None,
        response_sources: Optional[List[Dict]] = None,
        response_metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Save a classified patient case with optional full response.
        
        Args:
            user_id: User's ID
            original_query: The original free-text query
            structured_query: The StructuredQuery object with extracted fields
            case_name: Optional friendly name for the case
            demographics: Categorized demographics data
            diagnosis: Categorized diagnosis data
            staging: Categorized staging data
            pathology: Categorized pathology data
            treatment_history: Categorized treatment history
            risk_factors: Categorized risk factors
            response_answer: The full AI response text
            response_sources: List of source citations
            response_metadata: Additional response metadata (matching trials, etc.)
            
        Returns:
            The saved case record
        """
        print(f"[SavedCases] Saving case for user {user_id}")
        print(f"[SavedCases] response_answer length: {len(response_answer) if response_answer else 0}")
        print(f"[SavedCases] response_sources count: {len(response_sources) if response_sources else 0}")
        print(f"[SavedCases] response_metadata: {response_metadata}")
        
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        # Generate case name if not provided
        if not case_name:
            case_name = self._generate_case_name(structured_query)
        
        # Get query summary
        query_summary = structured_query.get_search_summary()
        
        # Convert structured query to dict
        structured_data = structured_query.to_dict()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO saved_cases (
                    user_id, case_name, original_query, query_summary,
                    structured_data, demographics, diagnosis, staging,
                    pathology, treatment_history, risk_factors,
                    response_answer, response_sources, response_metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                RETURNING id, case_name, query_summary, created_at
            """,
                user_id,
                case_name,
                original_query,
                query_summary,
                json.dumps(structured_data),
                json.dumps(demographics) if demographics else None,
                json.dumps(diagnosis) if diagnosis else None,
                json.dumps(staging) if staging else None,
                json.dumps(pathology) if pathology else None,
                json.dumps(treatment_history) if treatment_history else None,
                json.dumps(risk_factors) if risk_factors else None,
                response_answer,
                json.dumps(response_sources) if response_sources else None,
                json.dumps(response_metadata) if response_metadata else None,
            )
            
            return {
                "id": row["id"],
                "case_name": row["case_name"],
                "query_summary": row["query_summary"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
    
    def _generate_case_name(self, query: StructuredQuery) -> str:
        """Generate a descriptive case name from the structured query"""
        parts = []
        
        if query.age:
            parts.append(f"{query.age}yo")
        if query.gender:
            parts.append(query.gender[0].upper())  # M or F
        if query.cancer_type:
            # Abbreviate common cancer types
            ct = query.cancer_type
            if "squamous" in ct.lower():
                parts.append("SCC")
            elif "adenocarcinoma" in ct.lower():
                parts.append("Adeno")
            else:
                parts.append(ct[:15])
        if query.cancer_location:
            parts.append(query.cancer_location[:15])
        if query.tnm_t or query.tnm_n:
            tnm = ""
            if query.tnm_t:
                tnm += query.tnm_t
            if query.tnm_n:
                tnm += query.tnm_n
            parts.append(tnm)
        
        if not parts:
            return f"Case {datetime.now().strftime('%Y%m%d_%H%M')}"
        
        return " ".join(parts)
    
    async def get_user_cases(
        self,
        user_id: str,
        limit: int = 20,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all saved cases for a user.
        
        Args:
            user_id: User's ID
            limit: Maximum number of cases to return
            include_archived: Whether to include archived cases
            
        Returns:
            List of saved cases
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            archive_filter = "" if include_archived else "AND is_archived = FALSE"
            
            rows = await conn.fetch(f"""
                SELECT 
                    id, case_name, original_query, query_summary,
                    structured_data, demographics, diagnosis, staging,
                    pathology, treatment_history, risk_factors,
                    response_answer, response_sources, response_metadata,
                    use_count, last_used_at, created_at, is_archived
                FROM saved_cases
                WHERE user_id = $1 {archive_filter}
                ORDER BY 
                    CASE WHEN last_used_at IS NOT NULL THEN last_used_at ELSE created_at END DESC
                LIMIT $2
            """, user_id, limit)
            
            cases = []
            for row in rows:
                case = {
                    "id": row["id"],
                    "case_name": row["case_name"],
                    "original_query": row["original_query"],
                    "query_summary": row["query_summary"],
                    "structured_data": row["structured_data"] if isinstance(row["structured_data"], dict) else json.loads(row["structured_data"]) if row["structured_data"] else {},
                    "demographics": row["demographics"] if isinstance(row["demographics"], dict) else json.loads(row["demographics"]) if row["demographics"] else {},
                    "diagnosis": row["diagnosis"] if isinstance(row["diagnosis"], dict) else json.loads(row["diagnosis"]) if row["diagnosis"] else {},
                    "staging": row["staging"] if isinstance(row["staging"], dict) else json.loads(row["staging"]) if row["staging"] else {},
                    "pathology": row["pathology"] if isinstance(row["pathology"], dict) else json.loads(row["pathology"]) if row["pathology"] else {},
                    "treatment_history": row["treatment_history"] if isinstance(row["treatment_history"], dict) else json.loads(row["treatment_history"]) if row["treatment_history"] else {},
                    "risk_factors": row["risk_factors"] if isinstance(row["risk_factors"], dict) else json.loads(row["risk_factors"]) if row["risk_factors"] else {},
                    "response_answer": row["response_answer"],
                    "response_sources": row["response_sources"] if isinstance(row["response_sources"], list) else json.loads(row["response_sources"]) if row["response_sources"] else None,
                    "response_metadata": row["response_metadata"] if isinstance(row["response_metadata"], dict) else json.loads(row["response_metadata"]) if row["response_metadata"] else None,
                    "use_count": row["use_count"],
                    "last_used_at": row["last_used_at"].isoformat() if row["last_used_at"] else None,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "is_archived": row["is_archived"],
                }
                cases.append(case)
            
            return cases
    
    async def get_case(self, user_id: str, case_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific saved case.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            
        Returns:
            The case record or None if not found
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    id, case_name, original_query, query_summary,
                    structured_data, demographics, diagnosis, staging,
                    pathology, treatment_history, risk_factors,
                    response_answer, response_sources, response_metadata,
                    use_count, last_used_at, created_at, is_archived
                FROM saved_cases
                WHERE id = $1 AND user_id = $2
            """, case_id, user_id)
            
            if not row:
                return None
            
            return {
                "id": row["id"],
                "case_name": row["case_name"],
                "original_query": row["original_query"],
                "query_summary": row["query_summary"],
                "structured_data": row["structured_data"] if isinstance(row["structured_data"], dict) else json.loads(row["structured_data"]) if row["structured_data"] else {},
                "demographics": row["demographics"] if isinstance(row["demographics"], dict) else json.loads(row["demographics"]) if row["demographics"] else {},
                "diagnosis": row["diagnosis"] if isinstance(row["diagnosis"], dict) else json.loads(row["diagnosis"]) if row["diagnosis"] else {},
                "staging": row["staging"] if isinstance(row["staging"], dict) else json.loads(row["staging"]) if row["staging"] else {},
                "pathology": row["pathology"] if isinstance(row["pathology"], dict) else json.loads(row["pathology"]) if row["pathology"] else {},
                "treatment_history": row["treatment_history"] if isinstance(row["treatment_history"], dict) else json.loads(row["treatment_history"]) if row["treatment_history"] else {},
                "risk_factors": row["risk_factors"] if isinstance(row["risk_factors"], dict) else json.loads(row["risk_factors"]) if row["risk_factors"] else {},
                "response_answer": row["response_answer"],
                "response_sources": row["response_sources"] if isinstance(row["response_sources"], list) else json.loads(row["response_sources"]) if row["response_sources"] else None,
                "response_metadata": row["response_metadata"] if isinstance(row["response_metadata"], dict) else json.loads(row["response_metadata"]) if row["response_metadata"] else None,
                "use_count": row["use_count"],
                "last_used_at": row["last_used_at"].isoformat() if row["last_used_at"] else None,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "is_archived": row["is_archived"],
            }
    
    async def use_case(self, user_id: str, case_id: int) -> Optional[Dict[str, Any]]:
        """
        Mark a case as used and return it.
        Increments use_count and updates last_used_at.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            
        Returns:
            The case record with updated usage stats
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Update usage stats
            await conn.execute("""
                UPDATE saved_cases
                SET use_count = use_count + 1,
                    last_used_at = NOW()
                WHERE id = $1 AND user_id = $2
            """, case_id, user_id)
        
        # Return the updated case
        return await self.get_case(user_id, case_id)
    
    async def update_case(
        self,
        user_id: str,
        case_id: int,
        case_name: Optional[str] = None,
        is_archived: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update a saved case.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            case_name: New case name (optional)
            is_archived: Archive status (optional)
            
        Returns:
            The updated case record
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        updates = []
        params = []
        param_idx = 1
        
        if case_name is not None:
            updates.append(f"case_name = ${param_idx}")
            params.append(case_name)
            param_idx += 1
        
        if is_archived is not None:
            updates.append(f"is_archived = ${param_idx}")
            params.append(is_archived)
            param_idx += 1
        
        if not updates:
            return await self.get_case(user_id, case_id)
        
        updates.append("updated_at = NOW()")
        
        async with pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE saved_cases
                SET {", ".join(updates)}
                WHERE id = ${param_idx} AND user_id = ${param_idx + 1}
            """, *params, case_id, user_id)
        
        return await self.get_case(user_id, case_id)
    
    async def delete_case(self, user_id: str, case_id: int) -> bool:
        """
        Permanently delete a saved case.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            
        Returns:
            True if deleted, False if not found
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM saved_cases
                WHERE id = $1 AND user_id = $2
            """, case_id, user_id)
            
            return "DELETE 1" in result
    
    # ============================================
    # Alert Management Methods
    # ============================================
    
    async def enable_alerts(
        self,
        user_id: str,
        case_id: int,
        email_notifications: bool = True
    ) -> Dict[str, Any]:
        """
        Enable alerts for a saved case.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            email_notifications: Whether to send email notifications
            
        Returns:
            The alert settings
        """
        await self._ensure_schema()
        
        # First verify the case belongs to the user
        case = await self.get_case(user_id, case_id)
        if not case:
            raise ValueError("Case not found")
        
        db = get_account_db()
        pool = await db.get_pool()
        
        # Build search criteria from the case's structured data
        search_criteria = {
            "original_query": case.get("original_query"),
            "structured_data": case.get("structured_data", {}),
            "demographics": case.get("demographics", {}),
            "diagnosis": case.get("diagnosis", {}),
            "staging": case.get("staging", {}),
        }
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO case_alerts (
                    user_id, case_id, alerts_enabled, email_notifications, search_criteria
                ) VALUES ($1, $2, TRUE, $3, $4)
                ON CONFLICT (user_id, case_id) DO UPDATE SET
                    alerts_enabled = TRUE,
                    email_notifications = $3,
                    search_criteria = $4,
                    updated_at = NOW()
                RETURNING id, alerts_enabled, email_notifications, created_at
            """, user_id, case_id, email_notifications, json.dumps(search_criteria))
            
            return {
                "alert_id": row["id"],
                "case_id": case_id,
                "alerts_enabled": row["alerts_enabled"],
                "email_notifications": row["email_notifications"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
    
    async def disable_alerts(self, user_id: str, case_id: int) -> bool:
        """
        Disable alerts for a saved case.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            
        Returns:
            True if disabled, False if not found
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE case_alerts
                SET alerts_enabled = FALSE, updated_at = NOW()
                WHERE user_id = $1 AND case_id = $2
            """, user_id, case_id)
            
            return "UPDATE 1" in result
    
    async def get_alert_settings(self, user_id: str, case_id: int) -> Optional[Dict[str, Any]]:
        """
        Get alert settings for a case.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            
        Returns:
            Alert settings or None if not found
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, alerts_enabled, email_notifications, 
                       last_checked_at, last_alert_sent_at, new_matches_count,
                       created_at, updated_at
                FROM case_alerts
                WHERE user_id = $1 AND case_id = $2
            """, user_id, case_id)
            
            if not row:
                return None
            
            return {
                "alert_id": row["id"],
                "case_id": case_id,
                "alerts_enabled": row["alerts_enabled"],
                "email_notifications": row["email_notifications"],
                "last_checked_at": row["last_checked_at"].isoformat() if row["last_checked_at"] else None,
                "last_alert_sent_at": row["last_alert_sent_at"].isoformat() if row["last_alert_sent_at"] else None,
                "new_matches_count": row["new_matches_count"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
    
    async def get_user_alerts(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all alert settings for a user.
        
        Args:
            user_id: User's ID
            
        Returns:
            List of alert settings with case info
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    ca.id as alert_id, ca.case_id, ca.alerts_enabled, 
                    ca.email_notifications, ca.new_matches_count,
                    ca.last_checked_at, ca.last_alert_sent_at,
                    sc.case_name, sc.query_summary
                FROM case_alerts ca
                JOIN saved_cases sc ON ca.case_id = sc.id
                WHERE ca.user_id = $1
                ORDER BY ca.created_at DESC
            """, user_id)
            
            return [{
                "alert_id": row["alert_id"],
                "case_id": row["case_id"],
                "case_name": row["case_name"],
                "query_summary": row["query_summary"],
                "alerts_enabled": row["alerts_enabled"],
                "email_notifications": row["email_notifications"],
                "new_matches_count": row["new_matches_count"],
                "last_checked_at": row["last_checked_at"].isoformat() if row["last_checked_at"] else None,
                "last_alert_sent_at": row["last_alert_sent_at"].isoformat() if row["last_alert_sent_at"] else None,
            } for row in rows]
    
    async def get_new_matches(self, user_id: str, case_id: int) -> List[Dict[str, Any]]:
        """
        Get new trial matches for a case alert.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            
        Returns:
            List of new matching trials
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # First get the alert_id
            alert = await conn.fetchrow("""
                SELECT id FROM case_alerts
                WHERE user_id = $1 AND case_id = $2
            """, user_id, case_id)
            
            if not alert:
                return []
            
            rows = await conn.fetch("""
                SELECT trial_doc_id, trial_title, trial_doi, 
                       match_score, match_reasons, created_at, viewed_at
                FROM alert_matches
                WHERE alert_id = $1
                ORDER BY created_at DESC
                LIMIT 50
            """, alert["id"])
            
            return [{
                "doc_id": row["trial_doc_id"],
                "title": row["trial_title"],
                "doi": row["trial_doi"],
                "match_score": row["match_score"],
                "match_reasons": row["match_reasons"] if isinstance(row["match_reasons"], list) else json.loads(row["match_reasons"]) if row["match_reasons"] else [],
                "found_at": row["created_at"].isoformat() if row["created_at"] else None,
                "viewed": row["viewed_at"] is not None,
            } for row in rows]
    
    async def mark_matches_viewed(self, user_id: str, case_id: int) -> bool:
        """
        Mark all new matches as viewed for a case.
        
        Args:
            user_id: User's ID
            case_id: Case ID
            
        Returns:
            True if updated
        """
        await self._ensure_schema()
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Get alert_id and verify ownership
            alert = await conn.fetchrow("""
                SELECT id FROM case_alerts
                WHERE user_id = $1 AND case_id = $2
            """, user_id, case_id)
            
            if not alert:
                return False
            
            # Mark all unviewed matches as viewed
            await conn.execute("""
                UPDATE alert_matches
                SET viewed_at = NOW()
                WHERE alert_id = $1 AND viewed_at IS NULL
            """, alert["id"])
            
            # Reset the new_matches_count
            await conn.execute("""
                UPDATE case_alerts
                SET new_matches_count = 0, updated_at = NOW()
                WHERE id = $1
            """, alert["id"])
            
            return True


# Singleton instance
_saved_cases_service: Optional[SavedCasesService] = None


def get_saved_cases_service() -> SavedCasesService:
    """Get or create the saved cases service singleton"""
    global _saved_cases_service
    if _saved_cases_service is None:
        _saved_cases_service = SavedCasesService()
    return _saved_cases_service
