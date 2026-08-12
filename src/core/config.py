"""
Configuration management for the document ingestion pipeline.
"""

import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        extra='ignore',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    ) 
    
    # API Keys
    openai_api_key: str = ""
    mistral_api_key: str = ""
    qdrant_api_key: str = ""
    
    # Model Configuration
    openai_model: str = "gpt-4o"
    openai_mini_model: str = "gpt-4o-mini"  # Lighter model for intermediate/supplementary services
    study_profile_model: str = "gpt-4o-mini"  # Cheaper model for study profile extraction
    mistral_model: str = "pixtral-large-latest"
    mistral_ocr_model: str = "mistral-ocr-latest"
    
    # Embedding Configuration
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 3072
    max_tokens_per_chunk: int = 600
    
    # Qdrant Configuration
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "exueed_kb_latest"

        # PostgreSQL Configuration
    use_postgres_for_study_details: bool = True
    postgres_host: str = "34.21.60.224"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""  # From .env
    postgres_database: str = "display-study-details"

    # Accounts + Cache PostgreSQL Configuration (separate database)
    # host/port/user default empty and fall back to postgres_* at connection
    # time (see account_db.py get_pool()) — same server, same credentials,
    # by default. Previously these were hardcoded to a stale IP, which
    # silently broke auth whenever postgres_host changed without a matching
    # CACHE_POSTGRES_HOST override.
    cache_postgres_host: str = ""  # From .env; falls back to postgres_host if unset
    cache_postgres_port: int = 0  # From .env; falls back to postgres_port if unset
    cache_postgres_user: str = ""  # From .env; falls back to postgres_user if unset
    cache_postgres_password: str = ""  # From .env
    cache_postgres_database: str = "exueed_cache"
    cache_postgres_min_pool: int = 1
    cache_postgres_max_pool: int = 5
    cache_ttl_seconds: int = 86400
    cache_hot_max_entries: int = 200

    # Patients PostgreSQL Configuration (separate database, same GCP instance)
    # Created 2026-07-08 for the patient-centric pivot. Host/port/user default
    # empty and fall back to postgres_* at connection time (see
    # PatientDatabase.get_pool()) since all three DBs share one server by
    # default. Previously host/port/user were hardcoded to a stale IP, which
    # silently broke auth whenever postgres_host changed without a matching
    # PATIENTS_POSTGRES_HOST override — same bug class as cache_postgres_*.
    patients_postgres_host: str = ""  # From .env; falls back to postgres_host if unset
    patients_postgres_port: int = 0  # From .env; falls back to postgres_port if unset
    patients_postgres_user: str = ""  # From .env; falls back to postgres_user if unset
    patients_postgres_password: str = ""  # From .env; falls back to postgres_password if unset
    patients_postgres_database: str = "exueed-patients"
    patients_postgres_min_pool: int = 1
    patients_postgres_max_pool: int = 5

    # Clinician registration gating (added 2026-08-08 with the patient
    # portal). Clinicians appear in a patient-facing directory, so open
    # registration would let anyone present themselves as a doctor.
    # 'blocklist' rejects consumer/disposable providers, 'allowlist'
    # permits only clinician_email_allowlist, 'off' disables the check.
    # This is not identity verification: see clinician_domains.py.
    clinician_email_mode: str = "blocklist"
    clinician_email_allowlist: str = ""  # comma-separated, allowlist mode only

    # Auth configuration
    auth_secret_key: str = "change-me"
    auth_algorithm: str = "HS256"
    auth_access_token_expire_minutes: int = 1440
    
    
    # GCP Configuration
    gcp_bucket_name: str = ""
    gcp_user_uploads_bucket: str = ""  # Separate bucket for user uploads
    gcp_project_id: str = ""
    auto_sync_gcp: bool = False
    
    # Processing Configuration
    output_dir: str = "processed_documents"
    cache_dir: str = "cache"
    temp_dir: str = "temp"
    
    # Batch Sizes
    embed_batch_size: int = 64
    qdrant_batch_size: int = 512
    
    # Feature Flags
    enable_pixtral: bool = True
    enable_figure_extraction: bool = True
    enable_table_extraction: bool = True
    extract_study_profiles: bool = False  # Enable study profile extraction in pipeline

    # RAG Pipeline Consolidation Feature Flags (Phase 10)
    enable_canonicalization: bool = False
    enable_hard_gate: bool = False
    enable_soft_scorer: bool = False
    enable_pto_retrieval: bool = False
    enable_query_decomposition: bool = False
    enable_relaxed_numval: bool = False
    enable_perf_optimizations: bool = False

    @property
    def use_reconciled_structure(self) -> bool:
        """Read USE_RECONCILED_STRUCTURE from os.environ dynamically (no restart required)."""
        return os.environ.get("USE_RECONCILED_STRUCTURE", "false").lower() in ("true", "1", "yes")
   


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


def validate_settings() -> bool:
    """
    Validate that required settings are configured.
    
    Returns:
        bool: True if all required settings are present
        
    Raises:
        ValueError: If required settings are missing
    """
    settings = get_settings()
    
    required_keys = {
        "openai_api_key": "OPENAI_API_KEY",
        "mistral_api_key": "MISTRAL_API_KEY",
        "qdrant_url": "QDRANT_URL",
    }
    
    missing = []
    for attr, env_var in required_keys.items():
        if not getattr(settings, attr):
            missing.append(env_var)
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please set them in your .env file or environment."
        )
    
    return True


def ensure_directories():
    """Create required directories if they don't exist."""
    settings = get_settings()
    
    directories = [
        settings.output_dir,
        settings.cache_dir,
        settings.temp_dir,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
# Create module-level settings instance for direct import
settings = get_settings()
