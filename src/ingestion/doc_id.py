"""Shared doc_id normalization.

Qdrant payloads, the Postgres `studies.doc_id` column, and any audit tool that
cross-checks the two must all produce the same id from the same source
directory name. Keep this the single source of truth.
"""

from __future__ import annotations

import hashlib
import re


def normalize_doc_id(raw_name: str) -> str:
    """Create a filesystem-safe, collision-resistant doc_id.

    Non-word chars collapse to `_`, repeated `_` are squashed, the result is
    truncated to 50 chars, and an md5[:8] of the *original* name is appended
    so two dirs that normalize to the same prefix still get distinct ids.
    """
    raw_name = raw_name or "unknown"
    clean = re.sub(r"[^\w\-.]", "_", raw_name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    clean = clean[:50]
    h = hashlib.md5(raw_name.encode("utf-8")).hexdigest()[:8]
    return f"{clean}_{h}"
