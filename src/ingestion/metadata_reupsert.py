#!/usr/bin/env python3
"""
Metadata Re-Upsert Script

Re-scans all chunks in Qdrant against updated ontology/keyword files and
fixes section names — WITHOUT re-generating embeddings. Only the payload
(metadata) is updated; vectors remain untouched.

What it does:
  1. Scrolls through all points in the Qdrant collection
  2. For each chunk, re-scans the text against:
     - data/keywords/extractor_keywords.json
     - data/ontology/cancer_type_ontology.json
     - data/ontology/clinical_trial_ontology.json
  3. Normalizes section names using REAL section names from the
     _structured_content.json files (when --processed-docs-dir is
     provided), with text-heuristic fallback
  4. Adds a `section_type` classification field
  5. Marks abstract / first-paragraph chunks with `is_abstract: true`
  6. Updates Qdrant payloads via set_payload (no re-embedding)

Usage (Colab with processed_documents on Google Drive):
    python src/ingestion/metadata_reupsert.py \
        --qdrant-url YOUR_URL \
        --qdrant-api-key YOUR_KEY \
        --collection-name YOUR_COLLECTION \
        --processed-docs-dir /content/drive/MyDrive/processed_documents_complete \
        --batch-size 100 \
        --dry-run

Usage (without processed docs — text-heuristic section names only):
    python src/ingestion/metadata_reupsert.py \
        --qdrant-url YOUR_URL \
        --qdrant-api-key YOUR_KEY \
        --collection-name YOUR_COLLECTION \
        --batch-size 100

Cost: $0 (no OpenAI API calls)
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── Locate data files ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.name == "ingestion" else SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD KEYWORD / ONTOLOGY FILES
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_keyword_scanner(data_dir: Path):
    """
    Build the canonical KeywordTagger (ambiguity-aware, covers biomarkers,
    drugs, genomic alterations, AJCC staging, related_diseases,
    related_pathways). Returns the KeywordTagger instance.

    The return value is opaque — pass it directly to scan_text_for_keywords().
    """
    # Ensure src.ingestion is importable when invoked as a standalone script
    import sys as _sys
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    from src.ingestion.keyword_tagger import KeywordTagger
    return KeywordTagger(data_dir=data_dir)


def scan_text_for_keywords(
    text: str,
    tagger,
    _unused=None,
) -> Tuple[Dict[str, List[str]], List[str], List[str], List[str]]:
    """
    Scan text via the canonical KeywordTagger. Layers in staging/range
    expansion from query_expansion.

    Returns:
        (keyword_matches, keywords_flat, ontology_matches, expanded_synonyms)

    Note: dedicated slots (biomarkers, drugs, alterations, cancer_types) are
    also surfaced — call scan_text_for_keywords_detailed() if you need them.
    """
    detailed = scan_text_for_keywords_detailed(text, tagger)
    return (
        detailed["keyword_matches"],
        detailed["keywords_flat"],
        detailed["ontology_tags"],
        detailed["expanded_synonyms"],
    )


def scan_text_for_keywords_detailed(
    text: str,
    tagger,
) -> Dict[str, Any]:
    """Full-detail scan with staging/range expansion layered on top."""
    from src.ingestion.keyword_tagger import KeywordTagger

    if not isinstance(tagger, KeywordTagger):
        # Legacy callers passing (term_to_categories, ontology_tags) tuples
        # — build a fresh tagger from DATA_DIR.
        tagger = build_keyword_scanner(DATA_DIR)

    detailed = tagger.scan_text_detailed(text)
    flat_terms = detailed["keywords_flat"]
    expanded_terms: List[str] = list(detailed["expanded_synonyms"])

    try:
        import sys as _sys
        _project_root = str(Path(__file__).resolve().parent.parent.parent)
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)

        from src.api.services.query_expansion import (
            _build_staging_expansions,
            _build_range_expansions,
        )
        detected_lower = {t.lower() for t in flat_terms} | {t.lower() for t in expanded_terms}
        for t in (_build_staging_expansions(text) | _build_range_expansions(text)):
            if t.lower() not in detected_lower:
                expanded_terms.append(t)
                detected_lower.add(t.lower())
    except Exception as _exp_err:
        if not hasattr(scan_text_for_keywords, '_error_warned'):
            print(f"    [expansion] Error (this message shown once): {_exp_err}")
            scan_text_for_keywords._error_warned = True

    return {
        "keyword_matches": detailed["keyword_matches"],
        "keywords_flat": flat_terms,
        "ontology_tags": detailed["ontology_tags"],
        "expanded_synonyms": sorted(set(expanded_terms)),
        "cancer_types_detected": detailed["cancer_types_detected"],
        "biomarkers_detected": detailed["biomarkers_detected"],
        "drugs_detected": detailed["drugs_detected"],
        "genomic_alterations": detailed["genomic_alterations"],
        "ajcc_tags": detailed["ajcc_tags"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. SECTION NAME NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# Canonical section types for the section_type field
SECTION_TYPES = {
    "abstract", "introduction", "background", "methods", "results",
    "discussion", "conclusion", "eligibility", "references", "table",
    "acknowledgments", "supplementary", "other",
}

# Patterns to classify section text into section_type
_SECTION_PATTERNS: List[Tuple[str, str]] = [
    (r'\babstract\b', "abstract"),
    (r'\bbackground\b', "background"),
    (r'\bintroduction\b', "introduction"),
    (r'\bmethods?\b|\bmaterials?\s+and\s+methods?\b|\bstudy\s+design\b|\bpatients?\s+and\s+methods?\b', "methods"),
    (r'\bresults?\b|\boutcomes?\b|\bfindings?\b', "results"),
    (r'\bdiscussion\b', "discussion"),
    (r'\bconclusion\b', "conclusion"),
    (r'\beligib\w+\b|\binclusion\s+criteria\b|\bexclusion\s+criteria\b', "eligibility"),
    (r'\breferences?\b|\bbibliography\b', "references"),
    (r'\backnowledg\w+\b|\bfunding\b|\bdisclosures?\b|\bconflict\b', "acknowledgments"),
    (r'\bsupplement\w+\b|\bappendix\b', "supplementary"),
]

# Mapping from raw section names (including pixtral artifacts) to clean names
_SECTION_NAME_CLEANUP = {
    "pixtral_content": None,          # needs text-based inference
    "pixtral_pixtral_content": None,  # needs text-based inference
    "pixtral": None,
    "no_section": None,
}


def normalize_section_name(raw_section: str, chunk_text: str) -> Tuple[str, str]:
    """
    Normalize a section name and classify its type.

    Args:
        raw_section: The raw section name from the processed document
        chunk_text: The chunk's text content (used for heuristic classification
                    when the section name is uninformative)

    Returns:
        (cleaned_section_name, section_type)
    """
    raw = (raw_section or "").strip()
    raw_lower = raw.lower().replace("_", " ").strip()

    # ── Already a clean section name? ────────────────────────────────
    for pattern, stype in _SECTION_PATTERNS:
        if re.search(pattern, raw_lower):
            # Clean up the name but keep it
            cleaned = re.sub(r'pixtral[_\s]*:?\s*', '', raw, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'^[\s_-]+|[\s_-]+$', '', cleaned)
            if not cleaned:
                cleaned = stype.capitalize()
            return cleaned, stype

    # ── Known bad names (pixtral artifacts) → infer from text ────────
    is_bad_name = (
        not raw
        or "pixtral" in raw_lower
        or raw_lower in ("no section", "no_section", "body", "content")
        or len(raw_lower) < 3
    )

    if is_bad_name:
        # Infer section type from the text content
        text_lower = (chunk_text or "")[:500].lower()

        # Check for abstract indicators (first paragraph, specific phrasing)
        if any(kw in text_lower for kw in [
            "background:", "purpose:", "objective:", "aim:",
            "we aimed", "this study aimed", "the aim of this",
            "we conducted", "we performed", "we evaluated",
            "in this study", "in this trial", "in this analysis",
        ]):
            return "Abstract", "abstract"

        # Check for methods section indicators
        if any(kw in text_lower for kw in [
            "patients were enrolled", "eligibility criteria",
            "inclusion criteria", "exclusion criteria",
            "randomized to", "randomly assigned",
            "study design", "was a phase",
        ]):
            return "Methods", "methods"

        # Check for results indicators
        if any(kw in text_lower for kw in [
            "median follow-up", "overall survival was",
            "progression-free survival", "response rate",
            "hazard ratio", "p = 0.", "p=0.",
            "grade 3", "adverse events",
        ]):
            return "Results", "results"

        # Check for discussion indicators
        if any(kw in text_lower for kw in [
            "our findings", "these results suggest",
            "this study demonstrates", "limitations",
            "in conclusion", "consistent with",
        ]):
            return "Discussion", "discussion"

        # Check for eligibility
        if any(kw in text_lower for kw in [
            "inclusion criteria", "exclusion criteria",
            "eligible patients", "eligibility",
        ]):
            return "Eligibility", "eligibility"

        return raw or "Body", "other"

    # ── Clean up names with pixtral prefix ───────────────────────────
    cleaned = re.sub(r'pixtral[_\s]*:?\s*', '', raw, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^[\s_-]+|[\s_-]+$', '', cleaned)
    if not cleaned:
        cleaned = raw

    # Classify
    for pattern, stype in _SECTION_PATTERNS:
        if re.search(pattern, cleaned.lower()):
            return cleaned, stype

    return cleaned, "other"


def detect_abstract_chunk(
    section_type: str,
    section_window_idx: Optional[int],
    chunk_text: str,
) -> bool:
    """
    Determine if this chunk is the abstract or overview chunk that should
    be flagged for quick frontend reference.
    """
    if section_type == "abstract":
        return True
    # First section window of a document is often the abstract
    if section_window_idx == 0 and section_type in ("abstract", "other", "background"):
        text_lower = (chunk_text or "")[:300].lower()
        if any(kw in text_lower for kw in [
            "background", "purpose", "objective", "aim",
            "we aimed", "this study", "this trial",
        ]):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 2b. STRUCTURED CONTENT SECTION LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_structured_sections(
    processed_docs_dir: Path,
    full_text: bool = False,
    progress_every: int = 50,
) -> Dict[str, Dict[str, str]]:
    """
    Load real section names from _structured_content.json files.

    Scans the processed_documents directory structure:
      processed_documents_complete/
        [cancer_type]_processed_documents/
          [doc_id]/
            [doc_id]_structured_content.json

    Args:
        processed_docs_dir: path to the directory
        full_text: if True, store full section text (for doc-level
                   tagging / overview). If False, store only the
                   first 200 chars (enough for chunk-to-section matching).
        progress_every: print progress every N docs (Drive I/O is slow
                        and silent runs look like hangs).

    Returns:
        Dict mapping source_doc_dir_name → {section_name: text}
    """
    import time
    doc_sections: Dict[str, Dict[str, str]] = {}
    files_found = 0
    errors = 0

    if not processed_docs_dir or not processed_docs_dir.exists():
        print(f"  WARNING: processed_docs_dir '{processed_docs_dir}' not found")
        return doc_sections

    t_start = time.time()
    last_progress = t_start

    # Walk the directory tree
    for category_dir in sorted(processed_docs_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        print(f"    scanning {category_dir.name}...", flush=True)

        for doc_dir in sorted(category_dir.iterdir()):
            if not doc_dir.is_dir():
                continue

            # Find _structured_content.json
            sc_files = list(doc_dir.glob("*_structured_content.json"))
            if not sc_files:
                continue

            try:
                with open(sc_files[0], "r", encoding="utf-8") as f:
                    sc = json.load(f)

                sections = sc.get("sections", {})
                if not sections:
                    continue

                # Build lookup: section_name → text (full or first 200 chars)
                section_lookup: Dict[str, str] = {}
                for sec_name, sec_text in sections.items():
                    if isinstance(sec_text, str) and sec_text.strip():
                        clean_name = sec_name.strip().lstrip("#").strip()
                        if full_text:
                            section_lookup[clean_name] = sec_text.strip()
                        else:
                            section_lookup[clean_name] = sec_text.strip()[:200]

                if section_lookup:
                    doc_sections[doc_dir.name] = section_lookup
                    files_found += 1

                # Progress log — critical because Drive I/O is slow/silent
                now = time.time()
                if files_found % progress_every == 0 and now - last_progress > 2:
                    rate = files_found / max(now - t_start, 0.001)
                    print(f"      … {files_found} docs loaded "
                          f"({rate:.1f}/s, {now - t_start:.0f}s elapsed)",
                          flush=True)
                    last_progress = now

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    WARNING: Error reading {sc_files[0].name}: {e}")

    print(f"  Loaded structured sections for {files_found} documents "
          f"({errors} errors) in {time.time() - t_start:.0f}s")
    return doc_sections


def _clean_for_matching(text: str) -> str:
    """Strip LaTeX, markdown, whitespace, and special chars for robust matching."""
    s = text.lower()
    # Remove LaTeX commands and math
    s = re.sub(r'\$[^$]*\$', '', s)
    s = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    # Remove markdown headers
    s = re.sub(r'#{1,6}\s*', '', s)
    # Remove HTML tags
    s = re.sub(r'<[^>]+>', '', s)
    # Remove footnote markers
    s = re.sub(r'\[\^?\d+\]', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove non-alphanumeric except spaces
    s = re.sub(r'[^a-z0-9 ]', '', s)
    return s


def match_chunk_to_section(
    chunk_text: str,
    doc_sections: Dict[str, str],
) -> Optional[Tuple[str, str]]:
    """
    Match a chunk's text to a real section name from structured_content.

    Strategy:
    1. First try: match the section NAME itself against the chunk text
       (the chunk might start with "Abstract" or "Methods" as a heading)
    2. Then try: clean both texts and compare overlapping word sequences
    3. Then try: extract distinctive words from the section text and check
       if enough of them appear in the chunk

    Args:
        chunk_text: The chunk's text content
        doc_sections: {section_name: section_text_snippet} for this document

    Returns:
        (section_name, section_type) if matched, None otherwise
    """
    if not chunk_text or not doc_sections:
        return None

    chunk_clean = _clean_for_matching(chunk_text[:800])
    if len(chunk_clean) < 20:
        return None

    best_match = None
    best_score = 0

    for sec_name, sec_snippet in doc_sections.items():
        score = 0

        # ── Strategy 1: Section name appears in chunk text ───────────
        sec_name_clean = _clean_for_matching(sec_name)
        if sec_name_clean and len(sec_name_clean) >= 4 and sec_name_clean in chunk_clean:
            score = max(score, 100 + len(sec_name_clean))

        # ── Strategy 2: Cleaned text overlap ─────────────────────────
        sec_clean = _clean_for_matching(sec_snippet[:400])
        if len(sec_clean) >= 20:
            # Extract 5-word windows from section and check how many
            # appear in the chunk
            sec_words = sec_clean.split()
            if len(sec_words) >= 5:
                windows_found = 0
                windows_total = 0
                for i in range(0, min(len(sec_words) - 4, 10)):
                    window = ' '.join(sec_words[i:i+5])
                    windows_total += 1
                    if window in chunk_clean:
                        windows_found += 1
                if windows_total > 0 and windows_found >= 2:
                    score = max(score, 50 + windows_found * 10)

            # Also try: first meaningful sentence of section in chunk
            # Skip past any author/affiliation text to find real content
            sentences = re.split(r'[.!?]\s+', sec_clean)
            for sent in sentences[:5]:
                sent = sent.strip()
                if len(sent) >= 30 and sent in chunk_clean:
                    score = max(score, 80 + len(sent))
                    break

        if score > best_score:
            best_score = score
            best_match = sec_name

    if best_match and best_score >= 50:
        _, section_type = normalize_section_name(best_match, chunk_text)
        return best_match, section_type

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN SCRIPT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Re-scan Qdrant chunks against updated ontology/keywords + fix section names"
    )
    parser.add_argument("--qdrant-url", required=True, help="Qdrant server URL")
    parser.add_argument("--qdrant-api-key", required=True, help="Qdrant API key")
    parser.add_argument("--collection-name", required=True, help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Points per scroll batch")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--max-points", type=int, default=None, help="Limit points processed (for testing)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data/ directory path")
    parser.add_argument(
        "--processed-docs-dir", type=str, default=None,
        help="Path to processed_documents_complete/ directory containing "
             "_structured_content.json files with real section names. "
             "Example: /content/drive/MyDrive/processed_documents_complete"
    )
    # LLM validation layer (optional)
    parser.add_argument(
        "--validate-with-llm", action="store_true",
        help="Pass each chunk and its regex tags through gpt-4o-mini to "
             "validate and correct polarity, values, and misattributions. "
             "Requires --openai-api-key. ~$10 for a full collection pass.",
    )
    parser.add_argument(
        "--openai-api-key", type=str, default=None,
        help="OpenAI API key (required if --validate-with-llm)",
    )
    parser.add_argument(
        "--validation-model", default="gpt-4o-mini",
    )
    parser.add_argument(
        "--validation-workers", type=int, default=10,
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    print("\n" + "=" * 70)
    print("  METADATA RE-UPSERT (no re-embedding)")
    print("=" * 70)
    print(f"  Qdrant: {args.qdrant_url}")
    print(f"  Collection: {args.collection_name}")
    print(f"  Data dir: {data_dir}")
    print(f"  Processed docs: {args.processed_docs_dir or '(not provided — text heuristics only)'}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Max points: {args.max_points or 'all'}")
    print("=" * 70)

    # ── Load keyword/ontology tables ──────────────────────────────────
    print("\n📚 Loading keyword and ontology files...")
    tagger = build_keyword_scanner(data_dir)

    # ── Load structured sections from processed documents ─────────────
    import time as _time
    all_doc_sections: Dict[str, Dict[str, str]] = {}
    doc_level_tags_by_dir: Dict[str, Dict[str, Any]] = {}
    if args.processed_docs_dir:
        print(f"\n📄 Loading structured sections from {args.processed_docs_dir}...",
              flush=True)
        # full_text=True so we can build a real 20k-char overview for
        # doc-level tagging. Section-matching logic below still works
        # because we match against the start of each section's text.
        all_doc_sections = load_structured_sections(
            Path(args.processed_docs_dir),
            full_text=True,
        )

        # ── Pre-compute doc-level tags from each doc's overview text ──
        print(f"\n🏷  Pre-computing doc-level tags from {len(all_doc_sections)} docs...",
              flush=True)
        DOC_INHERIT = (
            "cancer_types_detected", "histologies_detected",
            "histopathologic_types", "sites_detected",
            "stages_detected", "tnm_detected", "grades_detected",
            "staging_qualifier", "disease_status_detected",
            "treatment_lines_detected", "drugs_detected",
            "biomarkers_detected", "genomic_alterations",
            "imaging_detected", "serum_markers_detected",
            "biomarker_status_flat", "patient_demographics_flat",
            "ajcc_tags",
        )
        t_pre = _time.time()
        n_done = 0
        for doc_dir_name, sections in all_doc_sections.items():
            # Build overview text = concat sections up to 20,000 chars
            overview_parts: List[str] = []
            total_len = 0
            for sec_name, sec_text in sections.items():
                if not isinstance(sec_text, str) or not sec_text.strip():
                    continue
                overview_parts.append(sec_text.strip())
                total_len += len(sec_text)
                if total_len >= 20000:
                    break
            overview_text = "\n\n".join(overview_parts)[:20000]
            if not overview_text.strip():
                n_done += 1
                continue
            overview_detail = scan_text_for_keywords_detailed(overview_text, tagger)
            doc_tags: Dict[str, Any] = {}
            for field in DOC_INHERIT:
                value = overview_detail.get(field)
                if value:
                    new_key = "doc_level_" + field.replace("_detected", "")
                    doc_tags[new_key] = (
                        list(value) if isinstance(value, list) else value
                    )
            if doc_tags:
                doc_level_tags_by_dir[doc_dir_name] = doc_tags
            n_done += 1
            if n_done % 50 == 0:
                rate = n_done / max(_time.time() - t_pre, 0.001)
                eta = (len(all_doc_sections) - n_done) / max(rate, 0.001)
                print(f"      … tagged {n_done}/{len(all_doc_sections)} "
                      f"docs ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
        print(f"  Built doc-level tags for {len(doc_level_tags_by_dir)} docs "
              f"in {_time.time() - t_pre:.0f}s", flush=True)
    else:
        print("\n⚠ No --processed-docs-dir provided — using text heuristics for section names")

    # ── Connect to Qdrant ─────────────────────────────────────────────
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        timeout=120,
    )

    # ── Scroll through all points ─────────────────────────────────────
    print(f"\n🔄 Scrolling through collection '{args.collection_name}'...")

    total_processed = 0
    total_updated = 0
    section_type_counts: Dict[str, int] = defaultdict(int)
    abstract_count = 0
    section_matched_from_structured = 0
    doc_dir_found_on_disk = 0
    doc_dir_not_found = 0
    keyword_delta = 0  # how many more keywords we found vs original
    total_expanded_synonyms = 0
    _seen_doc_dirs: Set[str] = set()

    offset = None  # scroll cursor

    while True:
        results, offset = client.scroll(
            collection_name=args.collection_name,
            limit=args.batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            break

        batch_updates = []

        for point in results:
            payload = dict(point.payload or {})

            # Skip PTO frames — they have their own metadata
            if payload.get("node_type") and "pto_frame" in str(payload.get("node_type", "")):
                continue

            text = payload.get("text", "")
            raw_section = payload.get("section", "")
            section_window_idx = payload.get("section_window_idx")

            # ── 1. Re-scan keywords ──────────────────────────────────
            old_kw = (payload.get("metadata") or {}).get("keyword_matches", {})
            old_kw_count = sum(len(v) for v in old_kw.values()) if isinstance(old_kw, dict) else 0

            detailed = scan_text_for_keywords_detailed(text, tagger)

            # Optional LLM validation pass — corrects polarity / values /
            # misattributions without re-extracting from scratch.
            if args.validate_with_llm:
                from src.ingestion.llm_validator import (
                    validate_and_correct,
                    is_validation_worthwhile,
                )
                if is_validation_worthwhile(detailed):
                    detailed = validate_and_correct(
                        text, detailed,
                        openai_api_key=args.openai_api_key,
                        model=args.validation_model,
                    )

            new_kw_matches = detailed["keyword_matches"]
            new_kw_flat = detailed["keywords_flat"]
            new_ont_matches = detailed["ontology_tags"]
            new_expanded = detailed["expanded_synonyms"]
            new_kw_count = sum(len(v) for v in new_kw_matches.values())
            keyword_delta += (new_kw_count - old_kw_count)

            # ── 2. Normalize section name ────────────────────────────
            # PRIMARY: try to match chunk text against real section names
            # from the structured_content.json file for this document
            source_doc_dir = payload.get("source_doc_dir_name", "")
            doc_sections = all_doc_sections.get(source_doc_dir, {})

            # If exact match fails, try fuzzy matching by normalized name
            if not doc_sections and source_doc_dir:
                # Qdrant source_doc_dir_name may differ from disk dir name:
                #   Qdrant: "doi_10. 1200_JCO.20.02914" (with spaces)
                #   Disk:   "doi_10._1200_JCO.20.02914" (underscores)
                # Or Qdrant may have a hash suffix that disk doesn't.
                # Normalize: strip spaces/underscores/hyphens, lowercase
                def _norm(s):
                    return re.sub(r'[\s_\-]+', '', s.lower().strip())

                norm_qdrant = _norm(source_doc_dir)
                for disk_name, sections in all_doc_sections.items():
                    norm_disk = _norm(disk_name)
                    # Check containment both ways (one may have extra hash suffix)
                    if (norm_qdrant == norm_disk
                        or norm_qdrant.startswith(norm_disk)
                        or norm_disk.startswith(norm_qdrant)
                        or norm_qdrant in norm_disk
                        or norm_disk in norm_qdrant):
                        doc_sections = sections
                        break

            # Track doc-dir match rate (per unique doc, not per chunk)
            if source_doc_dir and source_doc_dir not in _seen_doc_dirs:
                _seen_doc_dirs.add(source_doc_dir)
                if doc_sections:
                    doc_dir_found_on_disk += 1
                else:
                    doc_dir_not_found += 1
                    if doc_dir_not_found <= 5:
                        print(f"    [section] No disk match for: '{source_doc_dir}'")

            sc_match = match_chunk_to_section(text, doc_sections) if doc_sections else None
            if sc_match:
                cleaned_section, section_type = sc_match
                section_matched_from_structured += 1
            else:
                # FALLBACK: text heuristics
                cleaned_section, section_type = normalize_section_name(raw_section, text)
            section_type_counts[section_type] += 1

            # ── 3. Detect abstract chunk ─────────────────────────────
            is_abstract = detect_abstract_chunk(section_type, section_window_idx, text)
            if is_abstract:
                abstract_count += 1

            # ── 4. Build payload update ──────────────────────────────
            update = {}

            # Update metadata.keyword_matches + expanded synonyms
            metadata = dict(payload.get("metadata") or {})
            metadata["keyword_matches"] = new_kw_matches
            metadata["keywords_flat"] = new_kw_flat
            metadata["ontology_tags"] = new_ont_matches
            # Expanded synonyms: all known names for each detected term
            # (drug brand names, cancer ontology synonyms, clinical context)
            # e.g. "pembrolizumab" found in text → expanded_synonyms includes
            # "Keytruda", "MK-3475", "anti-PD1", "ICI", "CPI"
            metadata["expanded_synonyms"] = new_expanded
            # All structured clinical slots from the canonical tagger
            # (now including LLM-validated versions when enabled)
            for _slot_field in (
                "cancer_types_detected", "histologies_detected",
                "histopathologic_types", "sites_detected",
                "patient_demographics",
                "stages_detected", "tnm_detected",
                "grades_detected", "staging_qualifier",
                "disease_status_detected", "treatment_lines_detected",
                "drugs_detected", "biomarkers_detected",
                "genomic_alterations", "imaging_detected",
                "serum_markers_detected", "biomarker_status",
                "ajcc_tags", "_llm_validated", "_llm_corrections",
            ):
                if _slot_field in detailed:
                    metadata[_slot_field] = detailed[_slot_field]
            # Flat list counterparts for Qdrant KEYWORD indexing (Qdrant
            # can't index nested dicts; flat parallel lists are filterable).
            metadata["biomarker_status_flat"] = [
                f"{canonical}:{status}"
                for canonical, statuses in (detailed.get("biomarker_status") or {}).items()
                for status in (statuses or [])
            ]
            metadata["patient_demographics_flat"] = [
                f"{axis}:{value}"
                for axis, values in (detailed.get("patient_demographics") or {}).items()
                for value in (values or [])
            ]
            metadata["keyword_matches_flat"] = [
                f"{category}:{term}"
                for category, terms in (detailed.get("keyword_matches") or {}).items()
                for term in (terms or [])
            ]
            # ── Inherit doc-level tags from the overview of this doc ──
            # So a Methods/Results chunk retrieved by vector similarity
            # still carries the study's primary axis (cancer, biomarkers).
            if doc_level_tags_by_dir:
                doc_tags = doc_level_tags_by_dir.get(source_doc_dir)
                if doc_tags:
                    metadata.update(doc_tags)
            update["metadata"] = metadata

            # Update section fields
            if cleaned_section != raw_section:
                update["section"] = cleaned_section
                update["section_original"] = raw_section  # preserve original
            update["section_type"] = section_type
            update["is_abstract"] = is_abstract

            total_expanded_synonyms += len(new_expanded)
            batch_updates.append((point.id, update))
            total_processed += 1

        # ── Apply batch updates ──────────────────────────────────────
        if batch_updates and not args.dry_run:
            for point_id, update_payload in batch_updates:
                client.set_payload(
                    collection_name=args.collection_name,
                    payload=update_payload,
                    points=[point_id],
                )
            total_updated += len(batch_updates)

        elif batch_updates and args.dry_run:
            total_updated += len(batch_updates)
            # Show a few examples
            if total_processed <= 20:
                for pid, upd in batch_updates[:3]:
                    sec = upd.get("section", "(unchanged)")
                    stype = upd.get("section_type", "?")
                    is_abs = upd.get("is_abstract", False)
                    n_kw = sum(len(v) for v in upd.get("metadata", {}).get("keyword_matches", {}).values())
                    n_ont = len(upd.get("metadata", {}).get("ontology_tags", []))
                    n_exp = len(upd.get("metadata", {}).get("expanded_synonyms", []))
                    exp_sample = upd.get("metadata", {}).get("expanded_synonyms", [])[:5]
                    print(f"    [DRY RUN] point={str(pid)[:20]}... "
                          f"section='{sec}' type={stype} "
                          f"abstract={is_abs} kw={n_kw} ont={n_ont} "
                          f"expanded={n_exp}")
                    if exp_sample:
                        print(f"              expanded sample: {exp_sample}")

        if total_processed % 1000 == 0 and total_processed > 0:
            print(f"  Progress: {total_processed} points processed, {total_updated} updated")

        if args.max_points and total_processed >= args.max_points:
            print(f"  Reached max_points limit ({args.max_points})")
            break

        if offset is None:
            break

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  METADATA RE-UPSERT COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Points processed: {total_processed}")
    print(f"  Points updated: {total_updated}")
    print(f"  Abstracts detected: {abstract_count}")
    print(f"  Sections matched from structured_content: {section_matched_from_structured}")
    print(f"  Sections inferred from text heuristics: {total_processed - section_matched_from_structured}")
    print(f"  Doc dirs found on disk: {doc_dir_found_on_disk} "
          f"({doc_dir_not_found} not found)")
    print(f"  Keyword delta: {keyword_delta:+d} (net new keywords found)")
    print(f"  Expanded synonyms added: {total_expanded_synonyms} total "
          f"({total_expanded_synonyms / max(total_processed, 1):.1f} avg/chunk)")
    print(f"\n  Section type distribution:")
    for stype, count in sorted(section_type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total_processed, 1)
        print(f"    {stype:20s}: {count:6d} ({pct:5.1f}%)")
    if args.dry_run:
        print(f"\n  ⚠ DRY RUN — no changes written to Qdrant")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
