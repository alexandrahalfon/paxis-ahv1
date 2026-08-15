#!/usr/bin/env python3
"""
Section-Based Chunk Upsert + Keyword Tagging

Reads _structured_content.json files from the processed documents
directory, creates properly sectioned chunks with clean section names,
embeds them, upserts to Qdrant, and tags with all keyword/ontology/
synonym data.

This produces SUPPLEMENTARY points alongside the existing chunks —
it does NOT replace them. The new points have:
  - Clean section names from the source (Abstract, Methods, Results, etc.)
  - section_type classification (abstract, methods, results, discussion, etc.)
  - is_abstract flag on abstract chunks
  - Full keyword/ontology/synonym tagging from all 4 data files

Data files used for tagging:
  1. data/keywords/extractor_keywords.json (1200+ extraction keywords)
  2. data/ontology/cancer_type_ontology.json (22 cancer types with synonyms)
  3. data/ontology/clinical_trial_ontology.json (400+ controlled terms)
  4. data/ajcc_staging_tables.json (35 cancer types, aliases + TNM defs)

Usage (Colab):
    python src/ingestion/section_upsert.py \
        --qdrant-url URL \
        --qdrant-api-key KEY \
        --collection-name COLL \
        --openai-api-key OPENAI_KEY \
        --processed-docs-dir /content/drive/MyDrive/processed-documents-latest \
        --dry-run --max-docs 5

Cost: ~$15-20 for embedding 873 docs (~5000-20000 section chunks)
"""

import argparse
import copy
import hashlib
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── Ensure project root is in sys.path ──────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.name == "ingestion" else SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"


# ═══════════════════════════════════════════════════════════════════════════
# 1. SECTION PARSING
# ═══════════════════════════════════════════════════════════════════════════

# Section type classification patterns
_SECTION_TYPE_PATTERNS: List[Tuple[str, str]] = [
    (r'\babstract\b', "abstract"),
    (r'\bsummary\b', "abstract"),  # many papers use "Summary" for abstract
    (r'\bsynopsis\b', "abstract"),
    (r'\bbackground\b', "background"),
    (r'\bintroduction\b', "introduction"),
    (r'\bobjective\b|\bpurpose\b|\baim\b', "introduction"),
    (r'\bmethods?\b|\bmaterials?\s+and\s+methods?\b|\bstudy\s+design\b|\bpatients?\s+and\s+methods?\b', "methods"),
    (r'\bsearch\s+strategy\b|\bliterature\s+search\b', "methods"),
    (r'\bstatistical\s+analysis\b|\bstatistical\s+methods\b', "methods"),
    (r'\bpatient\s+population\b|\bpatient\s+selection\b|\bstudy\s+population\b', "methods"),
    (r'\btreatment\s+protocol\b|\btreatment\s+plan\b|\btreatment\s+regimen\b', "methods"),
    (r'\bdose\b.*\bfraction\b|\bdose\s+constraint\b|\bdose\s+escalat\b', "methods"),
    (r'\bresults?\b|\boutcomes?\b|\bfindings?\b', "results"),
    (r'\btreatment\s+outcomes?\b|\bsurvival\b|\befficacy\b', "results"),
    (r'\btoxicit\w+\b|\badverse\s+events?\b|\bsafety\b', "results"),
    (r'\blocal\s+control\b|\boverall\s+survival\b|\bprogression.free\b', "results"),
    (r'\bdiscussion\b', "discussion"),
    (r'\bconclusion\b', "conclusion"),
    (r'\beligib\w+\b|\binclusion\s+criteria\b|\bexclusion\s+criteria\b', "eligibility"),
    (r'\breferences?\b|\bbibliography\b', "references"),
    (r'\backnowledg\w+\b|\bfunding\b|\bdisclosures?\b|\bconflict\b|\bauthor\s+contrib', "acknowledgments"),
    (r'\bsupplement\w+\b|\bappendix\b', "supplementary"),
    (r'\bdata\s+sharing\b|\bdata\s+availab\b', "supplementary"),
]

# Sections to skip (not clinically useful for retrieval)
_SKIP_SECTIONS = {
    "references", "bibliography", "acknowledgments", "acknowledgements",
    "funding", "disclosures", "author contributions", "competing interests",
    "conflict of interest", "data availability", "supplementary material",
    "data sharing", "data sharing statement", "prior presentation",
    "affiliations", "corresponding author", "footnotes",
    "author disclosures", "conflicts of interest",
    "support", "asco answers", "cancernet", "knowledge generated",
    "relevance", "key objective",
    # Journal boilerplate / ads
    "save these dates", "cme opportunit", "annual meeting",
    "sports concussion", "writeclick", "rapid online correspondence",
    "editorial", "correspondence to", "go to neurology",
    "supplemental data", "supplemental material",
    # Misc non-content
    "study funding", "disclosure", "trial registration",
    "affiliations", "corresponding author", "footnotes",
}


def classify_section_type(section_name: str, section_text: str = "") -> str:
    """Classify a section name into a standard type."""
    name_lower = section_name.lower().strip().lstrip("#").strip()

    for pattern, stype in _SECTION_TYPE_PATTERNS:
        if re.search(pattern, name_lower):
            return stype

    # Check text content if name is ambiguous
    if section_text:
        text_lower = section_text[:300].lower()
        for pattern, stype in _SECTION_TYPE_PATTERNS:
            if re.search(pattern, text_lower):
                return stype

    return "other"


def should_skip_section(section_name: str, section_text: str = "") -> bool:
    """Check if a section should be skipped (references, acknowledgments, etc.)."""
    # Clean name thoroughly — remove markdown, numbers, special chars
    name_lower = section_name.lower().strip()
    name_lower = re.sub(r'[#*_\-\d.]+', ' ', name_lower).strip()
    name_lower = re.sub(r'\s+', ' ', name_lower)

    for skip in _SKIP_SECTIONS:
        if skip in name_lower:
            return True

    # Also check via section type classification — if it classifies as
    # references/acknowledgments/supplementary, skip it
    stype = classify_section_type(section_name)
    if stype in ("references", "acknowledgments", "supplementary"):
        return True

    # Check text content for reference lists (numbered citations)
    if section_text:
        text_start = section_text[:300].strip()
        # Reference sections often start with numbered entries
        if re.match(r'^\s*\d+\.\s+[A-Z]', text_start):
            # Count numbered entries — if >3 in first 300 chars, it's refs
            ref_count = len(re.findall(r'\n\s*\d+\.\s+', text_start))
            if ref_count >= 3:
                return True

    # Skip sections that are purely author/affiliation lists
    # (often the first section in structured_content is the paper title
    # followed by author names and affiliations)
    if section_text:
        text_lower = section_text[:500].lower()
        # If the text is mostly affiliations (departments, universities, hospitals)
        affiliation_signals = sum(1 for kw in [
            "department of", "university", "hospital", "institute",
            "school of medicine", "medical center", "e-mail:",
            "correspondence", "@", "orcid",
        ] if kw in text_lower)
        # Count actual clinical content signals
        content_signals = sum(1 for kw in [
            "patient", "treatment", "study", "trial", "survival",
            "dose", "radiation", "cancer", "tumor", "stage",
            "outcome", "result", "method",
        ] if kw in text_lower)
        # If mostly affiliations and little content, skip
        if affiliation_signals >= 3 and content_signals <= 1:
            return True

    return False


def parse_structured_sections(
    structured_content: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parse sections from a structured_content.json file into chunk-ready dicts.

    Each section is split into paragraphs. Short paragraphs are merged
    to avoid tiny chunks. The section name and type are attached to each
    chunk.

    The FIRST non-skipped section is checked as a potential abstract if
    its section_type wasn't already classified as abstract — many papers
    have the abstract as the first section under the paper title (with
    a heading like the paper title itself, or "Summary", or no heading).

    Returns:
        List of {section_name, section_type, is_abstract, text, paragraph_idx}
    """
    sections = structured_content.get("sections", {})
    if not sections:
        return []

    chunks = []

    # ── Overview chunk: first 20000 chars of all sections concatenated ──
    # Each section is prefixed with its markdown header "## {name}" so
    # the downstream LLM/embedding can see which section each piece of
    # text came from. Author/affiliation-only sections are skipped so
    # the overview starts with actual clinical content.
    overview_parts: List[str] = []

    for sec_name, sec_text in sections.items():
        if not isinstance(sec_text, str) or not sec_text.strip():
            continue
        # Skip author lists, references, acknowledgments, boilerplate
        if should_skip_section(sec_name, sec_text):
            continue
        # Skip sections that are mostly author names / departments
        text_head = sec_text[:300].lower()
        if sum(1 for kw in [
            "department of", "university", "hospital", "@", "md,",
            "phd,", "orcid", "correspondence to",
        ] if kw in text_head) >= 3:
            continue

        clean_name = sec_name.strip().lstrip("#").strip()
        # Prefix with markdown header so the section boundary is preserved
        overview_parts.append(f"## {clean_name}\n\n{sec_text.strip()}")

        if sum(len(p) for p in overview_parts) >= 20000:
            break

    overview_text = "\n\n".join(overview_parts)[:20000].strip()
    if overview_text and len(overview_text) >= 100:
        # Clean
        overview_text = re.sub(r'\[\^\d+\]', '', overview_text)
        overview_text = re.sub(r'<br\s*/?>', '\n', overview_text)
        overview_text = re.sub(r'\\(?:mathrm|text|textbf|textit)\{([^}]*)\}', r'\1', overview_text)
        overview_text = re.sub(r'\$([^$]*)\$', r'\1', overview_text)
        overview_text = re.sub(r'\\%', '%', overview_text)
        overview_text = re.sub(r'\\[a-zA-Z]+', '', overview_text)
        overview_text = re.sub(r'\s{3,}', '\n\n', overview_text).strip()
        chunks.append({
            "section_name": "Overview",
            "section_type": "abstract",
            "is_abstract": True,
            "text": overview_text,
            "paragraph_idx": 0,
        })

    # ── Per-section chunks ────────────────────────────────────────────
    is_first_content_section = True

    for sec_name, sec_text in sections.items():
        if not isinstance(sec_text, str) or not sec_text.strip():
            continue

        # Clean section name
        clean_name = sec_name.strip().lstrip("#").strip()
        if not clean_name:
            continue

        # Skip non-content sections
        if should_skip_section(clean_name, sec_text):
            continue

        section_type = classify_section_type(clean_name, sec_text)

        # The first content section is often the abstract even if not
        # explicitly named "Abstract". Check if the text looks like an
        # abstract: starts with purpose/background/objective phrasing,
        # or is a "Summary" section.
        is_abstract = section_type == "abstract"
        # Only promote the VERY FIRST content section to abstract —
        # not every section that happens to mention "background"
        if is_first_content_section and not is_abstract:
            text_lower = sec_text[:500].lower()
            # Require STRONG abstract signals (not just "background")
            strong_signals = sum(1 for kw in [
                "we aimed", "this study aimed", "the aim of this",
                "we conducted a", "we performed a", "we evaluated",
                "we report", "we present the results",
                "this phase", "this randomized", "this prospective",
                "methods.", "results.", "conclusions.",
                "purpose.", "objective.",
            ] if kw in text_lower)
            if strong_signals >= 2:
                section_type = "abstract"
                is_abstract = True
                clean_name = f"Abstract ({clean_name})" if "abstract" not in clean_name.lower() else clean_name

        is_first_content_section = False

        # Clean text: remove LaTeX artifacts, excessive whitespace
        text = sec_text.strip()
        text = re.sub(r'\[\^\d+\]', '', text)  # footnote markers like [^1] only
        text = re.sub(r'<br\s*/?>', '\n', text)  # HTML breaks
        text = re.sub(r'\\(?:mathrm|text|textbf|textit)\{([^}]*)\}', r'\1', text)  # LaTeX formatting
        # LaTeX math: extract the content between $...$ instead of deleting it
        # e.g. "$n=320$" → "n=320", "$p=0.14$" → "p=0.14"
        text = re.sub(r'\$([^$]*)\$', r'\1', text)  # unwrap LaTeX math, keep content
        text = re.sub(r'\\%', '%', text)  # LaTeX escaped percent
        text = re.sub(r'\\circledR', '®', text)  # registered trademark
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # remaining LaTeX commands
        text = re.sub(r'\s{3,}', '\n\n', text)  # excessive whitespace
        text = text.strip()

        if len(text) < 30:
            continue

        # Split into paragraphs (by double newline or long single newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) >= 20]

        if not paragraphs:
            continue

        # Merge short paragraphs to avoid tiny chunks
        # Target: 200-1500 chars per chunk
        merged = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < 1500:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    merged.append(current)
                current = para
        if current:
            merged.append(current)

        for i, para_text in enumerate(merged):
            if len(para_text) < 30:
                continue
            chunks.append({
                "section_name": clean_name,
                "section_type": section_type,
                "is_abstract": is_abstract,
                "text": para_text,
                "paragraph_idx": i,
            })

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 2. KEYWORD / ONTOLOGY / SYNONYM TAGGING
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_keyword_scanner(data_dir: Path):
    """
    Build the canonical KeywordTagger, which loads all 4 data files with
    ambiguity-aware cancer tagging, biomarker/drug/alteration extraction,
    and AJCC staging. Returns the KeywordTagger instance directly.

    (Legacy signature returned a tuple; callers should treat the return
    value as opaque and pass it to tag_chunk_text().)
    """
    from src.ingestion.keyword_tagger import KeywordTagger
    tagger = KeywordTagger(data_dir=data_dir)
    return tagger


# ── Fields that propagate from doc-level overview to every chunk ─────────
# These capture the document's overall clinical context. They're inherited
# by every chunk so that retrieval filters can match non-abstract chunks
# (Methods / Results) without losing the primary cancer / biomarker axis.
_DOC_LEVEL_INHERIT_FIELDS = [
    "cancer_types_detected",
    "histologies_detected",
    "histopathologic_types",
    "sites_detected",
    "stages_detected",
    "tnm_detected",
    "grades_detected",
    "staging_qualifier",
    "disease_status_detected",
    "treatment_lines_detected",
    "drugs_detected",
    "biomarkers_detected",
    "genomic_alterations",
    "imaging_detected",
    "serum_markers_detected",
    "biomarker_status_flat",
    "patient_demographics_flat",
    "ajcc_tags",
]


def _build_doc_level_tags(overview_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given an overview chunk's metadata, build the doc_level_* fields
    that every chunk in the same document will inherit.

    Keeps chunk-level fields separate from doc-level fields, so queries
    can filter on EITHER:
      - metadata.cancer_types_detected      → chunks that explicitly mention it
      - metadata.doc_level_cancer_types     → any chunk from a doc about it
    """
    doc_tags: Dict[str, Any] = {}
    for field in _DOC_LEVEL_INHERIT_FIELDS:
        value = overview_metadata.get(field)
        if value:
            # Copy to doc_level_<field> (strip "_detected" suffix for
            # cleaner naming, keep everything else as-is).
            new_key = "doc_level_" + field.replace("_detected", "")
            doc_tags[new_key] = list(value) if isinstance(value, list) else value
    return doc_tags


def tag_chunk_text(
    text: str,
    tagger,
    _unused=None,  # kept for legacy signature compatibility
) -> Dict[str, Any]:
    """
    Scan text against all keyword/ontology tables via KeywordTagger +
    expand synonyms + staging/range expansions.

    Returns metadata dict ready to attach to a chunk payload, including
    dedicated biomarker/drug/alteration/cancer_types slots.
    """
    from src.ingestion.keyword_tagger import KeywordTagger

    # Support the legacy calling convention where callers pass
    # (term_to_categories, ontology_tags) as two positional args — if the
    # first arg is a dict, build a tagger on the fly.
    if not isinstance(tagger, KeywordTagger):
        tagger = KeywordTagger(data_dir=DATA_DIR)

    detailed = tagger.scan_text_detailed(text)
    flat_terms = detailed["keywords_flat"]

    # Staging + range expansions (script-local, not in KeywordTagger)
    expanded_terms: List[str] = list(detailed["expanded_synonyms"])
    try:
        from src.api.services.query_expansion import (
            _build_staging_expansions,
            _build_range_expansions,
        )
        detected_lower = {t.lower() for t in flat_terms} | {t.lower() for t in expanded_terms}
        for t in (_build_staging_expansions(text) | _build_range_expansions(text)):
            if t.lower() not in detected_lower:
                expanded_terms.append(t)
                detected_lower.add(t.lower())
    except Exception as e:
        if not hasattr(tag_chunk_text, '_warned'):
            print(f"    [expansion] Warning: {e}")
            tag_chunk_text._warned = True

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
# 3. EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

def embed_texts(
    texts: List[str],
    openai_api_key: str,
    model: str = "text-embedding-3-large",
    batch_size: int = 100,
) -> List[List[float]]:
    """Embed texts using OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=openai_api_key)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        if (i + batch_size) % 500 == 0:
            print(f"    Embedded {i + len(batch)}/{len(texts)} chunks")

    return all_embeddings


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Create section-based chunks from structured_content.json + embed + upsert + tag"
    )
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-api-key", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--openai-api-key", required=True)
    parser.add_argument("--processed-docs-dir", required=True,
                        help="Path to processed_documents directory with _structured_content.json files")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("--batch-size", type=int, default=50, help="Qdrant upsert batch size")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit docs for testing")
    # LLM validation layer (optional)
    parser.add_argument(
        "--validate-with-llm", action="store_true",
        help="After regex tagging, pass each clinical chunk through "
             "gpt-4o-mini to validate and correct the extracted tags. "
             "Adds ~$0.00054 per chunk (~$10 for a full 859-doc pass).",
    )
    parser.add_argument(
        "--validation-model", default="gpt-4o-mini",
        help="OpenAI model for validation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--validation-workers", type=int, default=10,
        help="Concurrent LLM calls (default: 10)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    processed_dir = Path(args.processed_docs_dir)

    print("\n" + "=" * 70)
    print("  SECTION-BASED CHUNK UPSERT + KEYWORD TAGGING")
    print("=" * 70)
    print(f"  Qdrant: {args.qdrant_url}")
    print(f"  Collection: {args.collection_name}")
    print(f"  Processed docs: {processed_dir}")
    print(f"  Embed model: {args.embed_model}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Max docs: {args.max_docs or 'all'}")
    print("=" * 70)

    if not processed_dir.exists():
        print(f"ERROR: {processed_dir} not found")
        sys.exit(1)

    # ── Load keyword/ontology tables ──────────────────────────────────
    print("\n📚 Loading keyword and ontology files...")
    tagger = build_keyword_scanner(data_dir)

    # ── Walk processed documents ──────────────────────────────────────
    print(f"\n📄 Scanning {processed_dir} for structured_content.json files...")

    all_chunks: List[Dict[str, Any]] = []
    docs_processed = 0
    docs_skipped = 0

    for category_dir in sorted(processed_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name  # e.g. "h&n_processed_documents"

        for doc_dir in sorted(category_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            if args.max_docs and docs_processed >= args.max_docs:
                break

            # Find structured_content.json
            sc_files = list(doc_dir.glob("*_structured_content.json"))
            if not sc_files:
                docs_skipped += 1
                continue

            try:
                with open(sc_files[0], "r", encoding="utf-8") as f:
                    sc = json.load(f)
            except Exception as e:
                print(f"    Error reading {sc_files[0].name}: {e}")
                docs_skipped += 1
                continue

            # Extract doc_meta from structured content
            dm = sc.get("document_metadata", {})
            info = dm.get("document_info", {})
            pub = dm.get("publication_info", {})

            title = (info.get("title") or "").strip() or None
            authors = info.get("authors") or []
            author_et_al = None
            if authors:
                first = authors[0] if isinstance(authors[0], str) else str(authors[0])
                author_et_al = f"{first.split(',')[0]} et al." if len(authors) > 1 else first.split(",")[0]
            year = None
            for y_field in ("year", "publication_year"):
                v = pub.get(y_field)
                if v:
                    m = re.search(r'(\d{4})', str(v))
                    if m:
                        year = int(m.group(1))
                        break
            journal = (pub.get("journal") or "").strip() or None
            doi = None
            for d_field in ("doi", "DOI"):
                v = pub.get(d_field)
                if v and isinstance(v, str):
                    doi = v.strip()
                    break

            doc_meta = {
                "title": title,
                "authors": authors,
                "author_et_al": author_et_al,
                "year": year,
                "journal": journal,
                "doi": doi,
            }

            # Build doc_id (match the format used by colab_pipeline)
            from src.ingestion.doc_id import normalize_doc_id
            doc_id_raw = doc_dir.name
            doc_id = normalize_doc_id(doc_id_raw)

            # Parse sections into chunks
            section_chunks = parse_structured_sections(sc)
            if not section_chunks:
                docs_skipped += 1
                continue

            # ── Tag each chunk with its own chunk-level tags ──────
            chunks_for_doc: List[Dict[str, Any]] = []
            for chunk_data in section_chunks:
                chunk_id = f"{doc_id}__sec_{chunk_data['section_type']}_{chunk_data['paragraph_idx']}"
                tag_metadata = tag_chunk_text(chunk_data["text"], tagger)

                payload = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_id_raw": doc_id_raw,
                    "category": category,
                    "source_doc_dir_name": doc_dir.name,
                    "chunk_type": "section_paragraph",
                    "chunk_granularity": "section_based",
                    "section": chunk_data["section_name"],
                    "section_type": chunk_data["section_type"],
                    "is_abstract": chunk_data["is_abstract"],
                    "section_paragraph_num": chunk_data["paragraph_idx"],
                    "text": chunk_data["text"],
                    "doc_meta": doc_meta,
                    "metadata": tag_metadata,
                    "node_type": "section_chunk",
                }
                chunks_for_doc.append(payload)

            # ── Build DOC-LEVEL tags from the overview chunk ──────
            # The overview is the first ~20,000 chars concatenation
            # that captures abstract + intro + methods headings. Its
            # tags represent the document's overall clinical context
            # (cancer type, biomarkers, population, etc.).
            # Every chunk in this doc inherits these tags as
            # doc_level_* fields, so filter queries can retrieve
            # non-abstract chunks (Methods/Results) without losing
            # the document's primary axis.
            overview_payload = next(
                (c for c in chunks_for_doc if c["section_type"] == "overview"),
                None,
            )
            if overview_payload is None:
                # Fallback: use the first abstract chunk
                overview_payload = next(
                    (c for c in chunks_for_doc if c["is_abstract"]),
                    chunks_for_doc[0] if chunks_for_doc else None,
                )
            if overview_payload is not None:
                doc_tags = _build_doc_level_tags(overview_payload["metadata"])
                for c in chunks_for_doc:
                    c["metadata"].update(doc_tags)

            all_chunks.extend(chunks_for_doc)

            docs_processed += 1
            if docs_processed % 100 == 0:
                print(f"    Processed {docs_processed} docs, {len(all_chunks)} chunks so far")

        if args.max_docs and docs_processed >= args.max_docs:
            break

    print(f"\n  Documents processed: {docs_processed}")
    print(f"  Documents skipped: {docs_skipped}")
    print(f"  Section chunks created: {len(all_chunks)}")

    # Section type distribution
    type_counts: Dict[str, int] = defaultdict(int)
    abstract_count = 0
    for c in all_chunks:
        type_counts[c["section_type"]] += 1
        if c["is_abstract"]:
            abstract_count += 1

    print(f"  Abstracts: {abstract_count}")
    print(f"  Section types:")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {stype:20s}: {count}")

    if not all_chunks:
        print("\n  No chunks to upsert.")
        return

    # ── Optional LLM validation pass ──────────────────────────────────
    # Reviews the regex-extracted tags against the source text and
    # corrects polarity, missing values, misattributed cancer types.
    if args.validate_with_llm:
        from src.ingestion.llm_validator import (
            validate_many,
            is_validation_worthwhile,
        )
        print(f"\n🤖 Validating tags with {args.validation_model} "
              f"(workers={args.validation_workers})...")
        to_validate: List[Tuple[str, Dict[str, Any]]] = [
            (c["text"], c["metadata"]) for c in all_chunks
        ]
        t_validate = time.time()
        corrected_metadata = validate_many(
            to_validate,
            openai_api_key=args.openai_api_key,
            model=args.validation_model,
            max_workers=args.validation_workers,
            skip_empty=True,
        )
        for chunk, corrected in zip(all_chunks, corrected_metadata):
            chunk["metadata"] = corrected

        validated_count = sum(
            1 for m in corrected_metadata if m.get("_llm_validated")
        )
        corrected_count = sum(
            1 for m in corrected_metadata if m.get("_llm_corrections")
        )
        total_corrections = sum(
            len(m.get("_llm_corrections", [])) for m in corrected_metadata
        )
        print(f"  Validation complete in {time.time() - t_validate:.1f}s")
        print(f"  Chunks validated:   {validated_count}/{len(all_chunks)}")
        print(f"  Chunks corrected:   {corrected_count}")
        print(f"  Total corrections:  {total_corrections}")

    if args.dry_run:
        print(f"\n  ⚠ DRY RUN — showing 5 sample chunks:")
        for c in all_chunks[:5]:
            print(f"    [{c['section_type']:12s}] {c['section'][:40]:40s} "
                  f"| {len(c['text']):4d} chars "
                  f"| kw={len(c['metadata']['keywords_flat'])} "
                  f"ont={len(c['metadata']['ontology_tags'])} "
                  f"exp={len(c['metadata']['expanded_synonyms'])} "
                  f"| abstract={c['is_abstract']}")
        print(f"\n  Would embed {len(all_chunks)} chunks (~${len(all_chunks) * 0.00013:.2f})")
        print(f"  ⚠ DRY RUN — no changes written")
        return

    # ── Embed ─────────────────────────────────────────────────────────
    print(f"\n🔮 Embedding {len(all_chunks)} chunks...")
    t_embed = time.time()
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, args.openai_api_key, model=args.embed_model)
    print(f"  Embedding took {time.time() - t_embed:.1f}s")

    # ── Upsert to Qdrant ──────────────────────────────────────────────
    print(f"\n🚀 Upserting {len(all_chunks)} section chunks to Qdrant...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, PayloadSchemaType

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        timeout=120,
    )

    # Create payload indexes for fast filtered retrieval.
    # KEYWORD indexes on list fields work transparently — Qdrant matches
    # ANY element in the list. All ~15 indexes together use ~30 MB RAM
    # for a 17k-point collection; filter speedup is 50-100x.
    INDEX_FIELDS = [
        # ── Chunk-level metadata ──────────────────────────────────────
        ("section",                   PayloadSchemaType.KEYWORD),
        ("section_type",              PayloadSchemaType.KEYWORD),
        ("is_abstract",               PayloadSchemaType.BOOL),
        ("node_type",                 PayloadSchemaType.KEYWORD),
        ("chunk_granularity",         PayloadSchemaType.KEYWORD),
        ("chunk_type",                PayloadSchemaType.KEYWORD),
        ("category",                  PayloadSchemaType.KEYWORD),
        ("doc_id",                    PayloadSchemaType.KEYWORD),
        # ── Hard-filter axis 1: cancer / histology / site ──────────────
        ("metadata.cancer_types_detected",      PayloadSchemaType.KEYWORD),
        ("metadata.histologies_detected",       PayloadSchemaType.KEYWORD),
        ("metadata.histopathologic_types",      PayloadSchemaType.KEYWORD),
        ("metadata.sites_detected",             PayloadSchemaType.KEYWORD),
        # ── Hard-filter axis 2: staging / status ───────────────────────
        ("metadata.stages_detected",            PayloadSchemaType.KEYWORD),
        ("metadata.tnm_detected",               PayloadSchemaType.KEYWORD),
        ("metadata.grades_detected",            PayloadSchemaType.KEYWORD),
        ("metadata.staging_qualifier",          PayloadSchemaType.KEYWORD),
        ("metadata.disease_status_detected",    PayloadSchemaType.KEYWORD),
        # ── Hard-filter axis 3: treatment / drugs ──────────────────────
        ("metadata.treatment_lines_detected",   PayloadSchemaType.KEYWORD),
        ("metadata.drugs_detected",             PayloadSchemaType.KEYWORD),
        # ── Hard-filter axis 4: biomarkers / alterations / markers ─────
        ("metadata.biomarkers_detected",        PayloadSchemaType.KEYWORD),
        ("metadata.genomic_alterations",        PayloadSchemaType.KEYWORD),
        ("metadata.imaging_detected",           PayloadSchemaType.KEYWORD),
        ("metadata.serum_markers_detected",     PayloadSchemaType.KEYWORD),
        # ── Flattened parallel lists (from dict fields) ────────────────
        ("metadata.biomarker_status_flat",      PayloadSchemaType.KEYWORD),
        ("metadata.patient_demographics_flat",  PayloadSchemaType.KEYWORD),
        ("metadata.keyword_matches_flat",       PayloadSchemaType.KEYWORD),
        # ── Cross-axis catch-all categories ────────────────────────────
        ("metadata.ajcc_tags",                  PayloadSchemaType.KEYWORD),
        ("metadata.ontology_tags",              PayloadSchemaType.KEYWORD),
        ("metadata.keywords_flat",              PayloadSchemaType.KEYWORD),
        ("metadata.expanded_synonyms",          PayloadSchemaType.KEYWORD),
        # ── Doc-level context tags (inherited from overview chunk)  ────
        # Every chunk in a doc carries these, so a Methods/Results chunk
        # retrieved by vector similarity still carries the document's
        # primary axis (cancer type, biomarkers, population). Filter on
        # these to recall non-abstract chunks of studies that MATCH the
        # query profile.
        ("metadata.doc_level_cancer_types",         PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_histologies",          PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_histopathologic_types", PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_sites",                PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_stages",               PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_tnm",                  PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_grades",               PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_staging_qualifier",    PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_disease_status",       PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_treatment_lines",      PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_drugs",                PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_biomarkers",           PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_genomic_alterations",  PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_imaging",              PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_serum_markers",        PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_biomarker_status_flat", PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_patient_demographics_flat", PayloadSchemaType.KEYWORD),
        ("metadata.doc_level_ajcc_tags",            PayloadSchemaType.KEYWORD),
    ]
    created = 0
    for field_name, field_type in INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=args.collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            created += 1
        except Exception:
            pass  # index may already exist — idempotent
    print(f"  Ensured {created}/{len(INDEX_FIELDS)} payload indexes")

    points = []
    for chunk, embedding in zip(all_chunks, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"]))
        points.append(PointStruct(id=point_id, vector=embedding, payload=chunk))

    # Batch upsert
    t_upsert = time.time()
    for i in range(0, len(points), args.batch_size):
        batch = points[i:i + args.batch_size]
        client.upsert(collection_name=args.collection_name, points=batch)
        if (i + args.batch_size) % 500 == 0:
            print(f"    Upserted {i + len(batch)}/{len(points)} points")

    print(f"  Upsert took {time.time() - t_upsert:.1f}s")

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Section chunks upserted: {len(points)}")
    print(f"  Documents: {docs_processed}")
    print(f"  Abstracts: {abstract_count}")
    print(f"  Embedding cost: ~${len(points) * 0.00013:.2f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
