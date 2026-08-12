"""
LLM Validation Layer for the Keyword Tagger

Takes a chunk of text and the structured tags produced by the regex-based
KeywordTagger, and uses gpt-4o-mini to review and correct them against the
source text. The LLM is used as a VALIDATOR, not a primary extractor —
it only fixes things that are wrong (polarity flipped, value missing,
wrong cancer, invented tags).

Usage:
    from src.ingestion.llm_validator import validate_and_correct

    tagged = tagger.scan_text_detailed(chunk_text)
    corrected = validate_and_correct(
        chunk_text,
        tagged,
        openai_api_key=OPENAI_API_KEY,
    )

Behavior:
    - Preserves the exact schema of scan_text_detailed() output
    - Only overwrites the clinical validation fields
    - On any failure (network error, JSON parse error, etc.) returns the
      original untouched — never blocks the pipeline
    - Adds "_llm_validated": True and "_llm_corrections": [...] for audit

Cost estimate (gpt-4o-mini, ~2000 input / 300 output tokens per call):
    ~$0.00054 per chunk. For 18,000 chunks (859 docs × ~21 chunks each):
    ~$10. With batch API (50% off): ~$5.

Selective validation:
    `is_validation_worthwhile(extracted)` returns False for chunks with no
    clinical content (nothing to validate). Pass this check before calling
    the LLM to avoid wasting money on empty chunks.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


# ── Fields the LLM validates ─────────────────────────────────────────────
# These are the structured clinical fields that benefit from validation.
# Other fields (keywords_flat, keyword_matches, etc.) are low-risk and
# skipped to keep the prompt compact.
_VALIDATION_FIELDS = [
    # Patient demographics (hard filter: age / gender / ECOG / KPS)
    "patient_demographics",
    # Diagnosis axis
    "cancer_types_detected",        # Cancer by Location (breast/lung/GI/etc)
    "histologies_detected",         # Specific histology (IDC/SCC/DLBCL/etc)
    "histopathologic_types",        # Family (Carcinoma/Sarcoma/Lymphoma/etc)
    "sites_detected",               # Anatomical site
    # Staging axis (clinical vs pathologic, TNM, Stage Group, grade)
    "stages_detected",
    "tnm_detected",
    "grades_detected",
    "staging_qualifier",            # Clinical | Pathologic
    "disease_status_detected",
    # Treatment history axis
    "treatment_lines_detected",
    "drugs_detected",
    # Biomarker axis (with polarity/value in biomarker_status)
    "biomarkers_detected",
    "genomic_alterations",
    "imaging_detected",
    "serum_markers_detected",
    "biomarker_status",
]


SYSTEM_PROMPT = """You are a clinical oncology data validator for a clinical trial matching system.

You receive:
1. A PASSAGE of clinical text (abstract, methods, results, or eligibility)
2. A JSON object of TAGS extracted by a regex-based scanner

Your job: VERIFY each tag is correct against the passage and CORRECT errors.
Return a single JSON object with exactly the same keys.

============================================================
FIELDS YOU MUST VERIFY AND CORRECT (align with trial hard filters):
============================================================

1. PATIENT DEMOGRAPHICS (patient_demographics)
   Structure: {"age": [...], "gender": [...], "performance_status": [...]}
   Age formats: "range:18-75", "median:64", "min:18", "max:65", "individual:80"
   Gender formats: "male:60%", "female:n=25", "male and female", "pediatric", "adult"
   Performance: "ECOG 0", "ECOG 0-1", "KPS 70", "Lansky 80"
   Verify every entry is supported by the passage. Add missing entries when
   the passage explicitly states them.

2. DIAGNOSIS AXIS
   - cancer_types_detected: Cancer by Location — Breast Cancer / Lung Cancer
     / Gastrointestinal Cancers / Gynecologic Cancers / Sarcoma / etc.
     REMOVE cancers the passage is NOT about (a breast passage wrongly
     tagged as GI because HER2 is in both ontologies).
   - histologies_detected: specific histology (Invasive ductal carcinoma,
     Squamous cell carcinoma, DLBCL, GIST, Glioblastoma, Melanoma, etc.).
   - histopathologic_types: family classification — Carcinoma / Sarcoma /
     Lymphoma / Leukemia / Myeloma / Melanoma / Glioma / Meningioma /
     Neuroendocrine. Infer from histology if present.
   - sites_detected: anatomical site mentioned (Breast, Lung, Rectum,
     Bronchus, Oropharynx, Lymph node, etc.).

3. STAGING AXIS (critical — Clinical vs Pathologic matters)
   - stages_detected: "Stage I", "Stage IIB", "Stage IIIC", "Stage IV",
     "Stage IVB". Roman numerals with A/B/C substage.
   - tnm_detected: T/N/M components WITH prefix (c=clinical, p=pathologic,
     yp=post-neoadjuvant pathologic). E.g. "pT2", "cN3", "ypT1", "M1b".
     TNM must include axis and stage; prefix when known.
   - grades_detected: "G1" / "G2" / "G3" / "G4" / "GX" / "Gleason 4+3=7"
     / "ISUP grade group 3" / "WHO grade IV" / "poorly-differentiated".
   - staging_qualifier: "clinical" | "pathologic" | "pre-treatment" |
     "post-neoadjuvant". Verify this matches what TNM prefixes say (cTNM
     → clinical; pTNM → pathologic; ypTNM → post-neoadjuvant).
   - disease_status_detected: Metastatic / Locally advanced / Recurrent
     / Refractory / Progressive disease / Unresectable / Castration-
     resistant / Triple negative / ICI-refractory / etc.

4. TREATMENT HISTORY AXIS
   - treatment_lines_detected: First-line / Second-line / Third-line /
     Neoadjuvant / Adjuvant / Perioperative / Consolidation / Maintenance
     / Induction / Salvage / Palliative / Definitive / Post-ICI.
     Distinguish induction CHEMO (leukemia) from neoadjuvant — they're
     different concepts.
   - drugs_detected: canonical drug names used or received.

5. BIOMARKER AXIS (with polarity in biomarker_status)
   - biomarkers_detected: ER, PR, HER2, PD-L1, MSI-H, TMB-H, Ki-67, CD20,
     CPS, TPS, etc.
   - genomic_alterations: EGFR L858R, KRAS G12C, BRAF V600E, ALK fusion,
     NTRK fusion, BRCA, MGMT methylation, TP53 mutation, etc.
   - imaging_detected / serum_markers_detected: tests mentioned.
   - biomarker_status: Dict[canonical, List[str]] — polarity/value for
     every marker whose status can be determined from the passage.

   ★★★ CRITICAL — STATUS COMPLETION RULES ★★★
   For EVERY entry in biomarkers_detected, genomic_alterations,
   imaging_detected, and serum_markers_detected, check the passage for a
   polarity or numeric value. If found, ADD it to biomarker_status. If
   biomarker_status is missing entries that are determinable from the
   passage, ADD them. If biomarker_status has entries with the wrong
   polarity, FIX them.

   Allowed status values:
     "positive" | "negative" | "high" | "low" | "elevated" | "rising" |
     "falling" | "intact" | "mutated" | "wild-type" | "amplified" |
     "non-amplified" | "loss" | "methylated" | "unmethylated" |
     "value:<number>[unit]"

   Specific cue patterns to look for:
     + / positive / +ve / expressed / overexpressed / detected / present
     - / negative / -ve / not expressed / not detected / absent
     high / elevated / rising / increased / ≥ / >
     low / decreased / falling / < / ≤
     mutated / mutation / mutant
     wild-type / WT / unmutated
     amplified / amplification / amp
     non-amplified / non-amp
     loss / lost / deleted / deletion
     methylated / methylation / promoter methylation
     intact (for 1p/19q intact, etc.)
     IHC scores: 0, 1+, 2+, 3+ → record as "value:2+" etc.
     FISH: positive / negative / amplified / non-amplified
     Serum values with units: "PSA 18.2 ng/mL" → "value:18.2 ng/ml"
                              "CEA 12 ng/mL" → "value:12 ng/ml"
                              "LDH elevated" → "elevated"
                              "CA 15-3 rising" → "rising"
     Ki-67 / TMB / CPS / TPS with % or threshold: "Ki-67 30%" → "value:30%"
                                                   "TPS ≥50%" → "value:≥50%"
                                                   "CPS 22" → "value:22"
     Gleason score: text "Gleason 8" should be in grades_detected.

   Example good biomarker_status when text has "HER2-low, ER+/PR+, PIK3CA
   mutated, Ki-67 30%, CA 15-3 rising":
   {
     "HER2":    ["low"],
     "ER":      ["positive"],
     "PR":      ["positive"],
     "PIK3CA":  ["mutated"],
     "Ki-67":   ["value:30%"],
     "CA 15-3": ["rising"]
   }

============================================================
RULES:
============================================================
- GROUND every tag in the passage. No invented tags.
- PRESERVE the JSON schema — every input key must appear in the output.
- KEEP canonical forms (e.g. "HER2", not "HER2+"; polarity goes in
  biomarker_status, NOT in the biomarker name).
- SUPPRESS negated concepts: "no history of breast cancer" / "ruled out
  NSCLC" → do NOT tag those cancers.
- BE ACCURATE, not conservative: if the passage clearly states a fact,
  ADD or CORRECT it even if the scanner missed.

- ★ DO NOT REMOVE TAGS THE PASSAGE SUPPORTS ★
  If a site / status / drug / biomarker is EXPLICITLY named in the
  passage, keep it. Examples of REMOVAL mistakes to AVOID:
    * Text "PSMA-PET positive (bone, pelvic lymph nodes)" → KEEP "Bone"
      and "Pelvic lymph node" in sites_detected.
    * Text "at relapse, received X" → KEEP "Recurrent".
    * Text "Pluvicto (lutetium Lu 177 vipivotide tetraxetan)" → KEEP
      BOTH the brand name AND the generic.

- ★★ DO NOT DROP biomarker_status ENTRIES PRESENT IN THE INPUT ★★
  If the input biomarker_status has an entry for a biomarker and the
  passage CONFIRMS it (even implicitly via +/- suffix, "X-negative",
  "X mutated", a numeric value, etc.), the OUTPUT must retain that
  entry. Only REMOVE an entry if the passage contradicts it or doesn't
  mention that biomarker at all.
    * Text "ER+/PR+" with input {ER: positive, PR: positive} → OUTPUT
      MUST KEEP BOTH ER: positive AND PR: positive.
    * Text "HER2-low" with input {HER2: low, HER2-low: low} → KEEP
      HER2: low (drop the HER2-low duplicate since HER2 covers it).
    * A biomarker named in biomarkers_detected MUST have a
      biomarker_status entry if its polarity/value is in the passage.

- ★★ DO NOT CROSS-CATEGORIZE ★★
  Gene-level concepts (NOTCH1, BRCA, KRAS, BRAF, EGFR, TP53, PIK3CA,
  ALK, PTEN, etc.) belong in genomic_alterations, NOT in
  biomarkers_detected. Keep each category pure:
    * biomarkers_detected: protein/IHC markers (ER, PR, HER2, PD-L1,
      Ki-67, CD20, CD30, CD19, MSI-H, TMB-H, CPS, TPS, etc.).
    * genomic_alterations: genes/fusions/mutations (NOTCH1 mutation,
      EGFR L858R, BRAF V600E, KRAS G12C, ALK fusion, BRCA2, etc.).

- INFER FROM ABBREVIATIONS ONLY WHEN THE MAPPING IS UNAMBIGUOUS:
    * "HNSCC" → histology "Squamous cell carcinoma" ✓
    * "NSCLC" → histology "Non-small cell carcinoma" ✓
    * "DLBCL" → family "Lymphoma" ✓
    * "GIST" → histology "GIST" + family "Sarcoma" ✓
    * "IDC" → "Invasive ductal carcinoma" ✓
  ★★ DO NOT INFER A SPECIFIC HISTOLOGY FROM "TNBC" ★★
  TNBC is a hormone-receptor status, NOT a histology. A TNBC tumor
  can be Invasive ductal carcinoma, metaplastic, medullary, or many
  others. If the passage says "triple-negative breast cancer" WITHOUT
  naming a histology explicitly, histologies_detected must remain
  empty for that chunk. You MAY still add:
    - family "Carcinoma" to histopathologic_types
    - disease_status "Triple negative"
  But DO NOT put "Invasive ductal carcinoma" in histologies_detected
  unless the passage explicitly contains those words.

- COMPLETE biomarker_status for EVERY detected biomarker/alteration/
  marker whose polarity/value is determinable from the passage.
- Return ONLY the corrected JSON. No prose, no markdown, no explanation."""


# ═════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════

def is_validation_worthwhile(extracted: Dict[str, Any]) -> bool:
    """
    Decide whether a chunk has enough clinical content to be worth sending
    to the LLM. Skips chunks with no detected cancers, biomarkers, drugs,
    stages, or alterations (empty chunks or non-clinical text).
    """
    signals = (
        extracted.get("cancer_types_detected")
        or extracted.get("biomarkers_detected")
        or extracted.get("genomic_alterations")
        or extracted.get("drugs_detected")
        or extracted.get("stages_detected")
        or extracted.get("tnm_detected")
        or extracted.get("histologies_detected")
        or extracted.get("disease_status_detected")
        or extracted.get("patient_demographics")
        or extracted.get("treatment_lines_detected")
    )
    return bool(signals)


def validate_and_correct(
    chunk_text: str,
    extracted: Dict[str, Any],
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_text_chars: int = 3500,
    timeout: int = 30,
    client: Any = None,
) -> Dict[str, Any]:
    """
    Validate and correct extracted metadata using an LLM.

    On any error (API failure, JSON parse error, etc.) returns the original
    `extracted` dict unchanged — this function must never break the pipeline.

    Args:
        chunk_text: The source passage that was tagged
        extracted: The output of KeywordTagger.scan_text_detailed()
        openai_api_key: API key (or reads OPENAI_API_KEY env)
        model: Defaults to gpt-4o-mini (cheap + good enough)
        max_text_chars: Truncate very long chunks to keep cost bounded
        timeout: Per-call timeout in seconds
        client: Optional pre-built openai.OpenAI client (for reuse)

    Returns:
        Dict with the same schema as `extracted`, with validated fields
        corrected. Adds `_llm_validated: True` and `_llm_corrections`
        listing field-level changes.
    """
    if not chunk_text or not extracted:
        return extracted

    # Subset the extracted dict to the fields the LLM validates
    subset = {k: extracted.get(k) for k in _VALIDATION_FIELDS if k in extracted}
    if not subset:
        return extracted

    # Truncate passage to control cost — keep the most informative head
    passage = (chunk_text or "").strip()
    if len(passage) > max_text_chars:
        passage = passage[:max_text_chars] + " [...truncated]"

    user_prompt = (
        f"PASSAGE:\n{passage}\n\n"
        f"EXTRACTED TAGS (JSON):\n{json.dumps(subset, indent=2)}\n\n"
        f"Return the CORRECTED JSON with exactly these keys: "
        f"{sorted(subset.keys())}"
    )

    try:
        if client is None:
            import openai as _openai
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("[LLM Validator] No OPENAI_API_KEY — skipping validation")
                return extracted
            client = _openai.OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=timeout,
        )
        corrected_json = resp.choices[0].message.content
        corrected = json.loads(corrected_json)
        if not isinstance(corrected, dict):
            return extracted

        # Merge corrected fields back into the original extracted dict
        result = dict(extracted)
        corrections: List[str] = []
        for key in _VALIDATION_FIELDS:
            if key not in corrected:
                continue
            before = extracted.get(key)
            after = corrected[key]
            # Normalize lists for comparison (order-independent)
            if isinstance(before, list) and isinstance(after, list):
                if set(map(str, before)) != set(map(str, after)):
                    corrections.append(f"{key}: {before} -> {after}")
                    result[key] = after
            elif isinstance(before, dict) and isinstance(after, dict):
                if before != after:
                    corrections.append(f"{key}: {before} -> {after}")
                    result[key] = after
            elif before != after:
                corrections.append(f"{key}: {before!r} -> {after!r}")
                result[key] = after

        # ── Defensive post-processing: reconcile biomarker_status ──
        # gpt-4o-mini sometimes drops biomarker_status entries even when
        # the biomarker is still in biomarkers_detected / genomic_alterations.
        # If the regex had a status for a biomarker that's still present,
        # restore it (the LLM's drop is almost always a mistake — the regex
        # had it right).
        restored = _reconcile_biomarker_status(
            extracted, result,
        )
        if restored:
            corrections.append(
                f"biomarker_status: restored {sorted(restored)} dropped by LLM"
            )

        # ── Defensive post-processing: move genes out of biomarkers ──
        # gpt-4o-mini sometimes duplicates gene names into both
        # biomarkers_detected AND genomic_alterations. Genes should only
        # be in genomic_alterations; move them.
        moved = _dedupe_genes_from_biomarkers(result)
        if moved:
            corrections.append(
                f"biomarkers_detected: removed gene-level {sorted(moved)} "
                "(moved to genomic_alterations)"
            )

        result["_llm_validated"] = True
        if corrections:
            result["_llm_corrections"] = corrections
        return result

    except Exception as e:
        # Never break the pipeline — log and fall through
        print(f"[LLM Validator] Error validating chunk (continuing with "
              f"regex output): {type(e).__name__}: {e}")
        return extracted


# ═════════════════════════════════════════════════════════════════════════
# Defensive post-processors (run after the LLM response)
# ═════════════════════════════════════════════════════════════════════════

# Gene-level names that should live in genomic_alterations, not
# biomarkers_detected. Case-insensitive.
_GENE_NAMES_LOWER = {
    "egfr", "alk", "ros1", "ret", "ntrk", "ntrk1", "ntrk2", "ntrk3",
    "braf", "kras", "nras", "hras", "met", "her2", "erbb2",  # HER2 is
    # dual (protein/IHC AND gene); keep in biomarkers_detected when
    # measured as protein, move to alterations only if obviously gene-
    # level. For simplicity we KEEP HER2/ERBB2 in biomarkers since most
    # oncology papers reference it as a protein status.
    "tp53", "pten", "pik3ca", "atm", "rb1", "cdkn2a", "smad4",
    "notch1", "arid1a", "stk11", "vhl", "pbrm1", "setd2", "bap1",
    "fgfr1", "fgfr2", "fgfr3", "idh1", "idh2", "mgmt", "nf1", "nf2",
    "ptch1", "smo", "ctnnb1", "apc", "gnas", "med12", "spop",
    "palb2", "cdh1", "myc", "myod1", "wt1", "mycn",
    "kit", "pdgfra", "dog1",  # GIST markers that are gene-level
    "ewsr1", "ss18", "ssx", "fli1", "eml4",
    "bcl2", "bcl6", "ccnd1", "myd88", "ezh2",
    "ar-v7",  # splice variant
}
# These ARE protein-level biomarkers even though they have gene names:
_PROTEIN_BIOMARKERS_LOWER = {
    "her2", "erbb2", "er", "pr", "ki-67", "ki67", "mib-1", "ar",
    "pd-l1", "pd-1", "cd19", "cd20", "cd30", "cd117", "cd15",
    "msi-h", "mss", "dmmr", "pmmr", "tmb-h", "hrd", "cps", "tps",
    "s100", "hmb-45", "melan-a", "gfap", "synaptophysin",
    "vimentin", "cyclin d1", "sma", "cd99", "claudin 18.2",
    "nectin-4",
}


def _reconcile_biomarker_status(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Set[str]:
    """
    If the LLM dropped biomarker_status entries for markers that are still
    in biomarkers_detected / genomic_alterations / etc., restore them from
    the original regex output.

    Returns the set of restored keys.
    """
    orig_status = before.get("biomarker_status") or {}
    new_status = dict(after.get("biomarker_status") or {})

    # Collect the set of detected names (lowercased) across all slots
    detected_lower: Set[str] = set()
    for slot in ("biomarkers_detected", "genomic_alterations",
                 "serum_markers_detected", "imaging_detected"):
        for item in after.get(slot) or []:
            if isinstance(item, str):
                detected_lower.add(item.lower())
                # Also add the gene root (e.g. "BRCA2 mutation" → "brca2")
                first_word = item.split()[0].lower() if item.split() else ""
                if first_word:
                    detected_lower.add(first_word)

    new_status_lower = {k.lower() for k in new_status}
    restored: Set[str] = set()

    for orig_key, orig_val in orig_status.items():
        if not isinstance(orig_key, str):
            continue
        orig_key_lower = orig_key.lower()
        # Already in the LLM output? Skip.
        if orig_key_lower in new_status_lower:
            continue
        # Is the biomarker still detected somewhere? If so, restore.
        first_word = orig_key.split()[0].lower() if orig_key.split() else ""
        if orig_key_lower in detected_lower or first_word in detected_lower:
            new_status[orig_key] = orig_val
            restored.add(orig_key)

    after["biomarker_status"] = new_status
    return restored


def _dedupe_genes_from_biomarkers(after: Dict[str, Any]) -> Set[str]:
    """
    Remove gene-level names from biomarkers_detected when they're also
    (or should be) in genomic_alterations. Protein markers (HER2, ER, PR,
    CD20, PD-L1, etc.) are exempted.

    Returns the set of names that were moved/removed.
    """
    biomarkers = list(after.get("biomarkers_detected") or [])
    alterations = list(after.get("genomic_alterations") or [])
    alterations_lower = {a.lower() for a in alterations if isinstance(a, str)}

    moved: Set[str] = set()
    kept: List[str] = []
    for b in biomarkers:
        if not isinstance(b, str):
            kept.append(b)
            continue
        bl = b.lower()
        # Keep protein-level biomarkers even if they share a gene name
        if bl in _PROTEIN_BIOMARKERS_LOWER:
            kept.append(b)
            continue
        # If it's a gene-level name, remove from biomarkers.
        # It can already be in alterations (dedup) or we append it.
        if bl in _GENE_NAMES_LOWER or bl in alterations_lower:
            moved.add(b)
            # Make sure it's represented in alterations (as-is)
            if bl not in alterations_lower:
                alterations.append(b)
                alterations_lower.add(bl)
            continue
        kept.append(b)

    if moved:
        after["biomarkers_detected"] = kept
        after["genomic_alterations"] = sorted(set(alterations))
    return moved


# ═════════════════════════════════════════════════════════════════════════
# Batch / parallel helpers
# ═════════════════════════════════════════════════════════════════════════

def validate_many(
    chunks: List[Tuple[str, Dict[str, Any]]],
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_workers: int = 10,
    skip_empty: bool = True,
    progress_every: int = 100,
) -> List[Dict[str, Any]]:
    """
    Validate many (chunk_text, extracted) pairs concurrently.

    Returns a list of corrected dicts in the same order as input.
    Chunks where `is_validation_worthwhile` returns False are returned
    unchanged (no LLM call), to save cost.

    Args:
        chunks: List of (chunk_text, extracted_dict) tuples
        openai_api_key: API key
        model: gpt-4o-mini by default
        max_workers: concurrent LLM calls
        skip_empty: skip chunks with no clinical content
        progress_every: print progress every N chunks
    """
    import openai as _openai
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[LLM Validator] No OPENAI_API_KEY — returning originals unchanged")
        return [extracted for _, extracted in chunks]

    client = _openai.OpenAI(api_key=api_key)

    results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
    to_validate: List[int] = []
    for i, (text, extracted) in enumerate(chunks):
        if skip_empty and not is_validation_worthwhile(extracted):
            results[i] = extracted  # unchanged
        else:
            to_validate.append(i)

    print(f"[LLM Validator] {len(to_validate)} / {len(chunks)} chunks will be validated "
          f"(skipping {len(chunks) - len(to_validate)} empty)")

    t_start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                validate_and_correct,
                chunks[i][0],
                chunks[i][1],
                openai_api_key=api_key,
                model=model,
                client=client,
            ): i
            for i in to_validate
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"[LLM Validator] Chunk {i} failed: {e}")
                results[i] = chunks[i][1]
            done += 1
            if done % progress_every == 0:
                rate = done / (time.time() - t_start + 0.001)
                eta = (len(to_validate) - done) / max(rate, 0.001)
                print(f"  [{done}/{len(to_validate)}] rate={rate:.1f}/s "
                      f"eta={eta:.0f}s")

    # Fill any remaining None (shouldn't happen)
    for i, r in enumerate(results):
        if r is None:
            results[i] = chunks[i][1]
    return results  # type: ignore[return-value]
