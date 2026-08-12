"""
FastAPI Routes for Patient Matching

Provides endpoint for matching patients to clinical studies.
Supports both structured (form) and unstructured (free-text) input.
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.models.query_models import (
    PatientProfile,
    UnstructuredPatientMatchRequest,
    StudyMatch,
    PatientMatchResponse,
)
from src.api.services.auth_dependencies import get_current_user_optional
from src.api.services.cache_service import get_cache_service, make_cache_key

router = APIRouter(prefix="/rag")


# Human-readable labels for structured matcher criteria
_CRITERIA_LABELS = {
    "cancer_site": "Cancer site",
    "site_detail": "Specific site",
    "histology": "Histology",
    "stage": "Stage",
    "tnm_t": "T stage",
    "tnm_n": "N stage",
    "treatment": "Treatment",
    "age_range": "Age",
    "performance_status": "Performance status",
}

# Structured matcher key values to readable names
_SITE_LABELS = {
    "head_neck": "Head & Neck", "breast": "Breast", "lung": "Lung",
    "prostate": "Prostate", "gi": "GI", "gyn": "GYN", "gu": "GU",
    "cns": "CNS", "lymphoma": "Lymphoma", "sarcoma": "Sarcoma",
    "skin": "Skin", "thyroid": "Thyroid",
}
_HISTOLOGY_LABELS = {
    "scc": "Squamous cell carcinoma", "adenocarcinoma": "Adenocarcinoma",
    "small_cell": "Small cell", "large_cell": "Large cell",
    "clear_cell": "Clear cell", "transitional": "Transitional/Urothelial",
    "ductal": "Ductal", "lobular": "Lobular",
    "neuroendocrine": "Neuroendocrine", "melanoma": "Melanoma",
}
_TREATMENT_LABELS = {
    "radiation": "Radiation", "chemotherapy": "Chemotherapy",
    "surgery": "Surgery", "immunotherapy": "Immunotherapy",
    "targeted": "Targeted therapy", "hormonal": "Hormonal therapy",
}


def _format_matched_criteria(raw_criteria: list) -> dict:
    """
    Convert structured matcher criteria like 'cancer_site=head_neck'
    into human-readable tags split into demographics/cancer_characteristics/key_matches.
    """
    demographics = []
    cancer_chars = []
    key_matches = []

    for criterion in raw_criteria:
        if "=" not in criterion:
            key_matches.append(criterion)
            continue
        key, value = criterion.split("=", 1)

        if key == "cancer_site":
            cancer_chars.append(f"Site: {_SITE_LABELS.get(value, value)}")
        elif key == "site_detail":
            cancer_chars.append(f"Site: {value.replace('_', ' ').title()}")
        elif key == "histology":
            cancer_chars.append(_HISTOLOGY_LABELS.get(value, value))
        elif key == "stage":
            cancer_chars.append(f"Stage {value}")
        elif key.startswith("tnm_"):
            cancer_chars.append(value.upper())
        elif key == "treatment":
            key_matches.append(_TREATMENT_LABELS.get(value, value))
        elif key.startswith("age"):
            demographics.append(f"Age ~{value}")
        elif key == "perf_status":
            demographics.append(value)
        else:
            key_matches.append(f"{_CRITERIA_LABELS.get(key, key)}: {value}")

    return {
        "demographics": demographics,
        "cancer_characteristics": cancer_chars,
        "key_matches": key_matches,
    }


@router.post("/patient/match", response_model=PatientMatchResponse)
async def match_patient(
    patient_profile: PatientProfile,
    current_user: dict = Depends(get_current_user_optional),
):
    """
    Match patient characteristics to relevant clinical studies.
    
    **Enhanced multi-stage filtering workflow:**
    - Stage 1: Semantic query construction from patient profile
    - Stage 2: Category-based filtering (cancer type)
    - Stage 3: Vector search with embedding caching
    - Stage 4: Cross-encoder reranking
    - Stage 5: LLM-based semantic validation
    - Stage 6: Document deduplication and scoring
    
    **Patient Characteristics:**
    - age: Patient age
    - gender: Gender (male, female)
    - cancer_type: Type of cancer
    - cancer_stage: Cancer stage (I, II, III, IV)
    - histology: Histology type
    - molecular_markers: List of molecular markers (e.g., EGFR+, PD-L1+)
    - performance_status: Performance status (ECOG 0, 1, 2, etc.)
    - comorbidities: List of comorbidities
    - smoking_status: Smoking status
    
    **Returns:**
    - List of matching studies with match scores
    - Match reasons categorized by: demographics, cancer_characteristics, key_matches
    - Treatment information and key findings extracted from each study
    """
    try:
        from qdrant_client import QdrantClient
        from openai import OpenAI
        from src.core.config import settings
        from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
        from src.api.services.postgres_study_details_service import PostgresStudyDetailsService
        
        cache_service = get_cache_service()
        
        # Clear old patient matching cache to ensure new logic is used
        # This is a one-time migration - can be removed after deployment
        if current_user:
            await cache_service.clear_user_cache(current_user["id"], "patient_match")
            print(f"[Patient Matching] Cleared cache for user {current_user['id']}")

        # Create clients
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60
        )
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Create patient matching service
        service = SimplePatientMatchingService(
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            collection_name=settings.qdrant_collection,
            embed_model=settings.embed_model
        )
        
        # Build profile dict from Pydantic model
        profile_dict = {
            "cancer_type": patient_profile.cancer_type,
            "anatomical_site": patient_profile.anatomical_site,
            "age": patient_profile.age,
            "gender": patient_profile.gender,
            "cancer_stage": patient_profile.cancer_stage,
            "histology": patient_profile.histology,
            "molecular_markers": patient_profile.molecular_markers,
            "performance_status": patient_profile.performance_status,
            "comorbidities": patient_profile.comorbidities,
            "smoking_status": patient_profile.smoking_status,
            "prior_treatments": patient_profile.prior_treatments,
            "tnm_t": patient_profile.tnm_t,
            "tnm_n": patient_profile.tnm_n,
            "tnm_m": patient_profile.tnm_m,
            "recurrence_status": patient_profile.recurrence_status,
            "treatment_setting": patient_profile.treatment_setting,
            "lvi": patient_profile.lvi,
            "pni": patient_profile.pni,
            "margins": patient_profile.margins,
            "disease_descriptor": patient_profile.disease_descriptor,
        }
        # Remove None values
        profile_dict = {k: v for k, v in profile_dict.items() if v is not None}
        
        # Infer stage from TNM if not explicitly provided
        # DISABLED: Stage inference temporarily disabled
        # if not profile_dict.get("cancer_stage") and (
        #     profile_dict.get("tnm_t") or profile_dict.get("tnm_n") or profile_dict.get("tnm_m")
        # ):
        #     try:
        #         from src.api.services.stage_inference_service import infer_stage_for_query
        #         inference = infer_stage_for_query(
        #             cancer_type=profile_dict.get("cancer_type"),
        #             cancer_location=profile_dict.get("anatomical_site"),
        #             tnm_t=profile_dict.get("tnm_t"),
        #             tnm_n=profile_dict.get("tnm_n"),
        #             tnm_m=profile_dict.get("tnm_m"),
        #             metastatic_status=profile_dict.get("disease_descriptor"),
        #             age=profile_dict.get("age"),
        #         )
        #         if inference.stage_group:
        #             profile_dict["cancer_stage"] = inference.stage_group
        #             print(f"[Patient Matching] Stage inferred from TNM: {inference.tnm_used} -> Stage {inference.stage_group} ({inference.confidence})")
        #         elif inference.possible_stages:
        #             print(f"[Patient Matching] Stage ambiguous from TNM: {inference.tnm_used} -> possible {inference.possible_stages}")
        #     except Exception as e:
        #         print(f"[Patient Matching] Stage inference failed: {e}")
        
        # Build query structure for structured PostgreSQL matching
        from src.api.services.patient_matching_service_simple import build_query_structure_from_profile
        from src.api.services.structured_study_matcher import (
            match_studies_by_structure,
            boost_candidates_with_structured_matches,
        )
        
        query_structure = build_query_structure_from_profile(profile_dict)
        print(f"[Patient Matching] Query structure: {query_structure}")
        
        # Extract user criteria weights if provided
        criteria_weights = patient_profile.criteria_weights
        if criteria_weights:
            print(f"[Patient Matching] User criteria weights: {criteria_weights}")
        
        # Build the clinical_profile once — used below to score PG-only matches
        # with the canonical patient_match_scorer (closes the patient_match=None
        # gap that the multi-specialty path's scorer call doesn't cover).
        from src.api.services.patient_matching_service_simple import (
            _build_clinical_profile_from_patient_dict,
            score_pg_only_match,
        )
        clinical_profile_for_pg = _build_clinical_profile_from_patient_dict(profile_dict)

        # Run Qdrant matching and structured PostgreSQL matching in parallel
        import asyncio
        structured_task = None
        if query_structure:
            structured_task = match_studies_by_structure(query_structure, limit=50, user_weights=criteria_weights)
        
        if structured_task:
            qdrant_result, structured_result = await asyncio.gather(
                service.match_patient_comprehensive(profile_dict, top_k=15),
                structured_task,
            )
        else:
            qdrant_result = await service.match_patient_comprehensive(profile_dict, top_k=15)
            structured_result = None

        result = qdrant_result
        
        # Boost Qdrant matches with structured PostgreSQL scores
        if structured_result and structured_result.doc_ids:
            print(f"[Patient Matching] Structured matcher found {len(structured_result.doc_ids)} studies, boosting Qdrant candidates")
            
            # Boost existing Qdrant matches that also matched in PostgreSQL
            qdrant_matches = result.get("matches", [])
            existing_dois = set()
            for match in qdrant_matches:
                doc_id = match.get("doc_id")
                if doc_id and doc_id in structured_result.doc_ids:
                    pg_score = structured_result.match_scores.get(doc_id, 0.0)
                    # Blend: 50% Qdrant score + 50% structured score
                    original_score = match.get("match_score", 0.0)
                    match["match_score"] = round(original_score * 0.5 + pg_score * 0.5, 3)
                    match["_structured_match_score"] = pg_score
                    print(f"[Patient Matching] Boosted '{match.get('title', '')[:40]}': {original_score:.3f} -> {match['match_score']:.3f} (pg={pg_score:.2f})")
                elif doc_id:
                    # Study NOT in structured results — light penalty (semantic match is still valid)
                    original_score = match.get("match_score", 0.0)
                    match["match_score"] = round(original_score * 0.85, 3)
                    match["_structured_match_score"] = 0.0
                if match.get("doi"):
                    existing_dois.add(match["doi"])
            
            # Add PostgreSQL-only matches (not already in Qdrant results)
            try:
                from src.api.services.postgres_study_details_service import PostgresStudyDetailsService
                pg_service = PostgresStudyDetailsService()
                # Only fetch enough PG-only studies to fill remaining top_k slots
                pg_slots = max(0, 15 - len(qdrant_matches))

                for doc_id, pg_score in sorted(structured_result.match_scores.items(), key=lambda x: x[1], reverse=True):
                    if pg_slots <= 0:
                        break
                    if any(m.get("doc_id") == doc_id for m in qdrant_matches):
                        continue

                    try:
                        pg_study = await pg_service.get_study_details(doc_id=doc_id)
                        if not pg_study or pg_study.get("error"):
                            continue
                        doi = pg_study.get("doi")
                        if doi and doi in existing_dois:
                            continue

                        details = structured_result.match_details.get(doc_id, {})
                        study_details = pg_study.get("study_details", {})
                        diagnosis = pg_study.get("diagnosis", {})
                        formatted = _format_matched_criteria(details.get("matched_criteria", []))

                        cancer_chars = list(formatted["cancer_characteristics"])
                        for label_key, diag_key in [("Site", "Cancer Location"), ("Type", "Cancer Type"), ("Histology", "Histopathologic Type")]:
                            val = diagnosis.get(diag_key, {}).get("value")
                            if val and not any(label_key in c for c in cancer_chars):
                                cancer_chars.append(f"{label_key}: {val}")

                        pg_match = {
                            "title": pg_study.get("title") or "Unknown",
                            "author": None,
                            "doi": doi,
                            "year": int(str(study_details.get("Publication Date", {}).get("value", ""))[:4]) if study_details.get("Publication Date", {}).get("value") else None,
                            "match_score": round(pg_score, 3),
                            "treatment": "",
                            "demographics": formatted["demographics"],
                            "cancer_characteristics": cancer_chars,
                            "key_matches": formatted["key_matches"],
                            "key_info": pg_study.get("outcomes", {}).get("Primary Endpoint", {}).get("value", ""),
                            "relevant_text": f"Study type: {study_details.get('Study Type', {}).get('value', 'N/A')}. Patients: {study_details.get('Number of Patients', {}).get('value', 'N/A')}.",
                            "source": "postgres_structured",
                            "doc_id": doc_id,
                        }

                        # Score this PG-only match with the canonical patient_match_scorer
                        # so it carries patient_match_score / breakdown in the response,
                        # matching how multi-specialty matches are scored. This also
                        # gives the UI's "axes scored: N/7" tooltip something to show
                        # for PG matches. Pass `details` (from match_details) too —
                        # the matcher's own cancer_type/cancer_location strings are
                        # populated for these doc_ids even when pg_study.diagnosis
                        # comes back empty (different DB shape between matcher and
                        # details-service).
                        pm_score, pm_breakdown = await score_pg_only_match(
                            qdrant_client,
                            settings.qdrant_collection,
                            doc_id,
                            clinical_profile_for_pg,
                            pg_study_fallback=pg_study,
                            match_details=details,
                        )
                        if pm_score is not None:
                            pg_match["patient_match_score"] = pm_score
                            pg_match["patient_match_breakdown"] = pm_breakdown

                        qdrant_matches.append(pg_match)
                        if doi:
                            existing_dois.add(doi)
                        pg_slots -= 1
                    except Exception:
                        continue

                await pg_service.close()
            except Exception as pg_error:
                print(f"[Patient Matching] PostgreSQL detail fetch failed: {pg_error}")
            
            # Re-sort all matches by score and cap
            qdrant_matches.sort(key=lambda m: m.get("match_score", 0), reverse=True)
            result["matches"] = qdrant_matches[:15]
            result["total_matches"] = len(qdrant_matches)

        # Web fallback — DISABLED pending proper structured matching for
        # external trials (CT.gov/PubMed). Will be rebuilt with abstract
        # retrieval + structured profile extraction + weighted matching.
        # kb_scores = [m.get("match_score", 0) for m in result.get("matches", [])[:5]]
        # kb_avg = sum(kb_scores) / len(kb_scores) if kb_scores else 0
        # if kb_avg < 0.55:
        #     ct_matches = _fetch_ct_fallback_matches(profile_dict, n=5)
        #     ...

        matches = _result_to_study_matches(result)

        response = PatientMatchResponse(
            matches=matches,
            total_matches=result.get("total_matches", len(matches)),
            patient_summary=result.get("patient_summary", "Patient"),
        )
        # Note: Caching disabled temporarily to ensure new matching logic is used
        # Can re-enable after confirming the new logic works correctly
        return response

    except Exception as e:
        import traceback
        error_detail = f"Error matching patient: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


def _extract_treatment_from_pg(pg_study: dict) -> str:
    """Extract treatment information from PostgreSQL study result."""
    treatments = []
    
    # Check study type for treatment hints
    study_type = (pg_study.get("study_type") or "").lower()
    if "radiation" in study_type or "radiotherapy" in study_type:
        treatments.append("Radiotherapy")
    if "chemo" in study_type:
        treatments.append("Chemotherapy")
    if "surgery" in study_type or "surgical" in study_type:
        treatments.append("Surgery")
    if "immuno" in study_type:
        treatments.append("Immunotherapy")
    
    # Check primary endpoint for treatment hints
    endpoint = (pg_study.get("primary_endpoint") or "").lower()
    if "radiation" in endpoint or "rt" in endpoint:
        if "Radiotherapy" not in treatments:
            treatments.append("Radiotherapy")
    if "chemo" in endpoint:
        if "Chemotherapy" not in treatments:
            treatments.append("Chemotherapy")
    
    if treatments:
        return ", ".join(treatments)
    return "Treatment information available in study details"


def _fetch_ct_fallback_matches(profile_dict: dict, n: int = 5) -> list:
    """
    Fetch ClinicalTrials.gov matches when KB results are sparse.

    Builds a plain-text query from the patient profile and returns up to n
    trial dicts formatted like patient-matching match dicts, tagged
    source_type="clinicaltrials".
    """
    try:
        from src.api.services.literature_search_service import LiteratureSearchService
        svc = LiteratureSearchService()

        # Build query from profile
        parts = []
        cancer_type = profile_dict.get("cancer_type")
        # Fall back to anatomical_site if cancer_type is missing
        if not cancer_type and profile_dict.get("anatomical_site"):
            cancer_type = f"{profile_dict['anatomical_site']} cancer"
        if cancer_type:
            parts.append(cancer_type)
        if profile_dict.get("histology"):
            parts.append(profile_dict["histology"])
        if profile_dict.get("cancer_stage"):
            parts.append(f"stage {profile_dict['cancer_stage']}")
        markers = profile_dict.get("molecular_markers") or []
        if markers:
            parts.append(" ".join(markers[:3]))
        prior = profile_dict.get("prior_treatments") or []
        if prior:
            parts.append(" ".join(prior[:2]))

        query = " ".join(parts) if parts else "oncology"
        condition = cancer_type

        trials = svc.search_clinical_trials_structured(
            condition=condition,
            other_terms=" ".join(parts[1:]) if len(parts) > 1 else None,
            max_results=n * 2,
        )
        if not trials:
            trials = svc.search_clinical_trials_by_query(query, max_results=n * 2)

        matches = []
        for t in trials[:n]:
            proto = t.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            desc = proto.get("descriptionModule", {})
            contacts = proto.get("contactsLocationsModule", {})

            nct_id = ident.get("nctId", "")
            title = ident.get("briefTitle", "Unknown Trial")
            status = status_mod.get("overallStatus", "Unknown")
            phases = design.get("phases", [])
            phase_str = ", ".join(phases) if phases else "N/A"
            enrollment = (design.get("enrollmentInfo") or {}).get("count")
            summary = (desc.get("briefSummary") or "")[:400]

            matches.append({
                "title": title,
                "author": None,
                "doi": None,
                "pmid": None,
                "nct_id": nct_id,
                "year": None,
                "match_score": 0.45,  # Fixed score — online source, relevance unverified
                "treatment": phase_str,
                "demographics": [],
                "cancer_characteristics": [status, f"Enrollment: {enrollment}"] if enrollment else [status],
                "key_matches": [f"NCT: {nct_id}"] if nct_id else [],
                "key_info": summary,
                "relevant_text": summary,
                "source_type": "clinicaltrials",
                "source": "clinicaltrials_web",
            })

        print(f"[Patient Matching] CT.gov fallback: {len(matches)} trials fetched for '{query}'")
        return matches
    except Exception as e:
        print(f"[Patient Matching] CT.gov fallback failed: {e}")
        return []


def _result_to_study_matches(
    result: dict,
    rationales: list = None,
) -> list:
    """Convert service result to list of StudyMatch, optionally with rationale/strength."""
    matches = []
    raw_matches = result.get("matches", [])
    for i, match in enumerate(raw_matches):
        demographics = match.get("demographics", [])
        cancer_chars = match.get("cancer_characteristics", [])
        key_matches = match.get("key_matches", [])
        match_reasons = []
        if demographics:
            match_reasons.extend([f"Demographics: {d}" for d in demographics])
        if cancer_chars:
            match_reasons.extend([f"Cancer: {c}" for c in cancer_chars])
        if key_matches:
            match_reasons.extend([f"Marker: {k}" for k in key_matches])
        if not match_reasons:
            match_reasons = ["Semantic similarity to patient profile"]

        rationale = None
        strength = None
        if rationales and i < len(rationales):
            rationale = rationales[i].get("rationale")
            strength = rationales[i].get("strength")

        matches.append(
            StudyMatch(
                title=match.get("title", "Unknown"),
                author=match.get("author"),
                citation=match.get("citation"),
                doi=match.get("doi"),
                pmid=match.get("pmid"),
                doc_id=match.get("doc_id"),
                year=match.get("year"),
                match_score=match.get("match_score", 0.0),
                confidence=match.get("match_score", 0.0),
                match_reasons=match_reasons,
                match_rationale=rationale,
                match_strength=strength,
                relevant_text=match.get("relevant_text", ""),
                treatment=match.get("treatment"),
                key_info=match.get("key_info"),
                demographics=demographics,
                cancer_characteristics=cancer_chars,
                key_matches=key_matches,
                source_type=match.get("source_type"),
                # Pass through patient_match_scorer output from
                # match_patient_comprehensive (stamped on each chunk
                # before _build_match_data). The UI uses this to render
                # the same Strong/Moderate/Weak/Limited bucketed badge
                # the chat path uses for per-source clinical fit.
                patient_match_score=match.get("patient_match_score"),
                patient_match_breakdown=match.get("patient_match_breakdown"),
            )
        )
    return matches


@router.post("/patient/match/unstructured", response_model=PatientMatchResponse)
async def match_patient_unstructured(
    request: UnstructuredPatientMatchRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    """
    Match patient from free-text description to clinical studies.
    
    Enhanced with conversation context support:
    - If cached_profile is provided, skips LLM extraction (saves ~1-2s)
    - If conversation_context is provided, uses doc_ids for retrieval boosting
    
    Returns interpretable match rationales and strength (strong/partial/possible).
    """
    import time
    start_time = time.time()
    
    try:
        from qdrant_client import QdrantClient
        from openai import OpenAI
        from src.core.config import settings
        from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
        from src.api.services.unstructured_patient_extractor import extract_patient_profile
        from src.api.services.staging_inference_validator import validate_and_correct_staging
        from src.api.services.match_rationale_generator import generate_rationales

        cache_service = get_cache_service()
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Check if we have a cached profile from conversation context
        profile_dict = None
        if request.cached_profile:
            # Use cached profile - skip LLM extraction entirely
            profile_dict = request.cached_profile
            print(f"[Patient Matching] Using cached profile (skipped LLM extraction)")
        else:
            # Extract profile from unstructured text via LLM
            t1 = time.time()
            profile_dict = extract_patient_profile(request.unstructured_description, openai_client)
            print(f"[Patient Matching] Profile extraction took {time.time() - t1:.2f}s")

        if not profile_dict:
            raise HTTPException(
                status_code=400,
                detail="Could not extract patient characteristics from the description. Please add cancer type, stage, or other clinical details.",
            )
        
        # Validate and correct staging using AJCC rules (DOI for T stage, ENE for N stage)
        profile_dict = validate_and_correct_staging(profile_dict, request.unstructured_description)
        
        print(f"[Patient Matching] Extracted profile: {profile_dict}")

        # Extract doc_ids from conversation context for boosting
        context_doc_ids = []
        if request.conversation_context:
            for entry in request.conversation_context:
                if hasattr(entry, 'doc_ids') and entry.doc_ids:
                    context_doc_ids.extend(entry.doc_ids)
                elif isinstance(entry, dict) and entry.get('doc_ids'):
                    context_doc_ids.extend(entry['doc_ids'])
            context_doc_ids = list(set(context_doc_ids))  # Deduplicate
            if context_doc_ids:
                print(f"[Patient Matching] Found {len(context_doc_ids)} doc_ids from conversation context for boosting")

        t2 = time.time()
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60,
        )
        service = SimplePatientMatchingService(
            qdrant_client=qdrant_client,
            openai_client=openai_client,
            collection_name=settings.qdrant_collection,
            embed_model=settings.embed_model,
        )

        result = await service.match_patient_comprehensive(profile_dict, top_k=request.top_k)
        print(f"[Patient Matching] Comprehensive matching took {time.time() - t2:.2f}s")
        patient_summary = result.get("patient_summary", "Patient")
        
        # Boost matches that appear in conversation context
        if context_doc_ids and result.get("matches"):
            for match in result["matches"]:
                if match.get("doc_id") in context_doc_ids:
                    original_score = match.get("match_score", 0.0)
                    match["match_score"] = min(0.95, original_score * 1.15)  # 15% boost, cap at 95%
                    match["_context_boosted"] = True
                    print(f"[Patient Matching] Context-boosted: {match.get('title', '')[:40]}")
        
        # Run structured PostgreSQL matching in parallel-style (Qdrant already done above)
        t3 = time.time()
        from src.api.services.patient_matching_service_simple import (
            build_query_structure_from_profile,
            _build_clinical_profile_from_patient_dict,
            score_pg_only_match,
        )
        from src.api.services.structured_study_matcher import match_studies_by_structure

        query_structure = build_query_structure_from_profile(profile_dict)
        print(f"[Patient Matching Unstructured] Query structure: {query_structure}")

        # Clinical profile for the patient_match_scorer call below (PG-only path)
        clinical_profile_for_pg = _build_clinical_profile_from_patient_dict(profile_dict)
        
        # Extract user criteria weights if provided
        criteria_weights = request.criteria_weights
        if criteria_weights:
            print(f"[Patient Matching Unstructured] User criteria weights: {criteria_weights}")
        
        structured_result = None
        if query_structure:
            try:
                structured_result = await match_studies_by_structure(query_structure, limit=50, user_weights=criteria_weights)
            except Exception as struct_err:
                print(f"[Patient Matching Unstructured] Structured matching failed: {struct_err}")
        
        # Boost Qdrant matches with structured PostgreSQL scores
        if structured_result and structured_result.doc_ids:
            print(f"[Patient Matching Unstructured] Structured matcher found {len(structured_result.doc_ids)} studies")
            
            qdrant_matches = result.get("matches", [])
            existing_dois = set()
            for match in qdrant_matches:
                doc_id = match.get("doc_id")
                if doc_id and doc_id in structured_result.doc_ids:
                    pg_score = structured_result.match_scores.get(doc_id, 0.0)
                    original_score = match.get("match_score", 0.0)
                    match["match_score"] = round(original_score * 0.5 + pg_score * 0.5, 3)
                    match["_structured_match_score"] = pg_score
                elif doc_id:
                    # Light penalty — semantic match is still valid evidence
                    original_score = match.get("match_score", 0.0)
                    match["match_score"] = round(original_score * 0.85, 3)
                    match["_structured_match_score"] = 0.0
                if match.get("doi"):
                    existing_dois.add(match["doi"])
            
            # Add PostgreSQL-only matches — cap fetches to fill remaining top_k slots
            try:
                from src.api.services.postgres_study_details_service import PostgresStudyDetailsService
                pg_service = PostgresStudyDetailsService()
                pg_slots = max(0, request.top_k - len(qdrant_matches))

                for doc_id, pg_score in sorted(structured_result.match_scores.items(), key=lambda x: x[1], reverse=True):
                    if pg_slots <= 0:
                        break
                    if any(m.get("doc_id") == doc_id for m in qdrant_matches):
                        continue
                    try:
                        pg_study = await pg_service.get_study_details(doc_id=doc_id)
                        if not pg_study or pg_study.get("error"):
                            continue
                        doi = pg_study.get("doi")
                        if doi and doi in existing_dois:
                            continue

                        details = structured_result.match_details.get(doc_id, {})
                        study_details = pg_study.get("study_details", {})
                        diagnosis = pg_study.get("diagnosis", {})
                        formatted = _format_matched_criteria(details.get("matched_criteria", []))

                        cancer_chars = list(formatted["cancer_characteristics"])
                        for label_key, diag_key in [("Site", "Cancer Location"), ("Type", "Cancer Type"), ("Histology", "Histopathologic Type")]:
                            val = diagnosis.get(diag_key, {}).get("value")
                            if val and not any(label_key in c for c in cancer_chars):
                                cancer_chars.append(f"{label_key}: {val}")

                        pg_match = {
                            "title": pg_study.get("title") or "Unknown",
                            "author": None,
                            "doi": doi,
                            "year": int(str(study_details.get("Publication Date", {}).get("value", ""))[:4]) if study_details.get("Publication Date", {}).get("value") else None,
                            "match_score": round(pg_score, 3),
                            "treatment": "",
                            "demographics": formatted["demographics"],
                            "cancer_characteristics": cancer_chars,
                            "key_matches": formatted["key_matches"],
                            "key_info": pg_study.get("outcomes", {}).get("Primary Endpoint", {}).get("value", ""),
                            "relevant_text": f"Study type: {study_details.get('Study Type', {}).get('value', 'N/A')}. Patients: {study_details.get('Number of Patients', {}).get('value', 'N/A')}.",
                            "source": "postgres_structured",
                            "doc_id": doc_id,
                        }

                        # Score this PG-only match with the canonical patient_match_scorer
                        # so it carries patient_match_score / breakdown in the response,
                        # matching how multi-specialty matches are scored. The scorer's
                        # `stages` axis naturally penalises early-stage trials when the
                        # patient is metastatic, so stage-mismatched PG matches drop in
                        # rank instead of appearing alongside well-matched studies.
                        # Pass `details` (from match_details) too — the matcher's own
                        # cancer_type/cancer_location strings are populated for these
                        # doc_ids even when pg_study.diagnosis comes back empty.
                        pm_score, pm_breakdown = await score_pg_only_match(
                            qdrant_client,
                            settings.qdrant_collection,
                            doc_id,
                            clinical_profile_for_pg,
                            pg_study_fallback=pg_study,
                            match_details=details,
                        )
                        if pm_score is not None:
                            pg_match["patient_match_score"] = pm_score
                            pg_match["patient_match_breakdown"] = pm_breakdown

                        qdrant_matches.append(pg_match)
                        if doi:
                            existing_dois.add(doi)
                        pg_slots -= 1
                    except Exception:
                        continue

                await pg_service.close()
            except Exception as pg_error:
                print(f"[Patient Matching Unstructured] PostgreSQL detail fetch failed: {pg_error}")

            qdrant_matches.sort(key=lambda m: m.get("match_score", 0), reverse=True)
            result["matches"] = qdrant_matches[:request.top_k]
            result["total_matches"] = len(qdrant_matches)
        
        print(f"[Patient Matching] PostgreSQL matching took {time.time() - t3:.2f}s")

        # Web fallback — DISABLED pending proper structured matching for
        # external trials (CT.gov/PubMed). Will be rebuilt with abstract
        # retrieval + structured profile extraction + weighted matching.
        # See: _fetch_ct_fallback_matches()

        t4 = time.time()
        raw_matches = result.get("matches", [])
        rationales = generate_rationales(openai_client, patient_summary, raw_matches, max_matches=5)
        print(f"[Patient Matching] Rationale generation took {time.time() - t4:.2f}s")
        matches = _result_to_study_matches(result, rationales=rationales)

        response = PatientMatchResponse(
            matches=matches,
            total_matches=result.get("total_matches", len(matches)),
            patient_summary=patient_summary,
            extracted_profile=profile_dict,
        )
        print(f"[Patient Matching] Total request time: {time.time() - start_time:.2f}s")
        # Note: Caching disabled temporarily to ensure new matching logic is used
        return response
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error in unstructured patient match: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)
