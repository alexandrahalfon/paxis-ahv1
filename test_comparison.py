#!/usr/bin/env python
"""
Side-by-side comparison of patient matching:
- SimplePatientMatchingService
- EnhancedRAGService

Tests which service returns better study matches for a patient profile.
"""

import sys
import asyncio
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from openai import OpenAI
from src.core.config import settings
from src.api.services.patient_matching_service_simple import SimplePatientMatchingService
from src.api.services.enhanced_rag_service import get_enhanced_rag_service


async def run_comparison():
    """Run both services and compare results side-by-side."""
    
    # Patient profile
    patient_description = """72-year-old man with PSA 5.6 ng/mL on screening, 
normal prostate on digital rectal exam, biopsy showing Gleason 3+4 in 1 core 
and Gleason 3+3 in 3 cores, 4/12 cores positive, no perineural invasion. 
Intermediate risk localized prostate cancer."""

    patient_profile = {
        "cancer_type": "Prostate Cancer",
        "anatomical_site": "prostate",
        "cancer_stage": "localized",
        "age": 72,
        "gender": "male",
        "molecular_markers": ["Gleason 3+4", "Gleason 7", "PSA 5.6"],
        "histology": "adenocarcinoma",
        "disease_characteristics": "4/12 cores positive, no PNI, normal DRE",
        "risk_group": "intermediate risk"
    }

    rag_query = f"""Find clinical studies for a patient with the following profile:
{patient_description}
What clinical trials or studies are relevant for this patient's treatment options?"""

    # Initialize services
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    patient_service = SimplePatientMatchingService(qdrant, openai_client, settings.qdrant_collection)
    rag_service = get_enhanced_rag_service()

    print("=" * 100)
    print("PATIENT PROFILE")
    print("=" * 100)
    print(patient_description)
    print()

    # Run SimplePatientMatchingService
    print("[Running] SimplePatientMatchingService...")
    simple_results = patient_service.match_patient(
        patient_profile=patient_profile,
        top_k=10
    )
    simple_matches = simple_results.get("matches", [])
    print(f"[Done] Found {len(simple_matches)} matches")

    # Run EnhancedRAGService
    print("[Running] EnhancedRAGService...")
    rag_results = await rag_service.query(
        question=rag_query,
        category="prostate",
        top_k=10
    )
    rag_evidence = rag_results.get("evidence", [])
    print(f"[Done] Found {len(rag_evidence)} evidence chunks")

    # Deduplicate RAG results by doc_id
    seen_rag = set()
    rag_unique = []
    for e in rag_evidence:
        doc_id = e.get("doc_id", "")
        if doc_id and doc_id not in seen_rag:
            seen_rag.add(doc_id)
            rag_unique.append(e)

    # Output side-by-side comparison
    output = []
    output.append("=" * 100)
    output.append("SIDE-BY-SIDE COMPARISON: Patient Matching Results")
    output.append("=" * 100)
    output.append("")
    output.append(f"Patient: {patient_description[:100]}...")
    output.append("")
    output.append("-" * 100)
    output.append(f"{'SimplePatientMatchingService':<50} | {'EnhancedRAGService':<50}")
    output.append(f"{'(with semantic validation)':<50} | {'(RAG pipeline)':<50}")
    output.append("-" * 100)
    output.append("")

    max_rows = max(len(simple_matches), len(rag_unique))
    
    for i in range(max_rows):
        # Left side: SimplePatientMatchingService
        if i < len(simple_matches):
            m = simple_matches[i]
            title = m.get("title", "Unknown")[:45]
            score = m.get("match_score", 0)
            doc_id = m.get("doc_id", "")[:40]
            left_title = f"{i+1}. [{score:.0%}] {title}"
            left_id = f"   {doc_id}"
        else:
            left_title = ""
            left_id = ""

        # Right side: EnhancedRAGService
        if i < len(rag_unique):
            e = rag_unique[i]
            title = e.get("title", "Unknown")[:45]
            score = e.get("score_rerank", e.get("score", 0))
            doc_id = e.get("doc_id", "")[:40]
            right_title = f"{i+1}. [{score:.2f}] {title}"
            right_id = f"   {doc_id}"
        else:
            right_title = ""
            right_id = ""

        output.append(f"{left_title:<50} | {right_title:<50}")
        output.append(f"{left_id:<50} | {right_id:<50}")
        output.append("")

    # Find overlapping studies
    simple_doc_ids = {m.get("doc_id") for m in simple_matches}
    rag_doc_ids = {e.get("doc_id") for e in rag_unique}
    overlap = simple_doc_ids & rag_doc_ids
    only_simple = simple_doc_ids - rag_doc_ids
    only_rag = rag_doc_ids - simple_doc_ids

    output.append("-" * 100)
    output.append("OVERLAP ANALYSIS")
    output.append("-" * 100)
    output.append(f"Studies found by BOTH services: {len(overlap)}")
    output.append(f"Studies found ONLY by SimplePatientMatchingService: {len(only_simple)}")
    output.append(f"Studies found ONLY by EnhancedRAGService: {len(only_rag)}")
    output.append("")

    if overlap:
        output.append("OVERLAPPING STUDIES:")
        for doc_id in overlap:
            # Find title from either service
            title = next((m.get("title") for m in simple_matches if m.get("doc_id") == doc_id), "Unknown")
            output.append(f"  - {title[:70]}")
    output.append("")

    if only_simple:
        output.append("UNIQUE TO SimplePatientMatchingService:")
        for doc_id in only_simple:
            title = next((m.get("title") for m in simple_matches if m.get("doc_id") == doc_id), "Unknown")
            output.append(f"  - {title[:70]}")
    output.append("")

    if only_rag:
        output.append("UNIQUE TO EnhancedRAGService:")
        for doc_id in only_rag:
            title = next((e.get("title") for e in rag_unique if e.get("doc_id") == doc_id), "Unknown")
            output.append(f"  - {title[:70]}")
    output.append("")

    # Print and save
    result_text = "\n".join(output)
    print(result_text)

    with open("test_output.txt", "w") as f:
        f.write(result_text)
    
    print("\n[Saved to test_output.txt]")


if __name__ == "__main__":
    asyncio.run(run_comparison())
