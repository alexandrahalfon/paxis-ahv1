"""
Test script: Module routing + response generation for each question type.

Tests the full pipeline:
  1. Query classification (legacy query_type)
  2. Module classification (general_knowledge / patient_specific / evidence_exploration)
  3. Intent detection (explicit_question / patient_description / etc.)
  4. Response generation with module-specific prompts

Requires: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY in .env
"""

import sys, os, json, time, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from src.core.config import settings

# ── Test queries, one per expected module ──────────────────────────────

TEST_QUERIES = {
    "general_knowledge": [
        "What is the standard radiation dose for anal canal squamous cell carcinoma?",
        "What did the RTOG 0617 trial show for locally advanced NSCLC?",
    ],
    "patient_specific": [
        "62 year old male with T3N1M0 squamous cell carcinoma of the oropharynx, HPV positive, ECOG 1, no prior treatment",
        "55 yo female with pT2N1 triple negative breast cancer, status post lumpectomy, positive margins, LVI present",
    ],
    "evidence_exploration": [
        "What is the best treatment approach for locally advanced rectal cancer?",
        "Compare concurrent chemoradiation versus sequential therapy for head and neck cancer",
    ],
}

SEPARATOR = "=" * 90


def run_tests():
    client = OpenAI(api_key=settings.openai_api_key)

    # Import classifiers
    from src.api.services.module_classifier import classify_query_module
    from src.api.services.enhanced_rag_service import classify_query as classify_query_type
    from src.api.services.query_intent_service import QueryIntentService
    from src.api.services.module_generation_prompts import get_prompt_for_module

    intent_service = QueryIntentService(openai_client=client)

    # We'll use a small set of fake evidence chunks so the LLM has something to work with.
    # This isolates the prompt/formatting behavior from the retrieval pipeline.
    fake_evidence = [
        {
            "doc_id": "test_001",
            "title": "Randomized Trial of Chemoradiation in Locally Advanced Disease",
            "text": (
                "A phase III randomized trial enrolled 544 patients with locally advanced disease. "
                "Patients were randomized to concurrent chemoradiation (CRT) with cisplatin 100 mg/m2 "
                "on days 1, 22, 43 versus sequential chemotherapy followed by radiation. "
                "The CRT arm showed 5-year overall survival of 67.3% vs 55.1% (HR 0.74, 95% CI 0.58-0.94, p=0.014). "
                "Grade 3+ mucositis was 43% in CRT vs 28% in sequential arm. "
                "Median follow-up was 4.8 years. Radiation dose was 70 Gy in 35 fractions over 7 weeks."
            ),
            "citation": "Smith et al., 2021, Journal of Clinical Oncology",
            "section": "results",
            "score": 0.92,
        },
        {
            "doc_id": "test_002",
            "title": "NCCN Guidelines: Treatment Recommendations",
            "text": (
                "For locally advanced squamous cell carcinoma, the standard of care is concurrent "
                "chemoradiation with cisplatin-based regimen. Radiation dose of 66-70 Gy in 2 Gy fractions "
                "is recommended. For HPV-positive oropharyngeal cancer, de-escalation trials are ongoing. "
                "ECOG 0-1 patients are candidates for definitive chemoradiation. "
                "Alternative regimens include cetuximab for cisplatin-ineligible patients. "
                "5-year OS for HPV+ oropharyngeal cancer exceeds 80% with standard CRT."
            ),
            "citation": "NCCN Guidelines v2.2024, Head and Neck Cancers",
            "section": "recommendations",
            "score": 0.89,
        },
        {
            "doc_id": "test_003",
            "title": "Outcomes After Adjuvant Radiation for High-Risk Breast Cancer",
            "text": (
                "In a cohort of 312 patients with pT2N1 breast cancer treated with adjuvant radiation "
                "after lumpectomy, 5-year locoregional recurrence was 4.2%. Patients with positive margins "
                "had higher recurrence (8.7% vs 3.1%, p=0.02). LVI was an independent predictor of distant "
                "metastasis (HR 2.1, 95% CI 1.3-3.4). Whole breast radiation was 50 Gy in 25 fractions "
                "with a 10-16 Gy boost to the tumor bed. Grade 3+ dermatitis occurred in 12% of patients."
            ),
            "citation": "Johnson et al., 2022, International Journal of Radiation Oncology",
            "section": "results",
            "score": 0.87,
        },
    ]

    for module_name, queries in TEST_QUERIES.items():
        print(f"\n{SEPARATOR}")
        print(f"  MODULE: {module_name.upper()}")
        print(SEPARATOR)

        for query in queries:
            print(f"\n{'─' * 80}")
            print(f"  QUERY: {query}")
            print(f"{'─' * 80}\n")

            # ── Step 1: Legacy query type classification ──
            qt = classify_query_type(query)
            print(f"[QueryType]  primary_type={qt['primary_type']}, confidence={qt['confidence']:.2f}")

            # ── Step 2: Module classification ──
            mc = classify_query_module(query)
            mc_dict = mc.to_dict()
            print(f"[Module]     module={mc_dict['module']}, confidence={mc_dict['confidence']:.2f}")
            print(f"             signals={mc_dict.get('signals_matched', [])}")
            print(f"             has_patient={mc_dict.get('has_patient_context')}, has_question={mc_dict.get('has_explicit_question')}")
            print(f"             follow_ups={mc_dict.get('suggested_follow_ups', [])}")

            # ── Step 3: Intent detection (regex, no LLM) ──
            intent = intent_service._detect_intent(client, query)
            print(f"[Intent]     type={intent.intent_type}, question_type={intent.detected_question_type}, confidence={intent.confidence}")

            # ── Step 4: Generate response using module-specific prompt ──
            prompt_config = get_prompt_for_module(mc_dict["module"])
            print(f"[Prompt]     response_format={prompt_config['response_format']}")

            # Build context
            ctx_blocks = []
            for i, e in enumerate(fake_evidence[:3], 1):
                ctx_blocks.append(f"[{i}] {e['citation']} | {e['section']}\n{e['text']}")
            context = "\n\n---\n\n".join(ctx_blocks)

            user_msg = prompt_config["user_template"].format(question=query, context=context)

            print(f"\n[Generating response with {mc_dict['module']} prompt...]")
            t0 = time.perf_counter()

            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {"role": "system", "content": prompt_config["system"]},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1500,
            )

            elapsed = time.perf_counter() - t0
            answer = resp.choices[0].message.content.strip()

            print(f"[Done] {elapsed:.1f}s, {len(answer)} chars\n")
            print(answer)
            print()

    print(f"\n{SEPARATOR}")
    print("  ALL TESTS COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    run_tests()
