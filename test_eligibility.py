#!/usr/bin/env python
"""
Test the new patient trial eligibility service.
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from src.api.services.patient_trial_eligibility_service import get_eligibility_service


async def test_eligibility():
    """Test eligibility checking for prostate cancer patient."""
    
    patient_description = """72-year-old man was found to have an elevated PSA of 5.6 on PSA screening and normal
prostate gland on digital rectal exam, and U/S guided biopsy revealed Gleason 3 + 4 in 1 core
and Gleason 3 + 3 in 3 additional cores, with a total involvement of 4/12 cores. No evidence of PNI."""

    print("=" * 80)
    print("PATIENT TRIAL ELIGIBILITY TEST")
    print("=" * 80)
    print(f"\nPatient: {patient_description}\n")
    
    service = get_eligibility_service()
    
    result = await service.find_eligible_trials(
        patient_description=patient_description,
        top_k=8,
        max_eligible=5,
        category="prostate",
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nPatient Summary: {result['patient_summary']}")
    print(f"Studies Evaluated: {result['total_evaluated']}")
    print(f"Eligible Trials Found: {result['total_eligible']}")
    
    print("\n" + "-" * 60)
    print("ELIGIBLE TRIALS:")
    print("-" * 60)
    
    for i, trial in enumerate(result['eligible_trials'], 1):
        print(f"\n{i}. {trial['title']}")
        print(f"   Doc ID: {trial['doc_id']}")
        print(f"   DOI: {trial.get('doi', 'N/A')}")
        print(f"   Confidence: {trial['confidence']}")
        print(f"   Reasoning: {trial['reasoning']}")
        print(f"   Criteria Matched: {', '.join(trial['eligibility_criteria_matched'][:3])}")
        if trial['eligibility_criteria_not_met']:
            print(f"   Criteria Not Met: {', '.join(trial['eligibility_criteria_not_met'][:2])}")
    
    # Save to file
    with open("test_output.txt", "w") as f:
        f.write("PATIENT TRIAL ELIGIBILITY TEST\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Patient: {patient_description}\n\n")
        f.write(f"Studies Evaluated: {result['total_evaluated']}\n")
        f.write(f"Eligible Trials Found: {result['total_eligible']}\n\n")
        f.write("-" * 60 + "\n")
        f.write("ELIGIBLE TRIALS:\n")
        f.write("-" * 60 + "\n")
        
        for i, trial in enumerate(result['eligible_trials'], 1):
            f.write(f"\n{i}. {trial['title']}\n")
            f.write(f"   Doc ID: {trial['doc_id']}\n")
            f.write(f"   DOI: {trial.get('doi', 'N/A')}\n")
            f.write(f"   Confidence: {trial['confidence']}\n")
            f.write(f"   Reasoning: {trial['reasoning']}\n")
            f.write(f"   Criteria Matched: {trial['eligibility_criteria_matched']}\n")
            f.write(f"   Criteria Not Met: {trial['eligibility_criteria_not_met']}\n")
    
    print("\n[Saved to test_output.txt]")


if __name__ == "__main__":
    asyncio.run(test_eligibility())
