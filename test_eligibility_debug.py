"""
Debug test for patient trial eligibility service.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()


async def test_eligibility_service():
    """Test the eligibility service and print detailed responses."""
    from src.api.services.patient_trial_eligibility_service import get_eligibility_service
    
    service = get_eligibility_service()
    
    # Test case: N1mi breast cancer patient
    patient = "55 year-old female who underwent breast-conserving surgery for a pT1cN1mi cM0 ER+ HER2- breast cancer and 21 gene recurrence score of 22"
    
    print(f"\nPatient: {patient}\n")
    print("="*80)
    
    result = await service.find_eligible_trials(
        patient_description=patient,
        top_k=10,
        max_eligible=5,
        min_yes_matches=2,
    )
    
    print(f"\nTotal evaluated: {result['total_evaluated']}")
    print(f"Yes count: {result['yes_count']}")
    print(f"Partial count: {result['partial_count']}")
    print(f"Total eligible: {result['total_eligible']}")
    
    print("\n" + "="*80)
    print("ELIGIBLE TRIALS:")
    print("="*80)
    
    for trial in result['eligible_trials']:
        print(f"\nTitle: {trial['title'][:60]}...")
        print(f"Match Category: {trial['match_category']}")
        print(f"Confidence: {trial['confidence']}")
        print(f"Reasoning: {trial['reasoning'][:300]}...")
        print("-"*40)


if __name__ == "__main__":
    asyncio.run(test_eligibility_service())
