"""
API endpoint for study details
Uses PostgreSQL for study profiles (admin studies + user uploads)
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.api.services.postgres_study_details_service import PostgresStudyDetailsService
from src.api.services.auth_dependencies import get_current_user_optional
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/study-details", tags=["study-details"])

# Request/Response models
class StudyDetailsRequest(BaseModel):
    """Request model for study details"""
    doc_id: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None  # Added for title-based search fallback

# Initialize service
_postgres_service = None

def get_postgres_service() -> PostgresStudyDetailsService:
    """Get or create the PostgreSQL study details service"""
    global _postgres_service
    if _postgres_service is None:
        _postgres_service = PostgresStudyDetailsService(
            pg_host=settings.postgres_host,
            pg_port=settings.postgres_port,
            pg_user=settings.postgres_user,
            pg_password=settings.postgres_password,
            pg_database=settings.postgres_database
        )
    return _postgres_service


async def get_user_upload_study_profile(doc_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Check if study profile exists in user uploads table."""
    try:
        from src.api.services.account_db import get_account_db
        import json
        
        db = get_account_db()
        pool = await db.get_pool()
        
        async with pool.acquire() as conn:
            # Search by doc_id pattern
            row = await conn.fetchrow("""
                SELECT upload_id, doc_id, filename, title, study_profile, doi, pmid
                FROM user_uploads
                WHERE user_id = $1 
                  AND (doc_id ILIKE $2 OR doc_id ILIKE $3)
                  AND study_profile IS NOT NULL
                LIMIT 1
            """, user_id, f"%{doc_id}%", doc_id)
            
            if row and row['study_profile']:
                study_profile = row['study_profile']
                if isinstance(study_profile, str):
                    study_profile = json.loads(study_profile)
                
                # Return in the same format as admin studies
                # The study_profile already has the right structure (study_details, outcomes, etc.)
                result = {
                    "source": "user_upload",
                    "doc_id": row['doc_id'],
                    "title": row['title'],
                    "doi": row['doi'],
                    "pmid": row['pmid'],
                    **study_profile  # Spread the study profile sections at top level
                }
                return result
    except Exception as e:
        logger.error(f"Error checking user uploads: {e}")
    
    return None


@router.post("")
async def get_study_details(
    request: StudyDetailsRequest,
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Get comprehensive study details from PostgreSQL
    
    Checks both admin studies table and user uploads table.
    
    Args:
        request: Request with doc_id, PMID, DOI, or title
        
    Returns:
        Complete study details with evidence quotes
    """
    # Normalize empty strings to None
    doc_id = request.doc_id if request.doc_id else None
    pmid = request.pmid if request.pmid else None
    doi = request.doi if request.doi else None
    title = request.title if request.title else None
    
    if not doc_id and not pmid and not doi and not title:
        raise HTTPException(
            status_code=400,
            detail="Either doc_id, pmid, doi, or title must be provided"
        )
    
    try:
        logger.info(f"Fetching study details: doc_id={doc_id}, pmid={pmid}, doi={doi}, title={title[:50] if title else None}")
        print(f"[StudyDetails] Request: doc_id={doc_id}, title={title[:50] if title else None}, current_user={current_user}")  # DEBUG
        
        # First check user uploads if user is logged in
        if current_user and doc_id:
            print(f"[StudyDetails] Checking user uploads for user {current_user.get('id')}")  # DEBUG
            user_upload_result = await get_user_upload_study_profile(
                doc_id=doc_id,
                user_id=current_user["id"]
            )
            if user_upload_result:
                print(f"[StudyDetails] Found in user uploads: {user_upload_result.get('doc_id')}")  # DEBUG
                logger.info(f"Found study in user uploads: {user_upload_result['doc_id']}")
                return user_upload_result
            else:
                print(f"[StudyDetails] Not found in user uploads")  # DEBUG
        
        # Then check admin studies table
        print(f"[StudyDetails] Checking admin studies table")  # DEBUG
        postgres_service = get_postgres_service()
        result = await postgres_service.get_study_details(
            doc_id=doc_id,
            pmid=pmid,
            doi=doi,
            title=title
        )
        
        if 'error' in result and result['error'] == 'Study not found':
            raise HTTPException(
                status_code=404,
                detail=f"Study not found: doc_id={doc_id}, pmid={pmid}, doi={doi}, title={title[:50] if title else None}"
            )
        
        if 'error' in result:
            raise HTTPException(
                status_code=500,
                detail=result['error']
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching study details: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================
# MULTI-STUDY COMPARISON ENDPOINT
# ============================================

from typing import List
from src.api.models.query_models import (
    StudyComparisonRequest,
    StudyComparisonResponse,
    StudyComparisonCategory,
    StudySummary,
    Artifact,
    ChartArtifact,
    ChartDataset,
)
from datetime import datetime


@router.post("/compare", response_model=StudyComparisonResponse)
async def compare_studies(
    request: StudyComparisonRequest,
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Compare 2-4 studies with comprehensive visualizations across categories.
    
    Categories include:
    - Treatment outcomes (OS, PFS, DFS)
    - Toxicity profiles
    - Patient demographics
    - Treatment details (dose, fractionation)
    """
    from src.core.config import settings
    from openai import OpenAI
    import json
    
    if len(request.study_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 studies required for comparison")
    if len(request.study_ids) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 studies can be compared")
    
    logger.info(f"[StudyComparison] Comparing {len(request.study_ids)} studies: {request.study_ids}")
    
    # Fetch all study details
    postgres_service = get_postgres_service()
    studies_data = []
    
    for doc_id in request.study_ids:
        try:
            # First check user uploads if logged in
            study_data = None
            if current_user:
                study_data = await get_user_upload_study_profile(doc_id, current_user["id"])
            
            # Then check admin studies
            if not study_data:
                study_data = await postgres_service.get_study_details(doc_id=doc_id)
            
            if study_data and 'error' not in study_data:
                studies_data.append(study_data)
                logger.info(f"[StudyComparison] Loaded study: {study_data.get('title', doc_id)}")
            else:
                logger.warning(f"[StudyComparison] Study not found: {doc_id}")
        except Exception as e:
            logger.error(f"[StudyComparison] Error loading study {doc_id}: {e}")
    
    if len(studies_data) < 2:
        raise HTTPException(
            status_code=404, 
            detail=f"Could not find enough studies. Found {len(studies_data)} of {len(request.study_ids)} requested."
        )
    
    # Build study summaries
    study_summaries = []
    for s in studies_data:
        # Extract year from publish_date or study_details
        year = None
        study_details = s.get('study_details', {})
        if study_details.get('publish_date', {}).get('value'):
            year = str(study_details['publish_date']['value'])[:4]
        
        # Get number of patients
        num_patients = None
        if study_details.get('number_of_patients', {}).get('value'):
            try:
                num_patients = int(study_details['number_of_patients']['value'])
            except (ValueError, TypeError):
                pass
        
        study_summaries.append(StudySummary(
            doc_id=s.get('doc_id', ''),
            title=s.get('title') or study_details.get('study_name', {}).get('value'),
            doi=s.get('doi'),
            year=year,
            cancer_type=s.get('diagnosis', {}).get('cancer_type', {}).get('value'),
            number_of_patients=num_patients,
            study_type=study_details.get('study_type', {}).get('value'),
            primary_endpoint=s.get('outcomes', {}).get('primary_endpoint', {}).get('value'),
        ))
    
    # Generate comparison categories
    categories = []
    
    # 1. Outcomes comparison
    outcomes_category = _generate_outcomes_comparison(studies_data, study_summaries)
    if outcomes_category:
        categories.append(outcomes_category)
    
    # 2. Toxicity comparison
    toxicity_category = _generate_toxicity_comparison(studies_data, study_summaries)
    if toxicity_category:
        categories.append(toxicity_category)
    
    # 3. Patient demographics comparison
    demographics_category = _generate_demographics_comparison(studies_data, study_summaries)
    if demographics_category:
        categories.append(demographics_category)
    
    # 4. Treatment details comparison
    treatment_category = _generate_treatment_comparison(studies_data, study_summaries)
    if treatment_category:
        categories.append(treatment_category)
    
    # Generate AI narrative
    openai_client = OpenAI(api_key=settings.openai_api_key)
    narrative = _generate_comparison_narrative(openai_client, studies_data, study_summaries)
    
    return StudyComparisonResponse(
        studies=study_summaries,
        categories=categories,
        narrative=narrative,
        generated_at=datetime.utcnow().isoformat()
    )


def _generate_outcomes_comparison(studies_data: List[Dict], summaries: List[StudySummary]) -> Optional[StudyComparisonCategory]:
    """Generate outcomes comparison charts."""
    charts = []
    study_labels = [s.title[:30] + "..." if s.title and len(s.title) > 30 else (s.title or s.doc_id[:20]) for s in summaries]
    
    # Colors for studies
    colors = [
        "rgba(59, 130, 246, 0.8)",   # Blue
        "rgba(16, 185, 129, 0.8)",   # Green
        "rgba(245, 158, 11, 0.8)",   # Amber
        "rgba(239, 68, 68, 0.8)",    # Red
    ]
    border_colors = [c.replace("0.8", "1") for c in colors]
    
    # Extract OS values
    os_values = []
    for s in studies_data:
        outcomes = s.get('outcomes', {})
        os_val = outcomes.get('overall_survival', {}).get('value')
        os_values.append(_extract_percentage(os_val))
    
    if any(v is not None for v in os_values):
        charts.append(Artifact(
            artifact_type="chart",
            chart=ChartArtifact(
                type="bar",
                title="Overall Survival",
                labels=study_labels,
                datasets=[ChartDataset(
                    label="OS",
                    data=[v if v is not None else 0 for v in os_values],
                    backgroundColor=colors[:len(os_values)],
                    borderColor=border_colors[:len(os_values)],
                )],
                unit="%",
                source="Study outcomes data"
            )
        ))
    
    # Extract PFS values
    pfs_values = []
    for s in studies_data:
        outcomes = s.get('outcomes', {})
        pfs_val = outcomes.get('progression_free_survival', {}).get('value')
        pfs_values.append(_extract_percentage(pfs_val))
    
    if any(v is not None for v in pfs_values):
        charts.append(Artifact(
            artifact_type="chart",
            chart=ChartArtifact(
                type="bar",
                title="Progression-Free Survival",
                labels=study_labels,
                datasets=[ChartDataset(
                    label="PFS",
                    data=[v if v is not None else 0 for v in pfs_values],
                    backgroundColor=colors[:len(pfs_values)],
                    borderColor=border_colors[:len(pfs_values)],
                )],
                unit="%",
                source="Study outcomes data"
            )
        ))
    
    # Extract DFS values
    dfs_values = []
    for s in studies_data:
        outcomes = s.get('outcomes', {})
        dfs_val = outcomes.get('disease_free_survival', {}).get('value')
        dfs_values.append(_extract_percentage(dfs_val))
    
    if any(v is not None for v in dfs_values):
        charts.append(Artifact(
            artifact_type="chart",
            chart=ChartArtifact(
                type="bar",
                title="Disease-Free Survival",
                labels=study_labels,
                datasets=[ChartDataset(
                    label="DFS",
                    data=[v if v is not None else 0 for v in dfs_values],
                    backgroundColor=colors[:len(dfs_values)],
                    borderColor=border_colors[:len(dfs_values)],
                )],
                unit="%",
                source="Study outcomes data"
            )
        ))
    
    # Extract local control values
    lc_values = []
    for s in studies_data:
        outcomes = s.get('outcomes', {})
        lc_val = outcomes.get('local_control', {}).get('value')
        lc_values.append(_extract_percentage(lc_val))
    
    if any(v is not None for v in lc_values):
        charts.append(Artifact(
            artifact_type="chart",
            chart=ChartArtifact(
                type="bar",
                title="Local Control",
                labels=study_labels,
                datasets=[ChartDataset(
                    label="LC",
                    data=[v if v is not None else 0 for v in lc_values],
                    backgroundColor=colors[:len(lc_values)],
                    borderColor=border_colors[:len(lc_values)],
                )],
                unit="%",
                source="Study outcomes data"
            )
        ))
    
    if not charts:
        return StudyComparisonCategory(
            category="outcomes",
            title="Treatment Outcomes",
            charts=[],
            summary="No numerical outcome data available for comparison.",
            data_available=False
        )
    
    return StudyComparisonCategory(
        category="outcomes",
        title="Treatment Outcomes",
        charts=charts,
        summary=f"Comparing survival outcomes across {len(summaries)} studies.",
        data_available=True
    )


def _generate_toxicity_comparison(studies_data: List[Dict], summaries: List[StudySummary]) -> Optional[StudyComparisonCategory]:
    """Generate toxicity comparison charts."""
    charts = []
    study_labels = [s.title[:25] + "..." if s.title and len(s.title) > 25 else (s.title or s.doc_id[:15]) for s in summaries]
    
    colors = [
        "rgba(59, 130, 246, 0.8)",
        "rgba(16, 185, 129, 0.8)",
        "rgba(245, 158, 11, 0.8)",
        "rgba(239, 68, 68, 0.8)",
    ]
    
    # Collect all toxicity types across studies
    all_toxicity_types = set()
    for s in studies_data:
        toxicity_list = s.get('toxicity', [])
        for tox in toxicity_list:
            tox_type = tox.get('toxicity_type', '')
            if tox_type and 'grade 3' in str(tox.get('grade', '')).lower():
                all_toxicity_types.add(tox_type)
    
    if all_toxicity_types:
        # Create grouped bar chart for Grade 3+ toxicities
        toxicity_labels = list(all_toxicity_types)[:6]  # Limit to 6 types
        datasets = []
        
        for i, s in enumerate(studies_data):
            toxicity_list = s.get('toxicity', [])
            tox_map = {}
            for tox in toxicity_list:
                if 'grade 3' in str(tox.get('grade', '')).lower():
                    tox_type = tox.get('toxicity_type', '')
                    freq = _extract_percentage(tox.get('frequency'))
                    if freq is not None:
                        tox_map[tox_type] = freq
            
            data = [tox_map.get(t, 0) for t in toxicity_labels]
            if any(d > 0 for d in data):
                datasets.append(ChartDataset(
                    label=study_labels[i],
                    data=data,
                    backgroundColor=[colors[i]] * len(data),
                    borderColor=[colors[i].replace("0.8", "1")] * len(data),
                ))
        
        if datasets:
            charts.append(Artifact(
                artifact_type="chart",
                chart=ChartArtifact(
                    type="bar",
                    title="Grade 3+ Toxicity Rates",
                    labels=toxicity_labels,
                    datasets=datasets,
                    unit="%",
                    source="Study toxicity data"
                )
            ))
    
    if not charts:
        return StudyComparisonCategory(
            category="toxicity",
            title="Toxicity Profiles",
            charts=[],
            summary="No comparable toxicity data available.",
            data_available=False
        )
    
    return StudyComparisonCategory(
        category="toxicity",
        title="Toxicity Profiles",
        charts=charts,
        summary=f"Comparing Grade 3+ toxicity rates across {len(summaries)} studies.",
        data_available=True
    )


def _generate_demographics_comparison(studies_data: List[Dict], summaries: List[StudySummary]) -> Optional[StudyComparisonCategory]:
    """Generate patient demographics comparison charts."""
    charts = []
    study_labels = [s.title[:25] + "..." if s.title and len(s.title) > 25 else (s.title or s.doc_id[:15]) for s in summaries]
    
    colors = [
        "rgba(59, 130, 246, 0.8)",
        "rgba(16, 185, 129, 0.8)",
        "rgba(245, 158, 11, 0.8)",
        "rgba(239, 68, 68, 0.8)",
    ]
    border_colors = [c.replace("0.8", "1") for c in colors]
    
    # Patient count comparison
    patient_counts = []
    for s in studies_data:
        study_details = s.get('study_details', {})
        count = study_details.get('number_of_patients', {}).get('value')
        if count:
            try:
                patient_counts.append(int(str(count).replace(',', '')))
            except (ValueError, TypeError):
                patient_counts.append(0)
        else:
            patient_counts.append(0)
    
    if any(c > 0 for c in patient_counts):
        charts.append(Artifact(
            artifact_type="chart",
            chart=ChartArtifact(
                type="bar",
                title="Number of Patients",
                labels=study_labels,
                datasets=[ChartDataset(
                    label="Patients",
                    data=patient_counts,
                    backgroundColor=colors[:len(patient_counts)],
                    borderColor=border_colors[:len(patient_counts)],
                )],
                unit=" patients",
                source="Study demographics"
            )
        ))
    
    # Median age comparison
    median_ages = []
    for s in studies_data:
        patient_chars = s.get('patient_characteristics', {})
        age = patient_chars.get('median_age', {}).get('value')
        median_ages.append(_extract_number(age))
    
    if any(a is not None for a in median_ages):
        charts.append(Artifact(
            artifact_type="chart",
            chart=ChartArtifact(
                type="bar",
                title="Median Patient Age",
                labels=study_labels,
                datasets=[ChartDataset(
                    label="Age",
                    data=[a if a is not None else 0 for a in median_ages],
                    backgroundColor=colors[:len(median_ages)],
                    borderColor=border_colors[:len(median_ages)],
                )],
                unit=" years",
                source="Study demographics"
            )
        ))
    
    if not charts:
        return StudyComparisonCategory(
            category="demographics",
            title="Patient Demographics",
            charts=[],
            summary="No comparable demographic data available.",
            data_available=False
        )
    
    return StudyComparisonCategory(
        category="demographics",
        title="Patient Demographics",
        charts=charts,
        summary=f"Comparing patient populations across {len(summaries)} studies.",
        data_available=True
    )


def _generate_treatment_comparison(studies_data: List[Dict], summaries: List[StudySummary]) -> Optional[StudyComparisonCategory]:
    """Generate treatment details comparison charts."""
    charts = []
    study_labels = [s.title[:25] + "..." if s.title and len(s.title) > 25 else (s.title or s.doc_id[:15]) for s in summaries]
    
    colors = [
        "rgba(59, 130, 246, 0.8)",
        "rgba(16, 185, 129, 0.8)",
        "rgba(245, 158, 11, 0.8)",
        "rgba(239, 68, 68, 0.8)",
    ]
    border_colors = [c.replace("0.8", "1") for c in colors]
    
    # Radiation dose comparison
    rt_doses = []
    for s in studies_data:
        treatment = s.get('treatment', {})
        radiation_details = treatment.get('radiation_details', [])
        if radiation_details:
            dose = radiation_details[0].get('total_dose')
            rt_doses.append(_extract_number(dose))
        else:
            rt_doses.append(None)
    
    if any(d is not None for d in rt_doses):
        charts.append(Artifact(
            artifact_type="chart",
            chart=ChartArtifact(
                type="bar",
                title="Radiation Dose",
                labels=study_labels,
                datasets=[ChartDataset(
                    label="Total Dose",
                    data=[d if d is not None else 0 for d in rt_doses],
                    backgroundColor=colors[:len(rt_doses)],
                    borderColor=border_colors[:len(rt_doses)],
                )],
                unit=" Gy",
                source="Study treatment data"
            )
        ))
    
    # Median follow-up comparison
    followups = []
    for s in studies_data:
        outcomes = s.get('outcomes', {})
        fu = outcomes.get('median_followup', {}).get('value')
        followups.append(_extract_number(fu))
    
    if any(f is not None for f in followups):
        charts.append(Artifact(
            artifact_type="chart",
            chart=ChartArtifact(
                type="bar",
                title="Median Follow-up",
                labels=study_labels,
                datasets=[ChartDataset(
                    label="Follow-up",
                    data=[f if f is not None else 0 for f in followups],
                    backgroundColor=colors[:len(followups)],
                    borderColor=border_colors[:len(followups)],
                )],
                unit=" months",
                source="Study outcomes data"
            )
        ))
    
    if not charts:
        return StudyComparisonCategory(
            category="treatment",
            title="Treatment Details",
            charts=[],
            summary="No comparable treatment data available.",
            data_available=False
        )
    
    return StudyComparisonCategory(
        category="treatment",
        title="Treatment Details",
        charts=charts,
        summary=f"Comparing treatment parameters across {len(summaries)} studies.",
        data_available=True
    )


def _generate_comparison_narrative(openai_client, studies_data: List[Dict], summaries: List[StudySummary]) -> str:
    """Generate AI narrative comparing the studies."""
    try:
        # Build context from studies
        study_contexts = []
        for i, (s, summary) in enumerate(zip(studies_data, summaries)):
            outcomes = s.get('outcomes', {})
            treatment = s.get('treatment', {})
            
            context = f"""Study {i+1}: {summary.title or 'Unknown'}
- Patients: {summary.number_of_patients or 'N/A'}
- Cancer Type: {summary.cancer_type or 'N/A'}
- Primary Endpoint: {summary.primary_endpoint or 'N/A'}
- OS: {outcomes.get('overall_survival', {}).get('value', 'N/A')}
- PFS: {outcomes.get('progression_free_survival', {}).get('value', 'N/A')}
- DFS: {outcomes.get('disease_free_survival', {}).get('value', 'N/A')}"""
            study_contexts.append(context)
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a clinical oncology expert comparing clinical studies.
Write a concise 3-4 paragraph narrative comparing the studies.
Focus on: key differences in outcomes, patient populations, and treatment approaches.
Be objective and highlight clinically meaningful differences."""},
                {"role": "user", "content": f"""Compare these {len(summaries)} studies:

{chr(10).join(study_contexts)}

Write a comparison narrative:"""}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[StudyComparison] Narrative generation failed: {e}")
        return f"Comparison of {len(summaries)} studies. See visualizations above for detailed data comparison."


def _extract_percentage(value) -> Optional[float]:
    """Extract percentage value from string."""
    if value is None:
        return None
    
    import re
    value_str = str(value)
    
    # Look for percentage patterns
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', value_str)
    if match:
        return float(match.group(1))
    
    # Look for decimal that might be a percentage
    match = re.search(r'(\d+(?:\.\d+)?)', value_str)
    if match:
        num = float(match.group(1))
        # If it's a small decimal, it might be a proportion
        if num <= 1.0:
            return num * 100
        return num
    
    return None


def _extract_number(value) -> Optional[float]:
    """Extract numeric value from string."""
    if value is None:
        return None
    
    import re
    value_str = str(value)
    
    match = re.search(r'(\d+(?:\.\d+)?)', value_str)
    if match:
        return float(match.group(1))
    
    return None
