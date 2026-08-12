"""
Query Classifier Service

Extracts structured clinical data from free-text queries and maps it to the 
PostgreSQL studies schema for strict matching.

This service parses complex clinical descriptions like:
"68 year old female, non-smoker, with SCC of R maxilla s/p maxillectomy, pT4N0..."

And produces a structured query object that can be used to filter studies in PostgreSQL.
"""

import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

from src.api.services.stage_inference_service import infer_stage_for_query

@dataclass
class StructuredQuery:
    """
    Structured query object matching PostgreSQL studies schema.
    All fields are optional - only extracted fields will be populated.
    """
    # Patient Demographics
    age: Optional[int] = None
    age_range_min: Optional[int] = None
    age_range_max: Optional[int] = None
    gender: Optional[str] = None  # 'male' | 'female'
    race_ethnicity: Optional[str] = None
    performance_status: Optional[str] = None  # ECOG 0-4
    
    # Cancer Diagnosis
    cancer_type: Optional[str] = None  # e.g., "Squamous Cell Carcinoma", "Breast Cancer"
    cancer_location: Optional[str] = None  # e.g., "maxilla", "oral cavity", "head and neck"
    histopathologic_type: Optional[str] = None  # e.g., "poorly differentiated SCC"
    tumor_grade: Optional[str] = None  # e.g., "poorly differentiated", "grade 3"
    molecular_subtype: Optional[str] = None  # e.g., "HER2+", "EGFR mutant"
    
    # Staging
    tnm_t: Optional[str] = None  # T stage: T1, T2, T3, T4, etc.
    tnm_n: Optional[str] = None  # N stage: N0, N1, N2, N3, etc.
    tnm_m: Optional[str] = None  # M stage: M0, M1
    overall_stage: Optional[str] = None  # I, II, III, IV
    risk_stratification: Optional[str] = None  # low, intermediate, high
    metastatic_status: Optional[str] = None  # localized, locally advanced, metastatic
    
    # Pathology Details
    depth_of_invasion: Optional[str] = None  # DOI in mm
    lymphovascular_invasion: Optional[str] = None  # LVI+, LVI-
    perineural_invasion: Optional[str] = None  # PNI+, PNI-
    margin_status: Optional[str] = None  # positive, negative, close
    lymph_nodes_examined: Optional[int] = None
    lymph_nodes_positive: Optional[int] = None
    
    # Treatment History
    prior_surgery: Optional[str] = None  # type of surgery performed
    prior_radiation: Optional[bool] = None
    prior_chemotherapy: Optional[bool] = None
    recurrence_status: Optional[str] = None  # primary, recurrent, nodal recurrence
    
    # Stage Inference
    stage_inferred: bool = False
    stage_ambiguous: bool = False
    stage_possible_stages: List[str] = field(default_factory=list)
    stage_required_factors: List[str] = field(default_factory=list)
    stage_inference_notes: List[str] = field(default_factory=list)
    stage_confidence: str = ""

    # Comorbidities & Risk Factors
    smoking_status: Optional[str] = None  # never, former, current
    comorbidities: List[str] = field(default_factory=list)
    
    # Study Preferences (from user preferences)
    study_type: Optional[str] = None  # prospective, retrospective, RCT
    study_phase: Optional[str] = None  # I, II, III
    treatment_modality: Optional[str] = None  # radiation, chemotherapy, surgery
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and empty lists"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if isinstance(value, list) and len(value) == 0:
                    continue
                result[key] = value
        return result
    
    def get_search_summary(self) -> str:
        """Generate a human-readable summary of the extracted query"""
        parts = []
        
        if self.age:
            parts.append(f"{self.age} year old")
        if self.gender:
            parts.append(self.gender)
        if self.cancer_type:
            parts.append(f"with {self.cancer_type}")
        if self.cancer_location:
            parts.append(f"of {self.cancer_location}")
        if self.overall_stage:
            parts.append(f"stage {self.overall_stage}")
        elif self.tnm_t or self.tnm_n:
            tnm = []
            if self.tnm_t:
                tnm.append(self.tnm_t)
            if self.tnm_n:
                tnm.append(self.tnm_n)
            if self.tnm_m:
                tnm.append(self.tnm_m)
            parts.append(f"({'/'.join(tnm)})")
        if self.recurrence_status:
            parts.append(f"({self.recurrence_status})")
        if self.histopathologic_type:
            parts.append(f"- {self.histopathologic_type}")
            
        return " ".join(parts) if parts else "General oncology query"


# LLM extraction schema - comprehensive for clinical queries
EXTRACTION_SCHEMA = """
{
    "age": "integer or null - patient age in years",
    "gender": "string or null - 'male' or 'female' only",
    "race_ethnicity": "string or null - e.g., 'Asian', 'Caucasian', 'African American'",
    "performance_status": "string or null - ECOG score as '0', '1', '2', '3', or '4'",
    
    "cancer_type": "string or null - cancer type, e.g., 'Squamous Cell Carcinoma', 'Adenocarcinoma', 'Breast Cancer'",
    "cancer_location": "string or null - anatomical location, e.g., 'maxilla', 'oral cavity', 'head and neck', 'lung'",
    "histopathologic_type": "string or null - histology details, e.g., 'poorly differentiated SCC', 'invasive ductal carcinoma'",
    "tumor_grade": "string or null - grade, e.g., 'well differentiated', 'moderately differentiated', 'poorly differentiated', 'grade 1/2/3'",
    "molecular_subtype": "string or null - molecular markers, e.g., 'HER2+', 'ER+/PR+', 'EGFR mutant', 'PD-L1 positive'",
    
    "tnm_t": "string or null - T stage only, e.g., 'T1', 'T2', 'T3', 'T4', 'T4a', 'T4b'",
    "tnm_n": "string or null - N stage only, e.g., 'N0', 'N1', 'N2', 'N2a', 'N2b', 'N3'",
    "tnm_m": "string or null - M stage only, e.g., 'M0', 'M1'",
    "overall_stage": "string or null - overall stage as 'I', 'II', 'III', 'IV' or 'IIA', 'IIB', 'IIIA', etc.",
    "risk_stratification": "string or null - 'low', 'intermediate', 'high', or 'very high'",
    "metastatic_status": "string or null - 'localized', 'locally advanced', 'regional', or 'metastatic'",
    
    "depth_of_invasion": "string or null - DOI measurement, e.g., '15 mm', '5mm'",
    "lymphovascular_invasion": "string or null - 'positive', 'negative', 'LVI+', 'LVI-'",
    "perineural_invasion": "string or null - 'positive', 'negative', 'PNI+', 'PNI-'",
    "margin_status": "string or null - 'positive', 'negative', 'close'",
    "lymph_nodes_examined": "integer or null - total number of lymph nodes examined",
    "lymph_nodes_positive": "integer or null - number of positive lymph nodes",
    
    "prior_surgery": "string or null - type of surgery, e.g., 'maxillectomy', 'mastectomy', 'neck dissection'",
    "prior_radiation": "boolean or null - true if prior RT mentioned",
    "prior_chemotherapy": "boolean or null - true if prior chemo mentioned",
    "recurrence_status": "string or null - 'primary', 'recurrent', 'nodal recurrence', 'local recurrence', 'distant recurrence'",
    
    "smoking_status": "string or null - 'never', 'former', 'current' only",
    "comorbidities": "array of strings - e.g., ['hypertension', 'diabetes', 'COPD']"
}
"""

EXTRACTION_SYSTEM = """You are a medical data extraction specialist. Extract structured clinical data from patient descriptions and clinical queries.

CRITICAL RULES:
1. Output ONLY valid JSON matching the schema provided.
2. Use null for any field not explicitly mentioned or clearly implied.
3. Be precise with staging - extract exact TNM components (pT4N0 → tnm_t: "T4", tnm_n: "N0").
4. Normalize abbreviations:
   - SCC → Squamous Cell Carcinoma
   - HTN → hypertension
   - T2DM → type 2 diabetes
   - PMH → past medical history (extract conditions)
   - s/p → status post (indicates prior procedure)
   - LVI-, PNI- → negative for lymphovascular/perineural invasion
   - DOI → depth of invasion
5. For lymph nodes like "0/42 LN involved" → lymph_nodes_positive: 0, lymph_nodes_examined: 42
6. Identify recurrence from phrases like "nodal recurrence", "recurrent disease", "s/p [treatment] with recurrence"
7. Extract anatomical location precisely (maxilla, gingiva, hard palate → cancer_location: "oral cavity" or more specific)
8. Map cancer locations to standard categories when possible:
   - maxilla, gingiva, buccal mucosa, hard palate, tongue → "oral cavity" or "head and neck"
   - pharynx, larynx, hypopharynx → "head and neck"

Do not include any text outside the JSON object."""

EXTRACTION_USER_TEMPLATE = """Extract structured clinical data from this query/description:

"{text}"

Return exactly one JSON object with all applicable fields from the schema. Use null for fields not mentioned.

Schema:
{schema}"""


class QueryClassifierService:
    """
    Service to classify and structure clinical queries for PostgreSQL matching.
    """
    
    def __init__(self, openai_client: OpenAI = None):
        self.openai_client = openai_client
        
    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client"""
        if self.openai_client is None:
            from src.core.config import settings
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        return self.openai_client
    
    def classify_query(self, query_text: str) -> StructuredQuery:
        """
        Extract structured clinical data from a free-text query.
        
        Args:
            query_text: Free-text clinical description or query
            
        Returns:
            StructuredQuery object with extracted fields
        """
        text = (query_text or "").strip()
        if len(text) < 10:
            return StructuredQuery()
        
        try:
            client = self._get_client()
            
            from src.core.config import settings
            response = client.chat.completions.create(
                model=settings.openai_mini_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(
                        text=text,
                        schema=EXTRACTION_SCHEMA
                    )},
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            
            content = (response.choices[0].message.content or "").strip()
            
            # Strip markdown code block if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            
            data = json.loads(content)
            query = self._normalize_extracted(data)
            
            # RAG FALLBACK: If no TNM was extracted but clinical indicators are present,
            # use RAG to retrieve AJCC staging context and infer staging
            if not query.tnm_t and not query.tnm_n and not query.overall_stage:
                try:
                    from src.api.services.staging_rag_fallback import (
                        detect_staging_keywords,
                        staging_rag_fallback_sync,
                    )
                    
                    detection = detect_staging_keywords(text)
                    if detection.needs_rag_fallback:
                        logger.info(f"[QueryClassifier] RAG fallback triggered - clinical indicators: {detection.detected_clinical_indicators[:3]}")
                        
                        rag_result = staging_rag_fallback_sync(
                            query=text,
                            cancer_type=query.cancer_type,
                            cancer_location=query.cancer_location,
                        )
                        
                        if rag_result:
                            # Update query with RAG-inferred staging
                            if rag_result.get("tnm_t"):
                                query.tnm_t = rag_result["tnm_t"]
                            if rag_result.get("tnm_n"):
                                query.tnm_n = rag_result["tnm_n"]
                            if rag_result.get("tnm_m"):
                                query.tnm_m = rag_result["tnm_m"]
                            if rag_result.get("overall_stage"):
                                query.overall_stage = rag_result["overall_stage"]
                                query.stage_inferred = True
                                query.stage_confidence = rag_result.get("confidence", "medium")
                            
                            query.stage_inference_notes.append(
                                f"Stage inferred via RAG fallback: {rag_result.get('reasoning', 'N/A')}"
                            )
                            
                            logger.info(
                                f"[QueryClassifier] RAG fallback result: "
                                f"T={query.tnm_t}, N={query.tnm_n}, M={query.tnm_m} -> Stage {query.overall_stage}"
                            )
                except Exception as e:
                    logger.warning(f"[QueryClassifier] RAG fallback failed: {e}")
            
            # Infer stage if TNM present but overall_stage missing
            # DISABLED: Stage inference temporarily disabled
            # if not query.overall_stage and (query.tnm_t or query.tnm_n or query.tnm_m or query.metastatic_status):
            #     try:
            #         inference = infer_stage_for_query(
            #             cancer_type=query.cancer_type,
            #             cancer_location=query.cancer_location,
            #             tnm_t=query.tnm_t,
            #             tnm_n=query.tnm_n,
            #             tnm_m=query.tnm_m,
            #             metastatic_status=query.metastatic_status,
            #             age=query.age,
            #             histopathologic_type=query.histopathologic_type,
            #             molecular_subtype=query.molecular_subtype,
            #             tumor_grade=query.tumor_grade,
            #         )
            #         
            #         if inference.stage_group:
            #             query.overall_stage = inference.stage_group
            #             query.stage_inferred = True
            #             query.stage_confidence = inference.confidence
            #         
            #         if inference.is_ambiguous:
            #             query.stage_ambiguous = True
            #             query.stage_possible_stages = inference.possible_stages
            #             query.stage_required_factors = inference.required_factors
            #             query.stage_inference_notes = inference.notes
            #         
            #         if not query.metastatic_status and inference.metastatic_status:
            #             query.metastatic_status = inference.metastatic_status
            #         
            #         if inference.stage_group:
            #             logger.info(
            #                 f"[StageInference] {inference.tnm_used} -> Stage {inference.stage_group} "
            #                 f"({inference.source}, {inference.confidence})"
            #                 f"{' [AMBIGUOUS]' if inference.is_ambiguous else ''}"
            #             )
            #     except Exception as e:
            #         logger.warning(f"[StageInference] Failed: {e}")
            
            return query
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            # Fall back to regex-based extraction
            return self._regex_fallback(text)
    
    def _normalize_extracted(self, data: Dict[str, Any]) -> StructuredQuery:
        """Normalize extracted data to StructuredQuery"""
        query = StructuredQuery()
        
        # Demographics
        if data.get("age") is not None:
            try:
                query.age = int(data["age"])
            except (TypeError, ValueError):
                pass
        
        if data.get("gender"):
            g = str(data["gender"]).lower().strip()
            if g in ("male", "female"):
                query.gender = g
        
        if data.get("race_ethnicity"):
            query.race_ethnicity = str(data["race_ethnicity"]).strip()
        
        if data.get("performance_status"):
            ps = str(data["performance_status"]).strip()
            # Extract just the number if it's like "ECOG 1"
            ps_match = re.search(r'(\d)', ps)
            if ps_match and ps_match.group(1) in ("0", "1", "2", "3", "4"):
                query.performance_status = ps_match.group(1)
        
        # Cancer Diagnosis
        if data.get("cancer_type"):
            query.cancer_type = str(data["cancer_type"]).strip()
        
        if data.get("cancer_location"):
            query.cancer_location = str(data["cancer_location"]).strip()
        
        if data.get("histopathologic_type"):
            query.histopathologic_type = str(data["histopathologic_type"]).strip()
        
        if data.get("tumor_grade"):
            query.tumor_grade = str(data["tumor_grade"]).strip()
        
        if data.get("molecular_subtype"):
            query.molecular_subtype = str(data["molecular_subtype"]).strip()
        
        # Staging
        if data.get("tnm_t"):
            query.tnm_t = str(data["tnm_t"]).upper().strip()
        
        if data.get("tnm_n"):
            query.tnm_n = str(data["tnm_n"]).upper().strip()
        
        if data.get("tnm_m"):
            query.tnm_m = str(data["tnm_m"]).upper().strip()
        
        if data.get("overall_stage"):
            stage = str(data["overall_stage"]).upper().strip()
            # Normalize to roman numerals
            stage = stage.replace("STAGE ", "").replace("STAGE", "")
            query.overall_stage = stage
        
        if data.get("risk_stratification"):
            query.risk_stratification = str(data["risk_stratification"]).lower().strip()
        
        if data.get("metastatic_status"):
            query.metastatic_status = str(data["metastatic_status"]).lower().strip()
        
        # Pathology
        if data.get("depth_of_invasion"):
            query.depth_of_invasion = str(data["depth_of_invasion"]).strip()
        
        if data.get("lymphovascular_invasion"):
            lvi = str(data["lymphovascular_invasion"]).lower().strip()
            if "positive" in lvi or "lvi+" in lvi or lvi == "+":
                query.lymphovascular_invasion = "positive"
            elif "negative" in lvi or "lvi-" in lvi or lvi == "-":
                query.lymphovascular_invasion = "negative"
        
        if data.get("perineural_invasion"):
            pni = str(data["perineural_invasion"]).lower().strip()
            if "positive" in pni or "pni+" in pni or pni == "+":
                query.perineural_invasion = "positive"
            elif "negative" in pni or "pni-" in pni or pni == "-":
                query.perineural_invasion = "negative"
        
        if data.get("margin_status"):
            margin = str(data["margin_status"]).lower().strip()
            if "positive" in margin or margin == "+":
                query.margin_status = "positive"
            elif "negative" in margin or margin == "-" or "clear" in margin:
                query.margin_status = "negative"
            elif "close" in margin:
                query.margin_status = "close"
        
        if data.get("lymph_nodes_examined") is not None:
            try:
                query.lymph_nodes_examined = int(data["lymph_nodes_examined"])
            except (TypeError, ValueError):
                pass
        
        if data.get("lymph_nodes_positive") is not None:
            try:
                query.lymph_nodes_positive = int(data["lymph_nodes_positive"])
            except (TypeError, ValueError):
                pass
        
        # Treatment History
        if data.get("prior_surgery"):
            query.prior_surgery = str(data["prior_surgery"]).strip()
        
        if data.get("prior_radiation") is not None:
            query.prior_radiation = bool(data["prior_radiation"])
        
        if data.get("prior_chemotherapy") is not None:
            query.prior_chemotherapy = bool(data["prior_chemotherapy"])
        
        if data.get("recurrence_status"):
            query.recurrence_status = str(data["recurrence_status"]).lower().strip()
        
        # Risk Factors
        if data.get("smoking_status"):
            s = str(data["smoking_status"]).lower().strip()
            if s in ("never", "non-smoker", "non smoker", "nonsmoker"):
                query.smoking_status = "never"
            elif s in ("former", "ex-smoker", "ex smoker", "quit"):
                query.smoking_status = "former"
            elif s in ("current", "active", "smoker"):
                query.smoking_status = "current"
        
        if data.get("comorbidities"):
            comorbidities = data["comorbidities"]
            if isinstance(comorbidities, list):
                query.comorbidities = [str(c).strip() for c in comorbidities if str(c).strip()]
            elif isinstance(comorbidities, str) and comorbidities.strip():
                query.comorbidities = [comorbidities.strip()]
        
        return query
    
    def _regex_fallback(self, text: str) -> StructuredQuery:
        """Fallback regex-based extraction when LLM fails"""
        query = StructuredQuery()
        text_lower = text.lower()
        
        # Age
        age_match = re.search(r'(\d{1,3})\s*(?:year|yo|y\.?o\.?|years?\s*old)', text_lower)
        if age_match:
            query.age = int(age_match.group(1))
        
        # Gender
        if re.search(r'\b(female|woman|f)\b', text_lower):
            query.gender = "female"
        elif re.search(r'\b(male|man|m)\b', text_lower):
            query.gender = "male"
        
        # TNM staging
        tnm_match = re.search(r'p?T(\d[a-c]?)', text, re.IGNORECASE)
        if tnm_match:
            query.tnm_t = f"T{tnm_match.group(1).upper()}"
        
        n_match = re.search(r'p?N(\d[a-c]?)', text, re.IGNORECASE)
        if n_match:
            query.tnm_n = f"N{n_match.group(1).upper()}"
        
        m_match = re.search(r'p?M(\d)', text, re.IGNORECASE)
        if m_match:
            query.tnm_m = f"M{m_match.group(1)}"
        
        # Cancer type
        if 'scc' in text_lower or 'squamous' in text_lower:
            query.cancer_type = "Squamous Cell Carcinoma"
        elif 'adenocarcinoma' in text_lower:
            query.cancer_type = "Adenocarcinoma"
        
        # Smoking
        if 'non-smok' in text_lower or 'nonsmoker' in text_lower or 'never smok' in text_lower:
            query.smoking_status = "never"
        elif 'former smok' in text_lower or 'ex-smok' in text_lower:
            query.smoking_status = "former"
        elif 'current smok' in text_lower or 'active smok' in text_lower:
            query.smoking_status = "current"
        
        # LVI/PNI
        if 'lvi-' in text_lower or 'lvi negative' in text_lower:
            query.lymphovascular_invasion = "negative"
        elif 'lvi+' in text_lower or 'lvi positive' in text_lower:
            query.lymphovascular_invasion = "positive"
        
        if 'pni-' in text_lower or 'pni negative' in text_lower:
            query.perineural_invasion = "negative"
        elif 'pni+' in text_lower or 'pni positive' in text_lower:
            query.perineural_invasion = "positive"
        
        # Recurrence
        if 'recurrence' in text_lower or 'recurrent' in text_lower:
            if 'nodal' in text_lower:
                query.recurrence_status = "nodal recurrence"
            elif 'local' in text_lower:
                query.recurrence_status = "local recurrence"
            else:
                query.recurrence_status = "recurrent"
        
        return query
    
    def build_postgres_filters(self, query: StructuredQuery) -> Dict[str, Any]:
        """
        Convert StructuredQuery to PostgreSQL filter conditions.
        
        Returns a dict with:
        - conditions: list of SQL WHERE clause fragments
        - params: list of parameter values
        - param_types: dict mapping param index to expected type
        """
        conditions = []
        params = []
        param_idx = 1
        
        # Cancer type - flexible matching
        if query.cancer_type:
            conditions.append(f"(cancer_type ILIKE ${param_idx} OR histopathologic_type ILIKE ${param_idx})")
            params.append(f"%{query.cancer_type}%")
            param_idx += 1
        
        # Cancer location
        if query.cancer_location:
            conditions.append(f"cancer_location ILIKE ${param_idx}")
            params.append(f"%{query.cancer_location}%")
            param_idx += 1
        
        # Histopathologic type
        if query.histopathologic_type:
            conditions.append(f"histopathologic_type ILIKE ${param_idx}")
            params.append(f"%{query.histopathologic_type}%")
            param_idx += 1
        
        # Tumor grade
        if query.tumor_grade:
            conditions.append(f"tumor_grade ILIKE ${param_idx}")
            params.append(f"%{query.tumor_grade}%")
            param_idx += 1
        
        # Overall stage - check stage_distribution table
        if query.overall_stage:
            # This would need a JOIN with stage_distribution
            conditions.append(f"""
                EXISTS (
                    SELECT 1 FROM stage_distribution sd 
                    WHERE sd.study_id = studies.study_id 
                    AND sd.stage_category ILIKE ${param_idx}
                )
            """)
            params.append(f"%{query.overall_stage}%")
            param_idx += 1
        
        # Metastatic status
        if query.metastatic_status:
            conditions.append(f"metastatic_status ILIKE ${param_idx}")
            params.append(f"%{query.metastatic_status}%")
            param_idx += 1
        
        # Risk stratification
        if query.risk_stratification:
            conditions.append(f"risk_stratification ILIKE ${param_idx}")
            params.append(f"%{query.risk_stratification}%")
            param_idx += 1
        
        return {
            "conditions": conditions,
            "params": params,
            "query_summary": query.get_search_summary(),
            "extracted_fields": query.to_dict()
        }


# Singleton instance
_classifier_service: Optional[QueryClassifierService] = None


def get_query_classifier_service() -> QueryClassifierService:
    """Get or create the query classifier service singleton"""
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = QueryClassifierService()
    return _classifier_service
