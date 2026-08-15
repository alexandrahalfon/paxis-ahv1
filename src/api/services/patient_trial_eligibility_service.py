"""
Patient Trial Eligibility Service

New approach for patient-to-trial matching:
1. Send patient query through EnhancedRAGService to find relevant studies
2. For each top study, query it individually asking "Would this patient be eligible?"
3. Return only studies where the patient is eligible, with reasoning

This replaces the semantic similarity approach with explicit eligibility checking.
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from openai import OpenAI
from qdrant_client import models as qm

from src.core.config import settings


@dataclass
class EligibilityResult:
    """Result of eligibility check for a single study."""
    doc_id: str
    title: str
    is_eligible: bool
    match_category: str  # "yes", "partial", "no"
    reasoning: str
    confidence: str  # "high", "medium", "low"
    eligibility_criteria_matched: List[str]
    eligibility_criteria_not_met: List[str]
    doi: Optional[str] = None
    year: Optional[int] = None
    citation: Optional[str] = None


class PatientTrialEligibilityService:
    """
    Service for matching patients to trials using explicit eligibility checking.
    
    Flow:
    1. Use EnhancedRAGService to find candidate studies
    2. For each study, retrieve its full content
    3. Ask LLM: "Would this patient be eligible for this trial?"
    4. Return eligible studies with reasoning
    """
    
    # Anatomical site keywords - these take priority over treatment keywords
    # Ordered by specificity (more specific sites first)
    ANATOMICAL_SITE_KEYWORDS = {
        "H&N": [
            "oral tongue", "tongue cancer", "oral cavity", "oropharynx", "oropharyngeal",
            "nasopharynx", "nasopharyngeal", "larynx", "laryngeal", "hypopharynx",
            "tonsil", "tonsillar", "base of tongue", "floor of mouth", "buccal",
            "palate", "pharynx", "pharyngeal", "salivary", "parotid", "neck dissection",
            "glossectomy", "head and neck", "scc of the", "squamous cell carcinoma of the oral",
            "squamous cell carcinoma of the tongue",
        ],
        "Breast": [
            "breast cancer", "breast carcinoma", "mastectomy", "lumpectomy", "dcis",
            "lobular carcinoma", "ductal carcinoma", "her2+", "her2-positive",
            "triple negative breast", "axillary dissection",
        ],
        "Lung": [
            "lung cancer", "lung carcinoma", "nsclc", "sclc", "non-small cell lung",
            "small cell lung", "bronchogenic", "pulmonary nodule", "lung adenocarcinoma",
            "lung squamous", "mesothelioma",
        ],
        "Prostate": [
            "prostate cancer", "prostate carcinoma", "prostatectomy", "gleason",
            "psa screening", "prostate biopsy", "prostate adenocarcinoma",
        ],
        "GI": [
            "colon cancer", "colorectal", "rectal cancer", "esophageal cancer",
            "gastric cancer", "pancreatic cancer", "hepatocellular", "liver cancer",
            "anal cancer", "cholangiocarcinoma", "hemicolectomy", "colectomy",
        ],
        "GYN": [
            "cervical cancer", "endometrial cancer", "ovarian cancer", "uterine cancer",
            "vulvar cancer", "vaginal cancer", "hysterectomy",
        ],
        "GU": [
            "bladder cancer", "renal cell", "kidney cancer", "urothelial",
            "testicular cancer", "seminoma",
        ],
        "CNS": [
            "brain tumor", "glioblastoma", "glioma", "gbm", "meningioma",
            "brain metastases", "craniotomy",
        ],
        "Cutaneous": [
            "melanoma", "skin cancer", "basal cell carcinoma", "merkel cell",
            "cutaneous squamous",
        ],
        "Lymphoma": [
            "lymphoma", "hodgkin", "non-hodgkin", "dlbcl", "follicular lymphoma",
        ],
        "Sarcoma": [
            "sarcoma", "soft tissue sarcoma", "osteosarcoma", "ewing sarcoma",
        ],
    }
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self._rag_service = None
    
    def _get_rag_service(self):
        """Lazy load RAG service."""
        if self._rag_service is None:
            from src.api.services.enhanced_rag_service import get_enhanced_rag_service
            self._rag_service = get_enhanced_rag_service()
        return self._rag_service
    
    def _infer_cancer_site(self, patient_description: str) -> Optional[str]:
        """
        Infer cancer site from patient description using anatomical keywords.
        Prioritizes specific anatomical terms over treatment keywords.
        """
        text_lower = patient_description.lower()
        
        # Score each site based on keyword matches
        site_scores = {}
        for site, keywords in self.ANATOMICAL_SITE_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in text_lower:
                    # Longer keywords get higher scores (more specific)
                    score += len(kw.split())
            if score > 0:
                site_scores[site] = score
        
        if site_scores:
            # Return the site with highest score
            best_site = max(site_scores, key=site_scores.get)
            print(f"[Eligibility] Site inference scores: {site_scores}")
            return best_site
        
        return None
    
    async def find_eligible_trials(
        self,
        patient_description: str,
        top_k: int = 10,
        max_eligible: int = 5,
        min_yes_matches: int = 2,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find trials where the patient would be eligible.
        
        Args:
            patient_description: Free-text patient description
            top_k: Number of candidate studies to evaluate per batch
            max_eligible: Maximum total studies to return (yes + partial)
            min_yes_matches: Minimum "yes" matches to find before stopping (default 2)
            category: Optional cancer category filter
            
        Returns:
            Dict with eligible_trials (sorted: yes first, then partial), patient_summary, total_evaluated
        """
        print(f"[Eligibility] Starting eligibility check for patient...")
        
        # Step 1: Use RAG to find candidate studies
        rag_service = self._get_rag_service()
        
        # Normalize category to Qdrant format (e.g., "h&n" -> "h&n_processed_documents")
        # If no category provided, infer from patient description with priority for anatomical sites
        normalized_category = None
        if category:
            from src.api.services.enhanced_rag_service import normalize_category_filter
            normalized_category = normalize_category_filter(category)
            print(f"[Eligibility] Category: {category} -> {normalized_category}")
        else:
            # Auto-infer category from patient description
            # Use custom inference that prioritizes anatomical site keywords over treatment keywords
            inferred_site = self._infer_cancer_site(patient_description)
            if inferred_site:
                normalized_category = f"{inferred_site.lower()}_processed_documents"
                print(f"[Eligibility] Auto-inferred category: {inferred_site} -> {normalized_category}")
            else:
                print(f"[Eligibility] No category filter (could not infer from patient description)")
        
        # Build query for finding relevant studies
        search_query = f"Find clinical trials for: {patient_description}"
        
        print(f"[Eligibility] Step 1: Finding candidate studies...")
        
        # Use the RAG service's query expansion and embedding, but query Qdrant directly
        # to get more unique studies (the retriever's dedup_and_caps limits to 2 chunks per doc)
        retriever = rag_service.retriever
        
        # Apply query expansion (same as RAG pipeline)
        from src.api.services.enhanced_rag_service import expand_query
        expanded_query = expand_query(search_query)
        print(f"[Eligibility] Expanded query: {expanded_query[:150]}...")
        
        # Embed the expanded query
        qvec = retriever.embed_query(expanded_query)
        
        # Build category filter if specified
        qdrant_filter = None
        if normalized_category:
            qdrant_filter = qm.Filter(must=[
                qm.FieldCondition(key="category", match=qm.MatchValue(value=normalized_category))
            ])
        
        # Query Qdrant directly with a high limit to get many candidates
        # This bypasses the retriever's dedup_and_caps which limits to 2 chunks per doc
        search_results = retriever.qdrant.query_points(
            collection_name=retriever.collection,
            query=qvec,
            limit=500,  # Get 500 chunks to find many unique studies
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        ).points
        
        print(f"[Eligibility] Found {len(search_results)} chunks from Qdrant")
        
        # Deduplicate by doc_id to get unique studies
        seen_doc_ids = set()
        candidate_studies = []
        for point in search_results:
            payload = point.payload
            doc_meta = payload.get("doc_meta", {})
            doc_id = payload.get("doc_id")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                candidate_studies.append({
                    "doc_id": doc_id,
                    "title": doc_meta.get("title") or payload.get("title") or "Unknown",
                    "doi": doc_meta.get("doi"),
                    "year": doc_meta.get("year"),
                    "citation": doc_meta.get("citation"),
                    "text": payload.get("text", ""),
                    "score": point.score,
                })
        
        print(f"[Eligibility] Found {len(candidate_studies)} unique studies to evaluate")
        print(f"[Eligibility] Step 2: Evaluating studies for eligibility (need {min_yes_matches} 'yes' matches)...")
        
        # Step 2: Check eligibility for each study
        # Keep going until we find min_yes_matches "yes" matches or exhaust candidates
        yes_matches = []
        partial_matches = []
        no_matches = []
        total_evaluated = 0
        max_to_evaluate = len(candidate_studies)  # Evaluate all if needed
        
        # Process studies in parallel (batches of 3 to avoid rate limits)
        batch_size = 3
        for i in range(0, max_to_evaluate, batch_size):
            batch = candidate_studies[i:i + batch_size]
            tasks = [
                self._check_eligibility(study, patient_description)
                for study in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"[Eligibility] Error checking eligibility: {result}")
                    continue
                if result is None:
                    continue
                    
                total_evaluated += 1
                
                # Categorize the result
                if result.match_category == "yes":
                    yes_matches.append(result)
                    print(f"[Eligibility] YES match: {result.title[:40]}...")
                elif result.match_category == "partial":
                    partial_matches.append(result)
                    print(f"[Eligibility] PARTIAL match: {result.title[:40]}...")
                else:
                    no_matches.append(result)
                    print(f"[Eligibility] NO match: {result.title[:40]}...")
            
            # Check if we can stop early (have minimum yes matches AND enough total)
            # But keep going if we haven't found enough yes matches yet
            if len(yes_matches) >= min_yes_matches:
                # We have enough yes matches - can stop if we also have enough total
                if len(yes_matches) + len(partial_matches) >= max_eligible:
                    print(f"[Eligibility] Found {len(yes_matches)} 'yes' matches and {len(partial_matches)} partial, stopping search")
                    break
            # If we don't have enough yes matches, keep going through all candidates
        
        print(f"[Eligibility] Results: {len(yes_matches)} yes, {len(partial_matches)} partial, {len(no_matches)} no (evaluated {total_evaluated})")
        
        # Combine results: yes first, then partial (up to max_eligible)
        eligible_trials = yes_matches + partial_matches
        eligible_trials = eligible_trials[:max_eligible]
        
        # Build patient summary
        patient_summary = self._build_patient_summary(patient_description)
        
        return {
            "eligible_trials": [self._result_to_dict(r) for r in eligible_trials],
            "patient_summary": patient_summary,
            "total_evaluated": total_evaluated,
            "total_eligible": len(eligible_trials),
            "yes_count": len(yes_matches),
            "partial_count": len(partial_matches),
        }
    
    async def _check_eligibility(
        self,
        study: Dict[str, Any],
        patient_description: str,
    ) -> Optional[EligibilityResult]:
        """
        Check if patient is eligible for a specific study.
        
        Uses the SAME pipeline as the /query/study endpoint ("Have a question about this trial?"):
        1. Retrieve study chunks using the same multi-strategy approach
        2. Use the same Q&A system prompt
        3. Parse the prose response to determine eligibility
        """
        doc_id = study.get("doc_id")
        title = study.get("title", "Unknown")
        doi = study.get("doi")
        
        print(f"[Eligibility] Checking: {title[:50]}...")
        
        try:
            # Get full study content using the SAME retrieval logic as /query/study
            rag_service = self._get_rag_service()
            retriever = rag_service.retriever
            
            # Build the eligibility question (same format as user would ask in study Q&A)
            # Use "match the patients enrolled" instead of "eligible for this trial" because
            # many studies are retrospective analyses, not prospective trials with eligibility criteria
            eligibility_question = f"{patient_description}. Does this patient match the patients enrolled in this study?"
            
            # Embed the question (same as /query/study)
            qvec = retriever.embed_query(eligibility_question)
            
            # Use the SAME multi-strategy retrieval as /query/study endpoint
            hits = await self._get_study_chunks_multi_strategy(
                retriever=retriever,
                doc_id=doc_id,
                doi=doi,
                title=title,
                qvec=qvec
            )
            
            if not hits:
                print(f"[Eligibility] No chunks found for {doc_id}")
                return None
            
            # Extract study info and chunks (same as /query/study)
            study_title = title
            study_chunks = []
            
            for hit in hits:
                payload = hit.get("payload", hit)
                # Try multiple places for title
                if study_title == "Unknown":
                    study_title = (
                        payload.get("title") or 
                        payload.get("doc_meta", {}).get("title") or
                        payload.get("doc_meta", {}).get("study_name") or
                        study_title
                    )
                
                study_chunks.append({
                    "text": payload.get("text", ""),
                    "section": payload.get("section", ""),
                    "chunk_type": payload.get("chunk_type", ""),
                    "score": hit.get("score", 1.0)
                })
            
            # Format context from chunks (top 15 most relevant) - SAME as /query/study
            context_parts = []
            for i, chunk in enumerate(study_chunks[:15], 1):
                section = chunk.get("section", "")
                text = chunk.get("text", "")
                if text:
                    context_parts.append(f"[{i}] {section}:\n{text[:1000]}")
            
            study_context = "\n\n".join(context_parts)
            
            # Use the SAME system prompt as /query/study endpoint
            system_prompt = """You are a clinical research assistant helping answer questions about a specific clinical study.
You have access to the study content and should answer questions based ONLY on the information provided.

RULES:
1. Answer based ONLY on the study information provided
2. If the information is not available in the study data, say so clearly
3. Be specific and cite relevant details from the study
4. Keep answers concise but informative
5. If asked about something not covered in the study, acknowledge the limitation
"""

            user_prompt = f"""STUDY: {study_title}

STUDY CONTENT:
{study_context}

QUESTION: {eligibility_question}

Please answer the question based on the study information provided."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"[Eligibility] Study Q&A response for {title[:30]}: {answer[:100]}...")
            
            # Parse the prose response to determine match category
            match_category, confidence, criteria_matched, criteria_not_met = self._parse_eligibility_response(answer)
            
            return EligibilityResult(
                doc_id=doc_id,
                title=study_title,
                is_eligible=(match_category == "yes"),  # For backwards compatibility
                match_category=match_category,
                reasoning=answer,  # Full prose response as reasoning
                confidence=confidence,
                eligibility_criteria_matched=criteria_matched,
                eligibility_criteria_not_met=criteria_not_met,
                doi=study.get("doi"),
                year=study.get("year"),
                citation=study.get("citation"),
            )
            
        except Exception as e:
            print(f"[Eligibility] Error checking {doc_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_eligibility_response(self, answer: str) -> tuple:
        """
        Parse the prose response from study Q&A to determine if patient matches study population.
        
        Returns: (match_category, confidence, criteria_matched, criteria_not_met)
        
        match_category is one of:
        - "yes": Clear match, patient fits the study population
        - "partial": Some criteria match but not all, or uncertain
        - "no": Clear mismatch, patient does not fit
        """
        answer_lower = answer.lower()
        
        # Strong indicators patient does NOT match at all
        no_match_phrases = [
            "would not be eligible",
            "not be eligible",
            "not eligible",
            "does not match the",
            "doesn't match the",
            "does not fit the",
            "doesn't fit the",
            "excluded from",
            "meets exclusion criteria",
            "ineligible for",
            "cannot participate",
            "would not qualify",
            "does not qualify",
            "disqualifies",
            "disqualified",
            "would not have been included",
            "would not have been enrolled",
            "not similar to the patients",
            "different from the patients",
            "outside the eligibility",
            "no, the patient does not",
            "no, this patient does not",
        ]
        
        # Indicators of PARTIAL match (some criteria met, some not)
        partial_match_phrases = [
            "partially matches",
            "does not fully match",
            "doesn't fully match",
            "does not fully meet",
            "doesn't fully meet",
            "not fully match",
            "only partially",
            "some criteria",
            "most criteria",
            "not all criteria",
            "partially meets",
            "partially fits",
            "some aspects match",
            "with some differences",
            "close but",
            "similar but not",
            "appears to match",  # Hedged language
            "may match",
            "could potentially",
        ]
        
        # Strong indicators patient DOES match - must be unqualified positive statements
        yes_match_phrases = [
            "yes, the patient matches",
            "yes, this patient matches",
            "the patient matches the",
            "patient does match the",
            "patient matches the eligibility",
            "patient matches the criteria",
            "patient matches the characteristics",
            "would be eligible",
            "is eligible for",
            "meets the criteria",
            "meets all the criteria",
            "meets all criteria",
            "meets the inclusion criteria",
            "meets the eligibility criteria",
            "qualifies for the",
            "would qualify for",
            "fits the criteria",
            "fits the eligibility",
            "could participate in",
            "would have been included",
            "would have been enrolled",
            "similar to the patients enrolled",
            "fits the profile of",
            "within the study population",
            "consistent with the enrolled",
            "matches the enrolled population",
            "matches the study population",
            "yes, based on the",
            "therefore, the patient matches",
            "the patient appears to match",
            "aligns with the study",
            "fits within the",
        ]
        
        # Indicators of uncertainty or inability to determine
        uncertain_phrases = [
            "cannot determine",
            "unable to determine",
            "not enough information",
            "insufficient information",
            "no mention of eligibility",
            "does not include specific eligibility",
            "not specified in the study",
            "unclear from the study",
            "uncertain whether",
            "cannot say definitively",
            "hard to say",
            "difficult to determine",
            "no specific eligibility criteria",
            "does not provide specific criteria",
        ]
        
        # Check for each category
        is_no_match = any(phrase in answer_lower for phrase in no_match_phrases)
        is_partial = any(phrase in answer_lower for phrase in partial_match_phrases)
        is_yes_match = any(phrase in answer_lower for phrase in yes_match_phrases)
        is_uncertain = any(phrase in answer_lower for phrase in uncertain_phrases)
        
        # Also check the conclusion (last 200 chars) for definitive statements
        conclusion = answer_lower[-200:] if len(answer_lower) > 200 else answer_lower
        conclusion_yes = any(phrase in conclusion for phrase in [
            "matches the", "is eligible", "would qualify", "fits the", 
            "meets the criteria", "patient matches", "therefore, the patient"
        ])
        conclusion_no = any(phrase in conclusion for phrase in [
            "does not match", "not eligible", "would not qualify", 
            "does not fit", "not similar", "excluded"
        ])
        
        # Determine match category
        # Priority: conclusion > explicit signals > default
        if conclusion_yes and not conclusion_no and not is_no_match:
            # Conclusion says yes and no contradicting signals
            match_category = "yes"
            confidence = "high"
        elif conclusion_no and not conclusion_yes:
            # Conclusion says no
            match_category = "no"
            confidence = "high"
        elif is_no_match and not is_yes_match:
            # Clear no match signal
            match_category = "no"
            confidence = "high"
        elif is_partial or (is_no_match and is_yes_match):
            # Partial match (explicitly stated or mixed signals)
            match_category = "partial"
            confidence = "medium"
        elif is_yes_match and not is_no_match:
            # Clear yes match
            match_category = "yes"
            confidence = "high"
        elif is_uncertain:
            # Can't determine - treat as partial (might be relevant)
            match_category = "partial"
            confidence = "low"
        else:
            # No clear signal - default to partial (let user decide)
            match_category = "partial"
            confidence = "low"
        
        # Extract criteria (simplified - just note what was mentioned)
        criteria_matched = []
        criteria_not_met = []
        
        # Look for specific criteria mentions
        if "node-negative" in answer_lower or "n0" in answer_lower:
            if "n1mi" in answer_lower or "micrometast" in answer_lower:
                criteria_not_met.append("Nodal status: patient has N1mi but study enrolled node-negative patients")
            elif match_category == "yes":
                criteria_matched.append("Nodal status matches")
        
        if "er+" in answer_lower or "er-positive" in answer_lower or "hormone receptor" in answer_lower:
            if match_category == "yes":
                criteria_matched.append("Receptor status matches")
            elif match_category == "no":
                criteria_not_met.append("Receptor status may not match")
        
        if "stage" in answer_lower:
            if match_category == "yes":
                criteria_matched.append("Stage criteria met")
            elif match_category == "no":
                criteria_not_met.append("Stage criteria not met")
        
        return match_category, confidence, criteria_matched, criteria_not_met
    
    async def _get_study_chunks(
        self,
        retriever,
        doc_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a specific study from Qdrant."""
        try:
            # Try exact doc_id match
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
            ])
            
            # Use scroll to get all chunks for this study
            results, _ = retriever.qdrant.scroll(
                collection_name=retriever.collection,
                scroll_filter=doc_id_filter,
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            
            if results:
                return [{"text": p.payload.get("text", ""), "section": p.payload.get("section", "")} for p in results]
            
            # Fallback: try doc_id_raw
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="doc_id_raw", match=qm.MatchValue(value=doc_id))
            ])
            results, _ = retriever.qdrant.scroll(
                collection_name=retriever.collection,
                scroll_filter=doc_id_filter,
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            
            if results:
                return [{"text": p.payload.get("text", ""), "section": p.payload.get("section", "")} for p in results]
            
            return []
            
        except Exception as e:
            print(f"[Eligibility] Error getting chunks for {doc_id}: {e}")
            return []
    
    async def _get_study_chunks_multi_strategy(
        self,
        retriever,
        doc_id: str,
        doi: Optional[str],
        title: str,
        qvec: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Get study chunks using the SAME multi-strategy approach as /query/study endpoint.
        
        Tries multiple filter strategies in order:
        1. Exact match on doc_id
        2. doc_id_raw field
        3. source_doc_dir_name field
        4. DOI converted to doc_id_raw format
        5. DOI as source_doc_dir_name
        6. Scroll and filter manually
        7. Title-based semantic search
        """
        hits = []
        
        # Strategy 1: Exact match on doc_id (includes hash suffix)
        if doc_id:
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
            ])
            results = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            if results:
                hits = [{"payload": p.payload, "score": p.score} for p in results]
                print(f"[Eligibility] Strategy 1 (exact doc_id): {len(hits)} hits")
        
        # Strategy 2: Try doc_id_raw field (without hash suffix)
        if not hits and doc_id:
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="doc_id_raw", match=qm.MatchValue(value=doc_id))
            ])
            results = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            if results:
                hits = [{"payload": p.payload, "score": p.score} for p in results]
                print(f"[Eligibility] Strategy 2 (doc_id_raw): {len(hits)} hits")
        
        # Strategy 3: Try source_doc_dir_name field
        if not hits and doc_id:
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="source_doc_dir_name", match=qm.MatchValue(value=doc_id))
            ])
            results = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            if results:
                hits = [{"payload": p.payload, "score": p.score} for p in results]
                print(f"[Eligibility] Strategy 3 (source_doc_dir_name): {len(hits)} hits")
        
        # Strategy 4: Convert DOI to doc_id_raw format and try
        if not hits and doi:
            # Convert DOI "10.1200/jco.2014.59.5132" -> "doi_10.1200_jco.2014.59.5132"
            doi_as_doc_id = "doi_" + doi.replace("/", "_")
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="doc_id_raw", match=qm.MatchValue(value=doi_as_doc_id))
            ])
            results = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            if results:
                hits = [{"payload": p.payload, "score": p.score} for p in results]
                print(f"[Eligibility] Strategy 4 (doi->doc_id_raw '{doi_as_doc_id}'): {len(hits)} hits")
        
        # Strategy 5: Try source_doc_dir_name with DOI format
        if not hits and doi:
            doi_as_dir = "doi_" + doi.replace("/", "_")
            doc_id_filter = qm.Filter(must=[
                qm.FieldCondition(key="source_doc_dir_name", match=qm.MatchValue(value=doi_as_dir))
            ])
            results = retriever.qdrant.query_points(
                collection_name=retriever.collection,
                query=qvec,
                limit=50,
                query_filter=doc_id_filter,
                with_payload=True,
                with_vectors=False,
            ).points
            if results:
                hits = [{"payload": p.payload, "score": p.score} for p in results]
                print(f"[Eligibility] Strategy 5 (doi->source_doc_dir_name '{doi_as_dir}'): {len(hits)} hits")
        
        # Strategy 6: Scroll and filter manually (last resort before title search)
        if not hits:
            print(f"[Eligibility] Strategy 6: Scroll-based search")
            try:
                scroll_results, _ = retriever.qdrant.scroll(
                    collection_name=retriever.collection,
                    limit=200,
                    with_payload=True,
                    with_vectors=False,
                )
                
                # Build search terms
                search_terms = []
                if doc_id:
                    search_terms.append(doc_id.lower())
                if doi:
                    search_terms.append(doi.lower())
                    search_terms.append(("doi_" + doi.replace("/", "_")).lower())
                
                # Filter by doc_id/doc_id_raw/source_doc_dir_name containing any search term
                matching_points = []
                for point in scroll_results:
                    point_doc_id = (point.payload.get("doc_id") or "").lower()
                    doc_id_raw = (point.payload.get("doc_id_raw") or "").lower()
                    source_dir = (point.payload.get("source_doc_dir_name") or "").lower()
                    
                    for term in search_terms:
                        if term and (term in point_doc_id or term == doc_id_raw or term == source_dir):
                            matching_points.append(point)
                            break
                
                if matching_points:
                    hits = [{"payload": p.payload, "score": 1.0} for p in matching_points[:50]]
                    print(f"[Eligibility] Strategy 6 found {len(hits)} matching points")
            except Exception as e:
                print(f"[Eligibility] Strategy 6 failed: {e}")
        
        # Strategy 7: Search by title using semantic search (fallback)
        if not hits and title and title != "Unknown":
            print(f"[Eligibility] Strategy 7: Title-based search")
            try:
                title_vec = retriever.embed_query(title)
                title_hits = retriever.qdrant.query_points(
                    collection_name=retriever.collection,
                    query=title_vec,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                ).points
                
                # Filter to only include chunks where the title matches closely
                title_lower = title.lower()
                stop_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'study', 'trial', 'phase'}
                title_words = [w for w in title_lower.split() if len(w) >= 4 and w not in stop_words][:5]
                
                matching_hits = []
                for hit in title_hits:
                    payload = hit.payload
                    chunk_title = (payload.get("title") or payload.get("doc_meta", {}).get("title") or "").lower()
                    chunk_text = (payload.get("text") or "").lower()
                    
                    # Check if enough title words appear in the chunk's title or text
                    matches = sum(1 for w in title_words if w in chunk_title or w in chunk_text[:500])
                    if matches >= min(3, len(title_words)):
                        matching_hits.append(hit)
                
                if matching_hits:
                    hits = [{"payload": p.payload, "score": p.score} for p in matching_hits[:50]]
                    print(f"[Eligibility] Strategy 7 found {len(hits)} matching points by title")
            except Exception as e:
                print(f"[Eligibility] Strategy 7 failed: {e}")
        
        return hits
    
    def _build_patient_summary(self, patient_description: str) -> str:
        """Build a brief patient summary from the description."""
        # Take first 200 chars or first sentence
        if len(patient_description) <= 200:
            return patient_description
        
        first_period = patient_description.find('. ')
        if first_period > 0 and first_period < 200:
            return patient_description[:first_period + 1]
        
        return patient_description[:200] + "..."
    
    def _result_to_dict(self, result: EligibilityResult) -> Dict[str, Any]:
        """Convert EligibilityResult to dictionary."""
        return {
            "doc_id": result.doc_id,
            "title": result.title,
            "is_eligible": result.is_eligible,
            "match_category": result.match_category,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "eligibility_criteria_matched": result.eligibility_criteria_matched,
            "eligibility_criteria_not_met": result.eligibility_criteria_not_met,
            "doi": result.doi,
            "year": result.year,
            "citation": result.citation,
        }


# Singleton instance
_eligibility_service = None


def get_eligibility_service() -> PatientTrialEligibilityService:
    """Get singleton eligibility service instance."""
    global _eligibility_service
    if _eligibility_service is None:
        _eligibility_service = PatientTrialEligibilityService()
    return _eligibility_service
