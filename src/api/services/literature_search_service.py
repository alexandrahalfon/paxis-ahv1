"""
Service for searching medical literature databases for new studies.
"""

import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

from ...core.config import get_settings


class LiteratureSearchService:
    """Service for searching medical literature databases."""
    
    def __init__(self):
        """Initialize literature search service."""
        self.settings = get_settings()
    
    def search_pubmed(
        self, 
        query: str, 
        max_results: int = 50,
        date_from: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed for articles.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            date_from: Date filter (YYYY/MM/DD format)
            
        Returns:
            List of article dictionaries
        """
        try:
            # PubMed E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            
            # Step 1: Search
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "xml",
                "sort": "pub_date",
                "datetype": "pdat"
            }
            
            if date_from:
                # date_from is already in YYYY/MM/DD format
                search_params["mindate"] = date_from
                search_params["maxdate"] = datetime.now().strftime("%Y/%m/%d")
            
            search_url = f"{base_url}/esearch.fcgi"
            search_response = requests.get(search_url, params=search_params, timeout=30)
            search_response.raise_for_status()
            
            # Parse search results
            root = ET.fromstring(search_response.content)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            
            if not pmids:
                return []
            
            # Step 2: Fetch details
            fetch_url = f"{base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids[:max_results]),
                "retmode": "xml"
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
            fetch_response.raise_for_status()
            
            # Parse article details
            articles = self._parse_pubmed_xml(fetch_response.content)
            
            return articles
            
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: bytes) -> List[Dict[str, Any]]:
        """Parse PubMed XML response."""
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    # Authors
                    authors = []
                    for author in article.findall(".//Author"):
                        last = author.find("LastName")
                        first = author.find("ForeName")
                        if last is not None and first is not None:
                            authors.append(f"{last.text} {first.text}")
                    
                    # Journal
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else "Unknown"
                    
                    # Publication date
                    pub_date_elem = article.find(".//PubDate")
                    year = None
                    if pub_date_elem is not None:
                        year_elem = pub_date_elem.find("Year")
                        if year_elem is not None:
                            year = int(year_elem.text)
                    
                    # DOI
                    doi = None
                    for article_id in article.findall(".//ArticleId"):
                        if article_id.get("IdType") == "doi":
                            doi = article_id.text
                            break
                    
                    # Abstract
                    abstract_elem = article.find(".//Abstract/AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # PMID
                    pmid_elem = article.find(".//MedlineCitation/PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else None
                    
                    articles.append({
                        "title": title,
                        "authors": authors,
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                        "pmid": pmid,
                        "abstract": abstract,
                        "source": "PubMed"
                    })
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error parsing PubMed XML: {e}")
        
        return articles
    
    def search_clinical_trials(
        self,
        condition: str,
        intervention: Optional[str] = None,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search ClinicalTrials.gov.
        
        Args:
            condition: Medical condition (e.g., "lung cancer")
            intervention: Intervention type (e.g., "radiation therapy")
            max_results: Maximum number of results
            
        Returns:
            List of trial dictionaries
        """
        try:
            base_url = "https://clinicaltrials.gov/api/v2/studies"
            
            query_parts = [f"CONDITION:{condition}"]
            if intervention:
                query_parts.append(f"INTERVENTION:{intervention}")
            
            params = {
                "query.term": " AND ".join(query_parts),
                "format": "json",
                "pageSize": min(max_results, 100)
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            trials = []
            
            for study in data.get("studies", [])[:max_results]:
                protocol = study.get("protocolSection", {})
                identification = protocol.get("identificationModule", {})
                status = protocol.get("statusModule", {})
                
                trials.append({
                    "title": identification.get("briefTitle", "No title"),
                    "nct_id": identification.get("nctId", ""),
                    "status": status.get("overallStatus", "Unknown"),
                    "condition": condition,
                    "intervention": intervention,
                    "source": "ClinicalTrials.gov"
                })
            
            return trials
            
        except Exception as e:
            print(f"Error searching ClinicalTrials.gov: {e}")
            return []

    def search_clinical_trials_by_query(
        self,
        query_term: str,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search ClinicalTrials.gov with a prebuilt query.term string.
        """
        if not query_term:
            return []
        try:
            base_url = "https://clinicaltrials.gov/api/v2/studies"
            params = {
                "query.term": query_term,
                "format": "json",
                "pageSize": min(max_results, 100)
            }
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("studies", [])[:max_results]
        except Exception as e:
            print(f"Error searching ClinicalTrials.gov (query.term): {e}")
            return []

    def search_clinical_trials_structured(
        self,
        condition: Optional[str] = None,
        other_terms: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search ClinicalTrials.gov using structured query params.

        Uses ``query.cond`` for the condition/disease (produces much better
        results than stuffing everything into ``query.term``).  Additional
        terms like markers, stage, histology go into ``query.term``.

        Falls back progressively: condition+terms → condition-only → terms-only.
        """
        if not condition and not other_terms:
            return []

        base_url = "https://clinicaltrials.gov/api/v2/studies"

        def _fetch(params: dict) -> List[Dict[str, Any]]:
            params.update({"format": "json", "pageSize": min(max_results, 100)})
            print(f"[ClinicalTrials.gov] params={params}")
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("studies", [])[:max_results]

        try:
            # Primary: condition + terms
            if condition and other_terms:
                results = _fetch({"query.cond": condition, "query.term": other_terms})
                if results:
                    return results

            # Fallback 1: condition only
            if condition:
                results = _fetch({"query.cond": condition})
                if results:
                    return results

            # Fallback 2: terms only
            if other_terms:
                results = _fetch({"query.term": other_terms})
                if results:
                    return results

            return []
        except Exception as e:
            print(f"Error searching ClinicalTrials.gov (structured): {e}")
            return []
    
    def search_radiation_oncology(
        self,
        cancer_type: Optional[str] = None,
        days_back: int = 30,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for new radiation oncology studies.
        
        Args:
            cancer_type: Specific cancer type to filter (e.g., "lung cancer")
            days_back: How many days back to search
            max_results: Maximum number of results
            
        Returns:
            List of study dictionaries
        """
        # Build query
        query_parts = ["radiation oncology", "radiotherapy", "radiation therapy"]
        
        if cancer_type:
            query_parts.append(cancer_type)
        
        query = " AND ".join(query_parts)
        
        # Date filter
        date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        
        # Search PubMed
        pubmed_results = self.search_pubmed(query, max_results=max_results, date_from=date_from)
        
        # Search ClinicalTrials.gov if cancer type specified
        clinical_trials = []
        if cancer_type:
            clinical_trials = self.search_clinical_trials(
                condition=cancer_type,
                intervention="radiation therapy",
                max_results=max_results
            )
        
        # Combine results
        all_results = pubmed_results + clinical_trials
        
        return all_results
