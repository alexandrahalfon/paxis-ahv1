"""
PubMed Service - Fetch abstracts and metadata from PubMed/NCBI.
"""

import re
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
import httpx
import logging

logger = logging.getLogger(__name__)

PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


class PubMedService:
    """Service for fetching data from PubMed."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize PubMed service.
        
        Args:
            api_key: Optional NCBI API key for higher rate limits
        """
        self.api_key = api_key
        self._client = None
    
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=30)
        return self._client
    
    def fetch_abstract_by_pmid(self, pmid: str) -> Optional[str]:
        """
        Fetch abstract from PubMed using PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            Abstract text or None if not found
        """
        if not pmid:
            return None
        
        # Clean PMID
        pmid = str(pmid).strip()
        if not pmid.isdigit():
            # Try to extract numeric part
            match = re.search(r'\d+', pmid)
            if match:
                pmid = match.group()
            else:
                return None
        
        try:
            client = self._get_client()
            
            params = {
                "db": "pubmed",
                "id": pmid,
                "rettype": "abstract",
                "retmode": "xml"
            }
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = client.get(PUBMED_EFETCH_URL, params=params)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.text)
            
            # Find abstract text - handle structured abstracts
            abstract_parts = []
            for abstract_text in root.findall(".//AbstractText"):
                label = abstract_text.get("Label", "")
                text = abstract_text.text or ""
                
                # Handle nested elements (like <i>, <b>, etc.)
                if not text:
                    text = "".join(abstract_text.itertext())
                
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            
            if abstract_parts:
                return " ".join(abstract_parts)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch abstract for PMID {pmid}: {e}")
            return None
    
    def fetch_abstract_by_doi(self, doi: str) -> Optional[str]:
        """
        Fetch abstract from PubMed using DOI.
        First searches for PMID, then fetches abstract.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Abstract text or None if not found
        """
        if not doi:
            return None
        
        # Clean DOI
        doi = doi.strip()
        if doi.startswith("https://doi.org/"):
            doi = doi[16:]
        elif doi.startswith("http://doi.org/"):
            doi = doi[15:]
        elif doi.startswith("doi:"):
            doi = doi[4:]
        
        try:
            client = self._get_client()
            
            # Search for PMID using DOI
            params = {
                "db": "pubmed",
                "term": f"{doi}[doi]",
                "retmode": "json"
            }
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = client.get(PUBMED_ESEARCH_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            
            if id_list:
                pmid = id_list[0]
                return self.fetch_abstract_by_pmid(pmid)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch abstract for DOI {doi}: {e}")
            return None
    
    def fetch_metadata(self, pmid: str = None, doi: str = None) -> Dict[str, Any]:
        """
        Fetch full metadata from PubMed.
        
        Args:
            pmid: PubMed ID
            doi: DOI (used to find PMID if pmid not provided)
            
        Returns:
            Dictionary with title, authors, abstract, journal, year, etc.
        """
        # Get PMID if only DOI provided
        if not pmid and doi:
            pmid = self._get_pmid_from_doi(doi)
        
        if not pmid:
            return {}
        
        try:
            client = self._get_client()
            
            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml"
            }
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = client.get(PUBMED_EFETCH_URL, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.text)
            article = root.find(".//PubmedArticle")
            
            if not article:
                return {}
            
            # Extract metadata
            metadata = {
                "pmid": pmid,
                "title": "",
                "abstract": "",
                "authors": [],
                "journal": "",
                "year": None,
                "doi": ""
            }
            
            # Title
            title_elem = article.find(".//ArticleTitle")
            if title_elem is not None:
                metadata["title"] = "".join(title_elem.itertext())
            
            # Abstract
            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                label = abstract_text.get("Label", "")
                text = "".join(abstract_text.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            metadata["abstract"] = " ".join(abstract_parts)
            
            # Authors
            for author in article.findall(".//Author"):
                last_name = author.findtext("LastName", "")
                fore_name = author.findtext("ForeName", "")
                if last_name:
                    metadata["authors"].append(f"{last_name} {fore_name}".strip())
            
            # Journal
            journal_elem = article.find(".//Journal/Title")
            if journal_elem is not None:
                metadata["journal"] = journal_elem.text
            
            # Year
            year_elem = article.find(".//PubDate/Year")
            if year_elem is not None:
                try:
                    metadata["year"] = int(year_elem.text)
                except:
                    pass
            
            # DOI
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    metadata["doi"] = article_id.text
                    break
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to fetch metadata for PMID {pmid}: {e}")
            return {}
    
    def _get_pmid_from_doi(self, doi: str) -> Optional[str]:
        """Get PMID from DOI using PubMed search."""
        try:
            client = self._get_client()
            
            params = {
                "db": "pubmed",
                "term": f"{doi}[doi]",
                "retmode": "json"
            }
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = client.get(PUBMED_ESEARCH_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            
            return id_list[0] if id_list else None
            
        except Exception as e:
            logger.warning(f"Failed to get PMID for DOI {doi}: {e}")
            return None
    
    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


# Singleton instance
_pubmed_service: Optional[PubMedService] = None


def get_pubmed_service() -> PubMedService:
    """Get or create PubMed service singleton."""
    global _pubmed_service
    if _pubmed_service is None:
        # Try to get API key from settings
        try:
            from src.core.config import settings
            api_key = getattr(settings, 'ncbi_api_key', None)
        except:
            api_key = None
        _pubmed_service = PubMedService(api_key=api_key)
    return _pubmed_service
