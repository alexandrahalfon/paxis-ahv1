"""
Service for downloading PDFs from various sources (DOI, PMID, URLs).
"""

import requests
from pathlib import Path
from typing import Optional, Dict, Any
import time

from ...core.config import get_settings


class PDFDownloadService:
    """Service for downloading PDFs from medical literature sources."""
    
    def __init__(self):
        """Initialize PDF download service."""
        self.settings = get_settings()
        self.download_dir = Path("downloads")
        self.download_dir.mkdir(exist_ok=True)
    
    def download_from_doi(self, doi: str, filename: Optional[str] = None) -> Optional[Path]:
        """
        Attempt to download PDF from DOI.
        
        Args:
            doi: DOI identifier (e.g., "10.1016/j.ijrobp.2016.11.056")
            filename: Optional filename for saved PDF
            
        Returns:
            Path to downloaded PDF if successful, None otherwise
        """
        if not doi:
            return None
        
        try:
            # Try direct DOI resolution first
            doi_url = f"https://doi.org/{doi}"
            response = requests.get(doi_url, timeout=30, allow_redirects=True, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; Paxis/1.0)'
            })
            
            # Check if redirected to a PDF
            final_url = response.url
            if final_url.endswith('.pdf') or 'application/pdf' in response.headers.get('content-type', ''):
                if not filename:
                    filename = f"{doi.replace('/', '_')}.pdf"
                file_path = self.download_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return file_path
            
            # Try Sci-Hub as fallback (for open access)
            scihub_urls = [
                f"https://sci-hub.se/{doi}",
                f"https://sci-hub.st/{doi}"
            ]
            
            for url in scihub_urls:
                try:
                    response = requests.get(url, timeout=30, allow_redirects=True, headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; Paxis/1.0)'
                    })
                    if response.status_code == 200:
                        # Check if response is PDF
                        content_type = response.headers.get('content-type', '')
                        if 'application/pdf' in content_type or response.content[:4] == b'%PDF':
                            if not filename:
                                filename = f"{doi.replace('/', '_')}.pdf"
                            file_path = self.download_dir / filename
                            with open(file_path, 'wb') as f:
                                f.write(response.content)
                            return file_path
                except:
                    continue
            
            return None
            
        except Exception as e:
            print(f"Error downloading PDF from DOI {doi}: {e}")
            return None
    
    def download_from_pmid(self, pmid: str, filename: Optional[str] = None) -> Optional[Path]:
        """
        Attempt to download PDF from PubMed ID.
        
        Args:
            pmid: PubMed ID
            filename: Optional filename for saved PDF
            
        Returns:
            Path to downloaded PDF if successful, None otherwise
        """
        if not pmid:
            return None
        
        try:
            # Get article info from PubMed to extract DOI
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            fetch_url = f"{base_url}/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml"
            }
            
            response = requests.get(fetch_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML to get DOI
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            doi = None
            for article_id in root.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            if doi:
                return self.download_from_doi(doi, filename)
            
            return None
            
        except Exception as e:
            print(f"Error downloading PDF from PMID {pmid}: {e}")
            return None
    
    def download_from_url(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """
        Download PDF from direct URL.
        
        Args:
            url: Direct URL to PDF
            filename: Optional filename for saved PDF
            
        Returns:
            Path to downloaded PDF if successful, None otherwise
        """
        if not url:
            return None
        
        try:
            response = requests.get(url, timeout=30, allow_redirects=True, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; Paxis/1.0)'
            })
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'application/pdf' not in content_type and not url.endswith('.pdf'):
                # Check if content is actually PDF
                if response.content[:4] != b'%PDF':
                    return None
            
            if not filename:
                # Extract filename from URL or use timestamp
                filename = url.split('/')[-1] or f"downloaded_{int(time.time())}.pdf"
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            file_path = self.download_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return file_path
            
        except Exception as e:
            print(f"Error downloading PDF from URL {url}: {e}")
            return None
    
    def download_study_pdf(self, study: Dict[str, Any]) -> Optional[Path]:
        """
        Attempt to download PDF for a study using available identifiers.
        
        Tries in order: DOI → PMID → Direct URL
        
        Args:
            study: Study dictionary with doi, pmid, or url fields
            
        Returns:
            Path to downloaded PDF if successful, None otherwise
        """
        # Generate filename from study title
        title = study.get("title", "study")
        filename = "".join(c for c in title[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = filename.replace(' ', '_') + '.pdf'
        
        # Try DOI first
        if study.get("doi"):
            pdf_path = self.download_from_doi(study["doi"], filename)
            if pdf_path and pdf_path.exists():
                return pdf_path
        
        # Try PMID
        if study.get("pmid"):
            pdf_path = self.download_from_pmid(study["pmid"], filename)
            if pdf_path and pdf_path.exists():
                return pdf_path
        
        # Try direct URL if provided
        if study.get("url"):
            pdf_path = self.download_from_url(study["url"], filename)
            if pdf_path and pdf_path.exists():
                return pdf_path
        
        return None
