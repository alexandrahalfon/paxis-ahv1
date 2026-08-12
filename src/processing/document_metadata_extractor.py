#!/usr/bin/env python3
"""
Document Metadata Extractor

Extracts bibliographic and structural metadata from clinical trial PDFs.
Creates a "Document Index" with all metadata, separate from content indexing.

This captures:
- Title, authors, institutions
- Journal/publication info
- Dates and identifiers
- Trial registration details
- Funding and conflicts of interest
- Document structure info
"""

import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class DocumentMetadataExtractor:
    """Extract and structure document-level metadata."""
    
    def __init__(self, document_title: str = None):
        """Initialize metadata extractor."""
        self.metadata = {
            "document_id": None,
            "extraction_timestamp": datetime.now().isoformat(),
            "document_info": {},
            "publication_info": {},
            "trial_info": {},
            "authors": [],
            "institutions": [],
            "funding": {},
            "competing_interests": [],
        }
        if document_title:
            self.metadata["document_id"] = self._generate_document_id(document_title)
    
    def _generate_document_id(self, title: str) -> str:
        """Generate unique document ID from title."""
        # Remove special chars, lowercase, replace spaces with underscores
        doc_id = re.sub(r'[^a-z0-9]', '_', title.lower())
        # Remove multiple underscores
        doc_id = re.sub(r'_+', '_', doc_id)
        # Remove leading/trailing underscores
        doc_id = doc_id.strip('_')
        return doc_id
    
    def extract_document_info(
        self,
        title: str,
        authors: List[str],
        abstract: str = None,
        total_pages: int = None,
        keywords: List[str] = None,
    ) -> Dict:
        """
        Extract basic document information.
        
        Args:
            title: Document title
            authors: List of author names
            abstract: Document abstract/summary
            total_pages: Total number of pages
            keywords: List of keywords
        
        Returns:
            Metadata dictionary with document info
        """
        self.metadata["document_info"] = {
            "title": title,
            "authors": authors,
            "total_pages": total_pages,
            "abstract": abstract,
            "keywords": keywords or [],
        }
        
        # Generate/update document ID
        if not self.metadata["document_id"]:
            self.metadata["document_id"] = self._generate_document_id(title)
        
        return self.metadata
    
    def extract_publication_info(
        self,
        journal_name: str,
        volume: str = None,
        issue: str = None,
        pages: str = None,
        publication_date: str = None,
        online_date: str = None,
        doi: str = None,
        issn: str = None,
        publisher: str = None,
    ) -> Dict:
        """
        Extract publication/journal information.
        
        Args:
            journal_name: Journal name (e.g., "Lancet Oncology")
            volume: Volume number
            issue: Issue number
            pages: Page range (e.g., "259-269")
            publication_date: Print publication date
            online_date: Online publication date
            doi: Digital Object Identifier
            issn: International Standard Serial Number
            publisher: Publisher name
        
        Returns:
            Metadata dictionary with publication info
        """
        self.metadata["publication_info"] = {
            "journal": journal_name,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "publication_date": publication_date,
            "online_date": online_date,
            "doi": doi,
            "issn": issn,
            "publisher": publisher,
            "citation": self._generate_citation(journal_name, volume, pages, publication_date),
        }
        return self.metadata
    
    def extract_trial_info(
        self,
        trial_id: str = None,
        trial_name: str = None,
        nct_number: str = None,
        study_type: str = None,  # RCT, observational, etc.
        phase: int = None,
        disease_area: str = None,
        biomarker: str = None,
        intervention: str = None,
        enrollment_start: str = None,
        enrollment_end: str = None,
        follow_up_end: str = None,
        status: str = None,  # OPEN, CLOSED, etc.
    ) -> Dict:
        """
        Extract clinical trial specific information.
        
        Args:
            trial_id: Internal trial identifier
            trial_name: Full trial name
            nct_number: ClinicalTrials.gov NCT number
            study_type: Type of study (RCT, cohort, etc.)
            phase: Trial phase (1, 2, 3, 4)
            disease_area: Disease/condition studied
            biomarker: Biomarker or genetic target
            intervention: Main intervention
            enrollment_start: Trial enrollment start date
            enrollment_end: Trial enrollment end date
            follow_up_end: Follow-up data cutoff date
            status: Trial status (CLOSED, ONGOING, etc.)
        
        Returns:
            Metadata dictionary with trial info
        """
        self.metadata["trial_info"] = {
            "trial_id": trial_id,
            "trial_name": trial_name,
            "nct_number": nct_number,
            "registry_url": f"https://clinicaltrials.gov/ct2/show/{nct_number}" if nct_number else None,
            "study_type": study_type,
            "phase": phase,
            "disease_area": disease_area,
            "biomarker": biomarker,
            "intervention": intervention,
            "enrollment": {
                "start_date": enrollment_start,
                "end_date": enrollment_end,
            },
            "follow_up_end_date": follow_up_end,
            "status": status,
        }
        return self.metadata
    
    def add_author(
        self,
        name: str,
        institution: str = None,
        email: str = None,
        role: str = None,  # PI, Co-I, etc.
        orcid: str = None,
    ) -> Dict:
        """
        Add author information.
        
        Args:
            name: Author full name
            institution: Institution affiliation
            email: Email address
            role: Role (e.g., "Principal Investigator")
            orcid: ORCID identifier
        
        Returns:
            Updated metadata
        """
        author = {
            "name": name,
            "institution": institution,
            "email": email,
            "role": role,
            "orcid": orcid,
            "verification_status": "UNVERIFIED",  # VERIFIED, CORRUPTED, FLAGGED
            "confidence_score": 0.5,  # Will be updated after validation
        }
        self.metadata["authors"].append(author)
        return self.metadata
    
    def add_institution(
        self,
        name: str,
        city: str = None,
        state: str = None,
        country: str = None,
        department: str = None,
    ) -> Dict:
        """
        Add institution information.
        
        Args:
            name: Institution name
            city: City
            state: State/Province
            country: Country
            department: Department/Division
        
        Returns:
            Updated metadata
        """
        institution = {
            "name": name,
            "city": city,
            "state": state,
            "country": country,
            "department": department,
            "verification_status": "UNVERIFIED",
            "confidence_score": 0.5,
        }
        self.metadata["institutions"].append(institution)
        return self.metadata
    
    def extract_funding_info(
        self,
        funding_sources: List[str] = None,
        role_in_design: str = None,
        role_in_analysis: str = None,
    ) -> Dict:
        """
        Extract funding information.
        
        Args:
            funding_sources: List of funding organization names
            role_in_design: Funder's role in study design
            role_in_analysis: Funder's role in data analysis
        
        Returns:
            Updated metadata
        """
        self.metadata["funding"] = {
            "sources": funding_sources or [],
            "role_in_design": role_in_design,
            "role_in_analysis": role_in_analysis,
            "financial_disclosure_url": None,  # Can be populated if available
        }
        return self.metadata
    
    def add_competing_interest(
        self,
        author: str,
        interest_type: str,  # consulting_fees, grants, stock, speaking, etc.
        organizations: List[str] = None,
        amount: str = None,
        year: int = None,
    ) -> Dict:
        """
        Add competing interest declaration.
        
        Args:
            author: Author name
            interest_type: Type of conflict (consulting, grants, etc.)
            organizations: Organizations involved
            amount: Financial amount (if available)
            year: Year of conflict
        
        Returns:
            Updated metadata
        """
        coi = {
            "author": author,
            "type": interest_type,
            "organizations": organizations or [],
            "amount": amount,
            "year": year,
            "verification_status": "UNVERIFIED",
        }
        self.metadata["competing_interests"].append(coi)
        return self.metadata
    
    def _generate_citation(
        self, journal: str, volume: str, pages: str, pub_date: str
    ) -> str:
        """Generate formatted citation."""
        citation_parts = []
        
        if journal:
            citation_parts.append(f"{journal}")
        
        if pub_date:
            # Extract year from date
            year = pub_date[:4] if pub_date else ""
            if year:
                citation_parts.append(f"{year}")
        
        if volume:
            citation_parts.append(f"Vol {volume}")
        
        if pages:
            citation_parts.append(f"pp {pages}")
        
        return "; ".join(citation_parts) if citation_parts else "Citation incomplete"
    
    def save_document_index(self, output_path: str) -> bool:
        """
        Save document metadata index to JSON file.
        
        Args:
            output_path: Path to save JSON file
        
        Returns:
            True if successful
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            print(f"✅ Document metadata saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving document metadata: {e}")
            return False
    
    def get_metadata(self) -> Dict:
        """Return complete metadata dictionary."""
        return self.metadata
    
    def get_summary(self) -> str:
        """Return human-readable summary of document metadata."""
        summary = f"""
╔════════════════════════════════════════════════════════════════╗
║           DOCUMENT METADATA INDEX SUMMARY                      ║
╚════════════════════════════════════════════════════════════════╝

DOCUMENT ID: {self.metadata.get("document_id", "N/A")}
Extracted: {self.metadata.get("extraction_timestamp")}

TITLE: {self.metadata.get("document_info", {}).get("title", "N/A")}

PUBLICATION:
  Journal: {self.metadata.get("publication_info", {}).get("journal", "N/A")}
  Volume: {self.metadata.get("publication_info", {}).get("volume", "N/A")}
  Issue: {self.metadata.get("publication_info", {}).get("issue", "N/A")}
  Pages: {self.metadata.get("publication_info", {}).get("pages", "N/A")}
  Published: {self.metadata.get("publication_info", {}).get("publication_date", "N/A")}
  Online: {self.metadata.get("publication_info", {}).get("online_date", "N/A")}
  DOI: {self.metadata.get("publication_info", {}).get("doi", "N/A")}

TRIAL:
  NCT Number: {self.metadata.get("trial_info", {}).get("nct_number", "N/A")}
  Phase: {self.metadata.get("trial_info", {}).get("phase", "N/A")}
  Disease Area: {self.metadata.get("trial_info", {}).get("disease_area", "N/A")}
  Biomarker: {self.metadata.get("trial_info", {}).get("biomarker", "N/A")}
  Status: {self.metadata.get("trial_info", {}).get("status", "N/A")}
  Enrollment: {self.metadata.get("trial_info", {}).get("enrollment", {}).get("start_date")} to {self.metadata.get("trial_info", {}).get("enrollment", {}).get("end_date")}

AUTHORS: {len(self.metadata.get("authors", []))} total
  {chr(10).join([f"  - {a.get('name')} ({a.get('institution', 'N/A')})" for a in self.metadata.get("authors", [])[:5]])}
  {"..." if len(self.metadata.get("authors", [])) > 5 else ""}

INSTITUTIONS: {len(self.metadata.get("institutions", []))} total

FUNDING SOURCES: {", ".join(self.metadata.get("funding", {}).get("sources", ["N/A"]))}

COMPETING INTERESTS: {len(self.metadata.get("competing_interests", []))} declarations

CITATION: {self.metadata.get("publication_info", {}).get("citation", "N/A")}

════════════════════════════════════════════════════════════════
        """
        return summary
