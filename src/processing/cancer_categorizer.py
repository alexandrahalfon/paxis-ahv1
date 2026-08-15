"""
Auto-categorize medical documents by cancer type using LLM analysis.
"""

import os
from typing import Optional, Dict, Any
from openai import OpenAI

# 16 Medical Literature Categories
CANCER_TYPES = [
    "sarcoma",              # Sarcoma
    "radiotherapy_oncology", # General radiation oncology
    "radiopharm",           # Radiopharmaceuticals
    "prostate",             # Prostate cancer
    "peds",                 # Pediatric oncology
    "palliation",           # Palliative care
    "lymphoma",             # Lymphoma
    "lung",                 # Lung cancer
    "head_neck",            # Head and neck cancer
    "gyn",                  # Gynecological cancers
    "GU",                   # Genitourinary cancers
    "GI",                   # Gastrointestinal cancers
    "cutaneous",            # Skin/cutaneous cancers
    "CNS",                  # Central nervous system/brain
    "breast",               # Breast cancer
    "benign"                # Benign tumors
]

# Category display names for better readability
CATEGORY_DISPLAY_NAMES = {
    "sarcoma": "Sarcoma",
    "radiotherapy_oncology": "Radiotherapy & Oncology",
    "radiopharm": "Radiopharmaceuticals",
    "prostate": "Prostate Cancer",
    "peds": "Pediatric Oncology",
    "palliation": "Palliative Care",
    "lymphoma": "Lymphoma",
    "lung": "Lung Cancer",
    "head_neck": "Head & Neck Cancer",
    "gyn": "Gynecological Cancers",
    "GU": "Genitourinary Cancers",
    "GI": "Gastrointestinal Cancers",
    "cutaneous": "Cutaneous Cancers",
    "CNS": "CNS/Brain Tumors",
    "breast": "Breast Cancer",
    "benign": "Benign Tumors"
}

# Fallback for unclear documents
DEFAULT_CATEGORY = "radiotherapy_oncology"


class CancerCategorizer:
    """Automatically categorize documents by cancer type."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    
    def categorize_document(self, 
                           title: str = "",
                           abstract: str = "",
                           content_preview: str = "",
                           metadata: Dict[str, Any] = None) -> str:
        """
        Categorize document into one of 16 cancer types.
        
        Args:
            title: Document title
            abstract: Document abstract
            content_preview: First few paragraphs
            metadata: Document metadata
            
        Returns:
            Cancer type category (e.g., "lung_cancer")
        """
        # Combine available text
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        if abstract:
            text_parts.append(f"Abstract: {abstract}")
        if content_preview:
            text_parts.append(f"Content: {content_preview[:1000]}")
        
        if not text_parts:
            return DEFAULT_CATEGORY
        
        document_text = "\n\n".join(text_parts)
        
        # Create classification prompt with category descriptions
        cancer_list = "\n".join([f"- {ct}: {CATEGORY_DISPLAY_NAMES.get(ct, ct)}" for ct in CANCER_TYPES])
        
        prompt = f"""Analyze this medical/oncology document and classify it into ONE primary category:

{cancer_list}

Document:
{document_text}

Classification Guidelines:
- "sarcoma": Bone and soft tissue sarcomas
- "radiotherapy_oncology": General radiation therapy techniques, dosimetry, treatment planning
- "radiopharm": Radiopharmaceuticals, nuclear medicine, targeted radionuclide therapy
- "prostate": Prostate cancer (adenocarcinoma, treatment)
- "peds": Pediatric cancers (children, adolescents)
- "palliation": Palliative care, symptom management, end-of-life care
- "lymphoma": Hodgkin and non-Hodgkin lymphoma
- "lung": Lung cancer (NSCLC, SCLC, mesothelioma)
- "head_neck": Head and neck cancers (oral, throat, larynx, nasopharynx)
- "gyn": Gynecological cancers (cervical, ovarian, endometrial, vaginal, vulvar)
- "GU": Genitourinary cancers (bladder, kidney, testicular - NOT prostate)
- "GI": Gastrointestinal cancers (esophageal, gastric, colorectal, pancreatic, liver)
- "cutaneous": Skin cancers (melanoma, basal cell, squamous cell)
- "CNS": Brain and central nervous system tumors
- "breast": Breast cancer
- "benign": Benign tumors and non-malignant conditions

Rules:
1. Choose the MOST SPECIFIC category that fits
2. If it's general radiation oncology methodology → "radiotherapy_oncology"
3. If it's about radiopharmaceuticals/nuclear medicine → "radiopharm"
4. If it covers multiple sites but is pediatric → "peds"
5. If palliative focus regardless of cancer type → "palliation"
6. Return ONLY the category code (e.g., "lung" not "lung cancer")

Category:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap for classification
                messages=[
                    {"role": "system", "content": "You are a medical document classifier. Return only the category name, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=20
            )
            
            category = response.choices[0].message.content.strip().lower()
            
            # Validate category
            if category in CANCER_TYPES:
                return category
            else:
                print(f"⚠️  LLM returned invalid category: {category}, using default")
                return DEFAULT_CATEGORY
                
        except Exception as e:
            print(f"⚠️  Categorization failed: {e}, using default")
            return DEFAULT_CATEGORY
    
    def categorize_from_structured_content(self, structured_content: Dict[str, Any]) -> str:
        """
        Categorize from structured content JSON.
        
        Args:
            structured_content: The structured_content.json data
            
        Returns:
            Cancer type category
        """
        # Extract title
        title = structured_content.get("document_metadata", {}).get("title", "")
        
        # Extract abstract (usually in first section)
        abstract = ""
        sections = structured_content.get("sections", [])
        if sections:
            # Look for abstract section
            for section in sections[:3]:  # Check first 3 sections
                section_title = section.get("title", "").lower()
                if "abstract" in section_title or "summary" in section_title:
                    abstract = section.get("content", "")
                    break
            
            # If no abstract found, use first section content
            if not abstract and sections:
                abstract = sections[0].get("content", "")
        
        # Get preview from multiple sections
        content_preview = ""
        for section in sections[:5]:  # First 5 sections
            content_preview += section.get("content", "") + "\n\n"
        
        return self.categorize_document(
            title=title,
            abstract=abstract,
            content_preview=content_preview[:2000],  # Limit to 2000 chars
            metadata=structured_content.get("document_metadata", {})
        )


def main():
    """Test the categorizer."""
    categorizer = CancerCategorizer()
    
    # Test examples
    test_cases = [
        {
            "title": "Phase 3 Trial of Pembrolizumab for Non-Small Cell Lung Cancer",
            "abstract": "This study evaluates pembrolizumab in advanced NSCLC patients..."
        },
        {
            "title": "Trastuzumab in HER2-Positive Breast Cancer",
            "abstract": "We studied the efficacy of trastuzumab in metastatic breast cancer..."
        },
        {
            "title": "177Lu-DOTATATE for Neuroendocrine Tumors",
            "abstract": "Phase 3 trial of lutetium-177-labeled radiopharmaceutical therapy..."
        },
        {
            "title": "IMRT Techniques for Head and Neck Cancer",
            "abstract": "We compared IMRT planning techniques for oropharyngeal carcinoma..."
        },
        {
            "title": "Palliative Care in Advanced Cancer",
            "abstract": "Early palliative care improves quality of life across cancer types..."
        }
    ]
    
    print("Testing Cancer Categorizer:\n")
    for i, test in enumerate(test_cases, 1):
        category = categorizer.categorize_document(
            title=test["title"],
            abstract=test["abstract"]
        )
        print(f"{i}. {test['title'][:60]}...")
        print(f"   → {category}\n")


if __name__ == "__main__":
    main()
