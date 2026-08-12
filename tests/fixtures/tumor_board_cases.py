"""Canonical patient cases used by the tumor-board test suite."""


# The canonical complex-multiaxis case used as a smoke-test fixture
# throughout the repository (see CLAUDE.md §9).
CANONICAL_ORAL_TONGUE_SCC_CASE = (
    "80 y.o. male non-smoker with a PMH HTN, Hep C, BPH, CKD, latent "
    "syphilis, transverse colon adenocarcinoma complicated by LBO s/p "
    "(6/16/21) diagnostic lap, ex lap with extended right hemicolectomy "
    "6/2021 and ileostomy reversal 10/6/2021, and initial Stage II "
    "(pT2pN0M0R0, DOI 5.1 mm, PNI-, LVSI-) squamous cell carcinoma of "
    "the left oral tongue, status post left partial glossectomy, left "
    "neck dissection levels I-III, and radial forearm free flap "
    "reconstruction, and left STSG performed at Bellevue Hospital on "
    "12/2/2024 with Dr. Moses. In August 2025, he developed a recurrent "
    "lesion in the left level I neck associated with a multiloculated "
    "left sub-lingual collection, which was biopsy-proven recurrent SCC "
    "with a CPS score of 100, started on pembrolizumab (declined "
    "combination with chemotherapy) and is no longer a surgical "
    "candidate following significant locoregional progression on ICI "
    "with radiographic concern for metastatic disease to the right "
    "ventricle and progressing on systemic therapy."
)


# Simple queries used to verify relevance filters
SIMPLE_DOSE_QUERY = "What is the standard dose of cisplatin for head and neck SCC?"
EMPTY_CASE = "  "
