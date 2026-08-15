"""
Lay language to clinical vocabulary.

The retrieval pipeline was built for physician phrasing. A clinician types
"recurrent HNSCC, CPS 100, progressing on pembro". A patient types "I have
throat cancer and the immunotherapy stopped working I think".

This module bridges that gap by expanding a patient's words with the terms
the literature actually uses, so the same retrieval engine works for both.
It runs *before* retrieval and only ever adds terms; the patient's original
wording is preserved for the answer prompt.

Same shape as ``clinical_inference.py``'s INFERENCE_MAP, deliberately, so
the pattern is familiar and the two can be maintained together.

This is a starting set covering common oncology situations. It should grow
from real patient questions once the portal is live: log the messages that
retrieve nothing useful and mine them for missing phrasings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


# trigger pattern -> clinical terms to add
VOCAB_MAP: Dict[str, List[str]] = {

    # ── Drugs and regimens by nickname / description ────────────────────
    r"\bred (?:devil|medicine|drug|stuff)\b": [
        "doxorubicin", "anthracycline",
    ],
    r"\bchemo (?:pills|tablets)\b|\boral chemo\w*": [
        "oral chemotherapy", "capecitabine", "temozolomide",
    ],
    r"\bimmunotherapy\b|\bimmuno\b": [
        "immune checkpoint inhibitor", "anti-PD1", "anti-PD-L1", "checkpoint blockade",
    ],
    r"\bpembro\b": ["pembrolizumab", "anti-PD1", "checkpoint inhibitor"],
    r"\bnivo\b": ["nivolumab", "anti-PD1", "checkpoint inhibitor"],
    r"\bthe infusion\b|\bmy infusion\b|\bIV (?:treatment|drug)\b": [
        "intravenous therapy", "infusion",
    ],
    r"\btargeted (?:therapy|treatment|pill)\b": [
        "targeted therapy", "tyrosine kinase inhibitor", "molecularly targeted agent",
    ],
    r"\bhormone (?:therapy|pills|blockers?)\b": [
        "endocrine therapy", "hormonal therapy", "aromatase inhibitor", "tamoxifen",
    ],
    r"\bradiation\b|\bradiotherapy\b|\bproton\b|\bthe rays\b": [
        "radiotherapy", "external beam radiation", "radiation therapy",
    ],

    # ── Scans and tests ─────────────────────────────────────────────────
    r"\b(?:the )?big scan\b|\blighting? up scan\b|\bglow(?:ing)? scan\b|\bPET\b": [
        "PET/CT", "FDG-PET", "positron emission tomography",
    ],
    r"\bCAT scan\b|\bCT\b": ["computed tomography", "CT scan"],
    r"\bMRI\b|\bthe magnet\b": ["magnetic resonance imaging"],
    r"\bbiopsy\b|\bthey took a (?:sample|piece)\b": [
        "biopsy", "histopathology", "tissue sampling",
    ],
    r"\bblood counts?\b|\bmy counts\b": [
        "complete blood count", "neutrophil count", "haematologic parameters",
    ],
    r"\bgene(?:tic)? test\w*|\bmolecular test\w*|\bthey tested (?:the|my) tumou?r\b": [
        "next-generation sequencing", "molecular profiling", "genomic testing",
    ],

    # ── Disease status in patient words ─────────────────────────────────
    r"\bit (?:came|is) back\b|\bit'?s back\b|\bcame back\b": [
        "recurrence", "recurrent disease", "relapse",
    ],
    r"\bit (?:has )?spread\b|\bthey said it spread\b|\bspread to\b": [
        "metastatic", "metastasis", "distant spread", "M1 disease",
    ],
    r"\bstopped working\b|\bnot working (?:any ?more)?\b|\bquit working\b": [
        "treatment failure", "progression", "refractory", "acquired resistance",
    ],
    r"\bgot (?:worse|bigger)\b|\bgrowing\b|\bgrew\b": [
        "disease progression", "progressive disease",
    ],
    r"\bthey can'?t operate\b|\bnot able to operate\b|\bcan'?t do surgery\b|"
    r"\bnot a candidate for surgery\b|\btoo risky to operate\b": [
        "unresectable", "inoperable", "non-surgical candidate",
    ],
    r"\bno (?:evidence of )?(?:cancer|disease)\b|\ball clear\b|\bin remission\b|\bclean scan\b": [
        "complete response", "no evidence of disease", "remission",
    ],
    r"\bshrunk\b|\bshrinking\b|\bgot smaller\b|\bresponding\b": [
        "partial response", "tumour regression", "treatment response",
    ],
    r"\bearly stage\b|\bcaught (?:it )?early\b": ["early-stage", "localized disease"],
    r"\badvanced\b|\blate stage\b": ["advanced disease", "locally advanced", "metastatic"],

    # ── Side effects in patient words ───────────────────────────────────
    r"\bmouth sores\b|\bsores in my mouth\b": ["mucositis", "stomatitis", "oral mucositis"],
    r"\bpins and needles\b|\btingling\b|\bnumb (?:hands|feet|fingers|toes)\b": [
        "peripheral neuropathy", "chemotherapy-induced peripheral neuropathy",
    ],
    r"\bsick to my stomach\b|\bthrowing up\b|\bqueasy\b": [
        "nausea", "emesis", "chemotherapy-induced nausea and vomiting",
    ],
    r"\bso tired\b|\bexhaust\w+|\bwiped out\b|\bno energy\b": [
        "fatigue", "cancer-related fatigue",
    ],
    r"\bhair (?:loss|falling out)\b|\blosing my hair\b": ["alopecia"],
    r"\bskin (?:rash|burn)\w*|\bred(?:ness)? (?:on|where) (?:the )?radiation\b": [
        "radiation dermatitis", "rash", "cutaneous toxicity",
    ],
    r"\bcan'?t taste\b|\bfood tastes\b|\bmetallic taste\b": ["dysgeusia", "taste alteration"],
    r"\btrouble swallowing\b|\bhard to swallow\b|\bpainful to swallow\b": [
        "dysphagia", "odynophagia", "esophagitis",
    ],
    r"\bdry mouth\b": ["xerostomia"],
    r"\bswollen\b|\bswelling\b|\bpuffy\b": ["oedema", "lymphoedema"],
    r"\bbrain fog\b|\bcan'?t concentrate\b|\bforgetful\b": [
        "cognitive impairment", "chemotherapy-related cognitive dysfunction",
    ],

    # ── Biomarkers as patients see them on reports ──────────────────────
    r"\bCPS\b": ["combined positive score", "PD-L1 expression"],
    r"\bPD ?-? ?L1\b": ["PD-L1 expression", "programmed death-ligand 1"],
    r"\bHER ?2\b": ["HER2", "human epidermal growth factor receptor 2"],
    r"\btriple negative\b": ["triple-negative breast cancer", "TNBC"],
    r"\bER positive\b|\bestrogen positive\b": ["oestrogen receptor positive", "hormone receptor positive"],
    r"\bp16\b|\bHPV\b": ["HPV-positive", "p16-positive", "human papillomavirus"],
    r"\bmarkers? (?:were|was|came back)\b|\btumou?r markers?\b": ["tumour marker", "biomarker"],

    # ── Care setting / intent ───────────────────────────────────────────
    r"\bclinical trial\b|\bexperimental treatment\b|\bstudy drug\b": [
        "clinical trial", "investigational therapy", "trial eligibility",
    ],
    r"\bsecond opinion\b": ["second opinion", "multidisciplinary review"],
    r"\bpalliative\b|\bcomfort care\b|\bhospice\b": [
        "palliative care", "supportive care", "end-of-life care",
    ],
    r"\bbefore surgery\b|\bshrink (?:it|the tumou?r) first\b": ["neoadjuvant"],
    r"\bafter surgery\b|\bmop up\b|\bto be safe\b": ["adjuvant"],
}


@dataclass
class VocabExpansion:
    original: str
    expanded_query: str
    added_terms: List[str] = field(default_factory=list)
    matched_phrases: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added_terms)


def expand_patient_language(text: str, max_terms: int = 24) -> VocabExpansion:
    """Add clinical vocabulary to a patient's message for retrieval.

    Never replaces the patient's words, only appends terms. Returns the
    original untouched alongside the expanded string so the answer prompt
    can keep using the patient's own phrasing.
    """
    if not text or not text.strip():
        return VocabExpansion(original=text or "", expanded_query=text or "")

    added: List[str] = []
    matched: List[str] = []
    lower = text.lower()

    for pattern, terms in VOCAB_MAP.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        matched.append(m.group(0).strip()[:40])
        for t in terms:
            # Skip terms already present so we don't bloat the embedding
            if t.lower() not in lower and t not in added:
                added.append(t)

    added = added[:max_terms]
    expanded = f"{text} {' '.join(added)}".strip() if added else text
    return VocabExpansion(
        original=text,
        expanded_query=expanded,
        added_terms=added,
        matched_phrases=matched,
    )


# ── Gap detection ──────────────────────────────────────────────────────────
# Which clinical axes the patient has *not* mentioned, used to pick the
# single most useful follow-up question rather than interrogating them.

_HAS_CANCER_TYPE = re.compile(
    r"\b(breast|lung|prostate|colon|colorectal|rectal|bladder|kidney|renal|"
    r"liver|pancrea\w+|stomach|gastric|oesophag\w+|esophag\w+|ovar\w+|cervi\w+|"
    r"uterine|endometrial|head and neck|throat|tongue|larynx|thyroid|melanoma|"
    r"skin|lymphoma|leukemia|leukaemia|myeloma|sarcoma|brain|glioma|testic\w+)\b",
    re.IGNORECASE,
)
_HAS_TREATMENT = re.compile(
    r"\b(chemo\w*|radiation|radiotherapy|immunotherapy|surgery|operation|"
    r"pembrolizumab|pembro|nivolumab|nivo|targeted|hormone|infusion|pill|"
    r"tablet|trial|transplant)\b",
    re.IGNORECASE,
)
_HAS_STAGE = re.compile(
    r"\b(stage\s*(?:1|2|3|4|i{1,3}v?|iv)|early|advanced|metastatic|spread|"
    r"localis?ed|localiz?ed)\b",
    re.IGNORECASE,
)


def detect_gaps(text: str, known: Dict[str, object] | None = None) -> List[str]:
    """Return missing axes, most useful first.

    ``known`` is anything already established (from a linked record or
    earlier in the conversation) so we never ask twice.
    """
    known = known or {}
    combined = " ".join(
        [text or ""] + [str(v) for v in known.values() if v]
    )
    gaps: List[str] = []
    if not known.get("cancer_type") and not _HAS_CANCER_TYPE.search(combined):
        gaps.append("cancer_type")
    if not known.get("treatment") and not _HAS_TREATMENT.search(combined):
        gaps.append("treatment")
    if not known.get("stage") and not _HAS_STAGE.search(combined):
        gaps.append("stage")
    return gaps


# Phrasing for each gap. Warm, one question, always with an easy out, so
# a patient who does not know is never made to feel they failed.
GAP_QUESTIONS: Dict[str, str] = {
    "cancer_type": (
        "So I can point you at the right information, do you know what type "
        "of cancer you're being treated for? It's usually near the top of any "
        "report you've been given. No problem at all if you're not sure."
    ),
    "treatment": (
        "Do you know the name of the treatment you're on? Even part of the "
        "name, or just \"chemo\" or \"immunotherapy\", helps a lot. It's fine "
        "if you don't have it to hand."
    ),
    "stage": (
        "Has your team mentioned a stage, or whether it has spread anywhere? "
        "Don't worry if that hasn't come up yet."
    ),
}
