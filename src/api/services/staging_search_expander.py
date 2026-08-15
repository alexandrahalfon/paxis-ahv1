"""
Staging Search Expander Service

Provides bidirectional TNM ↔ Stage Group expansion for search queries.
Enables finding studies regardless of whether they report TNM (T2N1M0) 
or stage group (Stage IIB) notation.

Key Features:
- Loads comprehensive AJCC 8th Edition staging tables from JSON
- Extracts staging from query text (TNM, stage groups, c/p/yp prefixes)
- Bidirectional lookup: TNM → Stage Group and Stage Group → TNM
- Generates all notation variants for search expansion
- Cancer type inference from query text

Usage:
    from staging_search_expander import StagingSearchExpander, expand_query_with_staging
    
    # Quick usage
    terms = expand_query_with_staging("T2N1M0 breast cancer")
    # Returns: StagingSearchTerms with tnm_variants, stage_group_variants, etc.
    
    # Full control
    expander = StagingSearchExpander()
    terms = expander.expand_staging_for_search("Stage IIB breast cancer", cancer_type="breast")

Author: AI Assistant
Version: 2.0.0
"""

import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractedStaging:
    """Staging information extracted from text."""
    t_stage: Optional[str] = None
    n_stage: Optional[str] = None
    m_stage: Optional[str] = None
    stage_group: Optional[str] = None
    staging_type: str = "unknown"  # clinical, pathologic, yp (post-neoadjuvant), unknown
    confidence: float = 0.0
    source_text: str = ""
    
    def has_tnm(self) -> bool:
        """Check if any TNM component is present."""
        return any([self.t_stage, self.n_stage, self.m_stage])
    
    def is_complete_tnm(self) -> bool:
        """Check if all TNM components are present."""
        return all([self.t_stage, self.n_stage, self.m_stage])
    
    def to_tnm_tuple(self) -> Tuple[str, str, str]:
        """Convert to tuple for lookup."""
        return (
            self.t_stage.lower() if self.t_stage else "",
            self.n_stage.lower() if self.n_stage else "",
            self.m_stage.lower() if self.m_stage else ""
        )


@dataclass
class StagingSearchTerms:
    """All search term variants generated from staging information."""
    # Original extracted info
    original_query: str = ""
    extracted_staging: Optional[ExtractedStaging] = None
    inferred_cancer_type: Optional[str] = None
    
    # TNM variants for search
    tnm_variants: List[str] = field(default_factory=list)
    
    # Stage group variants for search
    stage_group_variants: List[str] = field(default_factory=list)
    
    # Inferred stage groups from TNM (with confidence)
    inferred_stage_groups: List[Tuple[str, float]] = field(default_factory=list)
    
    # Inferred TNM combinations from stage group
    inferred_tnm: List[Tuple[str, str, str]] = field(default_factory=list)
    
    # Combined search terms (deduplicated)
    all_search_terms: List[str] = field(default_factory=list)
    
    # Metadata
    expansion_notes: List[str] = field(default_factory=list)


# =============================================================================
# STAGING TABLES LOADER
# =============================================================================

def _find_ajcc_tables_path() -> Path:
    """
    Find the AJCC staging tables JSON file.
    
    Searches in order:
    1. data/ajcc_staging_tables.json (relative to project root)
    2. Same directory as this module (for testing)
    
    Returns:
        Path to the JSON file
    """
    # Try project data/ directory first
    # From src/api/services/ -> project root is 4 levels up
    module_dir = Path(__file__).parent
    project_root = module_dir.parent.parent.parent
    data_path = project_root / "data" / "ajcc_staging_tables.json"
    
    if data_path.exists():
        return data_path
    
    # Fallback: same directory as module (for testing)
    local_path = module_dir / "ajcc_staging_tables.json"
    if local_path.exists():
        return local_path
    
    # Return the expected production path even if it doesn't exist yet
    # (will raise FileNotFoundError when loading)
    return data_path


class AJCCStagingTables:
    """Loads and provides access to AJCC staging tables."""
    
    def __init__(self, json_path: Optional[Path] = None):
        """
        Initialize with path to ajcc_staging_tables.json.
        
        Args:
            json_path: Path to JSON file. If None, auto-discovers in data/ directory.
        """
        if json_path is None:
            json_path = _find_ajcc_tables_path()
        
        self.json_path = json_path
        self._raw_data: Dict[str, Any] = {}
        self._forward_tables: Dict[str, Dict[Tuple[str, str, str], str]] = {}  # TNM → Stage
        self._reverse_tables: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}  # Stage → TNM list
        self._aliases: Dict[str, str] = {}  # alias → canonical cancer type
        self._loaded = False
    
    def load(self) -> bool:
        """Load staging tables from JSON file."""
        if self._loaded:
            return True
            
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self._raw_data = json.load(f)
            
            self._build_lookup_tables()
            self._loaded = True
            logger.info(f"Loaded AJCC staging tables from {self.json_path}: {len(self._forward_tables)} cancer types")
            return True
            
        except FileNotFoundError:
            logger.error(f"AJCC staging tables not found at {self.json_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing AJCC staging tables: {e}")
            return False
    
    def _build_lookup_tables(self):
        """Build forward and reverse lookup tables from raw data."""
        for cancer_type, data in self._raw_data.items():
            if cancer_type.startswith('_'):
                continue
            
            if not isinstance(data, dict):
                continue
            
            # Build alias mapping
            aliases = data.get('aliases', [])
            for alias in aliases:
                self._aliases[alias.lower()] = cancer_type
            self._aliases[cancer_type.lower()] = cancer_type
            
            # Find staging table(s)
            staging_tables = {}
            for key, value in data.items():
                if key.startswith('staging_table') and isinstance(value, dict):
                    suffix = key.replace('staging_table', '').strip('_') or 'default'
                    staging_tables[suffix] = value
            
            if not staging_tables:
                continue
            
            # Build forward table (TNM → Stage)
            forward = {}
            reverse = {}
            
            for table_name, table in staging_tables.items():
                for stage_group, tnm_list in table.items():
                    if not isinstance(tnm_list, list):
                        continue
                    
                    for tnm in tnm_list:
                        if len(tnm) >= 3:
                            t, n, m = str(tnm[0]).lower(), str(tnm[1]).lower(), str(tnm[2]).lower()
                            key = (t, n, m)
                            
                            # Forward: TNM → Stage (first match wins for duplicates)
                            if key not in forward:
                                forward[key] = stage_group
                            
                            # Reverse: Stage → TNM list
                            if stage_group not in reverse:
                                reverse[stage_group] = []
                            if key not in reverse[stage_group]:
                                reverse[stage_group].append(key)
            
            if forward:
                self._forward_tables[cancer_type] = forward
            if reverse:
                self._reverse_tables[cancer_type] = reverse
    
    def get_cancer_types(self) -> List[str]:
        """Get list of all supported cancer types."""
        return list(self._forward_tables.keys())
    
    def resolve_cancer_type(self, text: str) -> Optional[str]:
        """Resolve cancer type from text using aliases."""
        text_lower = text.lower()
        
        # Direct alias match
        if text_lower in self._aliases:
            return self._aliases[text_lower]
        
        # Partial match in text
        for alias, canonical in self._aliases.items():
            if alias in text_lower:
                return canonical
        
        return None
    
    def lookup_stage_group(
        self, 
        cancer_type: str, 
        t: str, 
        n: str, 
        m: str
    ) -> Tuple[Optional[str], float, List[str]]:
        """
        Look up stage group from TNM.
        
        Returns:
            Tuple of (stage_group, confidence, alternative_stages)
        """
        if not self._loaded:
            self.load()
        
        cancer_type = self._aliases.get(cancer_type.lower(), cancer_type)
        
        if cancer_type not in self._forward_tables:
            return None, 0.0, []
        
        table = self._forward_tables[cancer_type]
        t_lower, n_lower, m_lower = t.lower(), n.lower(), m.lower()
        
        # Try exact match first
        key = (t_lower, n_lower, m_lower)
        if key in table:
            return table[key], 1.0, []
        
        # Try with wildcards
        alternatives = []
        
        # Try "any" wildcards in the table
        for (tt, tn, tm), stage in table.items():
            if tt == "any" or tt == t_lower:
                if tn == "any" or tn == n_lower:
                    if tm == "any" or tm == m_lower:
                        if stage not in alternatives:
                            alternatives.append(stage)
        
        if alternatives:
            return alternatives[0], 0.8, alternatives[1:]
        
        # Try stripping substage suffixes (e.g., T1a → T1)
        t_base = re.sub(r'[a-d]$', '', t_lower)
        n_base = re.sub(r'[a-c]$|mi$', '', n_lower)
        m_base = re.sub(r'[a-c]$', '', m_lower)
        
        for (tt, tn, tm), stage in table.items():
            tt_base = re.sub(r'[a-d]$', '', tt)
            tn_base = re.sub(r'[a-c]$|mi$', '', tn)
            tm_base = re.sub(r'[a-c]$', '', tm)
            
            if (tt == "any" or tt_base == t_base):
                if (tn == "any" or tn_base == n_base):
                    if (tm == "any" or tm_base == m_base):
                        if stage not in alternatives:
                            alternatives.append(stage)
        
        if alternatives:
            return alternatives[0], 0.6, alternatives[1:]
        
        return None, 0.0, []
    
    def lookup_tnm_from_stage(
        self, 
        cancer_type: str, 
        stage_group: str
    ) -> List[Tuple[str, str, str]]:
        """
        Look up possible TNM combinations from stage group.
        
        Returns:
            List of (T, N, M) tuples
        """
        if not self._loaded:
            self.load()
        
        cancer_type = self._aliases.get(cancer_type.lower(), cancer_type)
        
        if cancer_type not in self._reverse_tables:
            return []
        
        reverse = self._reverse_tables[cancer_type]
        
        # Normalize stage group (Stage IIB → IIB, stage 2b → IIB)
        stage_normalized = self._normalize_stage_group(stage_group)
        
        # Try exact match
        if stage_normalized in reverse:
            return reverse[stage_normalized]
        
        # Try without substage (IIB → II)
        stage_base = re.sub(r'[ABC]$', '', stage_normalized)
        if stage_base in reverse:
            return reverse[stage_base]
        
        return []
    
    def _normalize_stage_group(self, stage: str) -> str:
        """Normalize stage group to canonical form (e.g., 'stage 2b' → 'IIB')."""
        stage = stage.upper().strip()
        
        # Remove "STAGE " prefix
        stage = re.sub(r'^STAGE\s*', '', stage, flags=re.IGNORECASE)
        
        # Convert Arabic to Roman numerals
        conversions = [
            (r'^0', '0'),
            (r'^1([ABC]?)$', r'I\1'),
            (r'^2([ABC]?)$', r'II\1'),
            (r'^3([ABC]?)$', r'III\1'),
            (r'^4([ABC]?)$', r'IV\1'),
        ]
        
        for pattern, replacement in conversions:
            stage = re.sub(pattern, replacement, stage)
        
        return stage
    
    def get_t_definition(self, cancer_type: str, t_stage: str) -> Optional[str]:
        """Get T stage definition for a cancer type."""
        if not self._loaded:
            self.load()
        
        cancer_type = self._aliases.get(cancer_type.lower(), cancer_type)
        
        if cancer_type not in self._raw_data:
            return None
        
        data = self._raw_data[cancer_type]
        
        # Try different definition keys
        for key in ['t_definitions', 't_definitions_glottis']:
            if key in data:
                defs = data[key]
                if t_stage.lower() in defs:
                    return defs[t_stage.lower()]
        
        return None


# =============================================================================
# STAGING SEARCH EXPANDER
# =============================================================================

class StagingSearchExpander:
    """
    Main service for expanding staging queries for search.
    
    Handles:
    - Extracting staging from query text
    - Inferring cancer type
    - Bidirectional TNM ↔ Stage Group lookup
    - Generating all notation variants for comprehensive search
    """
    
    # Regex patterns for staging extraction
    PATTERNS = {
        'full_tnm': re.compile(
            r'\b([cyp]{0,2})T(is|a|[0-4][a-d]?(?:mi)?|x)\s*'
            r'([cyp]?)N([0-3][a-c]?(?:mi)?|x)\s*'
            r'([cyp]?)M([01][a-c]?|x)\b',
            re.IGNORECASE
        ),
        't_stage': re.compile(r'\b([cyp]{0,2})T(is|a|[0-4][a-d]?(?:mi)?|x)\b', re.IGNORECASE),
        'n_stage': re.compile(r'\b([cyp]?)N([0-3][a-c]?(?:mi)?|x)\b', re.IGNORECASE),
        'm_stage': re.compile(r'\b([cyp]?)M([01][a-c]?|x)\b', re.IGNORECASE),
        'stage_group': re.compile(
            r'\bstage\s*([0IV]{1,3}|[1-4])\s*([ABC]?)\b',
            re.IGNORECASE
        ),
        'stage_group_short': re.compile(
            r'(?<!level )(?<!Level )\b(I{1,3}V?|IV|[0-4])\s*([ABC])\b',
            re.IGNORECASE
        ),
    }
    
    def __init__(self, tables: Optional[AJCCStagingTables] = None):
        """
        Initialize the expander.
        
        Args:
            tables: AJCCStagingTables instance. If None, creates default.
        """
        self.tables = tables or AJCCStagingTables()
        self.tables.load()
    
    def extract_staging_from_text(self, text: str) -> ExtractedStaging:
        """
        Extract staging information from text.
        
        Args:
            text: Query or document text
            
        Returns:
            ExtractedStaging with extracted components
        """
        staging = ExtractedStaging(source_text=text)
        
        # Try full TNM pattern first
        match = self.PATTERNS['full_tnm'].search(text)
        if match:
            prefix = match.group(1) or ""
            staging.t_stage = match.group(2)
            staging.n_stage = match.group(4)
            staging.m_stage = match.group(6)
            staging.staging_type = self._parse_prefix(prefix)
            staging.confidence = 0.95
            return staging
        
        # Try individual components
        t_match = self.PATTERNS['t_stage'].search(text)
        if t_match:
            prefix = t_match.group(1) or ""
            staging.t_stage = t_match.group(2)
            staging.staging_type = self._parse_prefix(prefix)
            staging.confidence += 0.3
        
        n_match = self.PATTERNS['n_stage'].search(text)
        if n_match:
            staging.n_stage = n_match.group(2)
            staging.confidence += 0.3
        
        m_match = self.PATTERNS['m_stage'].search(text)
        if m_match:
            staging.m_stage = m_match.group(2)
            staging.confidence += 0.3
        
        # Try stage group patterns
        stage_match = self.PATTERNS['stage_group'].search(text)
        if not stage_match:
            stage_match = self.PATTERNS['stage_group_short'].search(text)
        
        if stage_match:
            stage_num = stage_match.group(1).upper()
            stage_suffix = (stage_match.group(2) or "").upper()
            
            # Convert Arabic to Roman
            roman_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV', '0': '0'}
            stage_num = roman_map.get(stage_num, stage_num)
            
            staging.stage_group = f"{stage_num}{stage_suffix}"
            staging.confidence = max(staging.confidence, 0.7)
        
        return staging
    
    def _parse_prefix(self, prefix: str) -> str:
        """Parse staging prefix to determine type."""
        prefix = prefix.lower()
        if 'yp' in prefix:
            return 'yp'  # Post-neoadjuvant pathologic
        elif 'p' in prefix:
            return 'pathologic'
        elif 'c' in prefix:
            return 'clinical'
        return 'unknown'
    
    def infer_cancer_type(self, text: str) -> Optional[str]:
        """
        Infer cancer type from text.
        
        Args:
            text: Query text
            
        Returns:
            Canonical cancer type name or None
        """
        return self.tables.resolve_cancer_type(text)
    
    def generate_tnm_variants(
        self, 
        t: str, 
        n: str, 
        m: str,
        include_prefixes: bool = True
    ) -> List[str]:
        """
        Generate all TNM notation variants.
        
        Args:
            t: T stage (e.g., "2", "2a")
            n: N stage (e.g., "1", "1mi")
            m: M stage (e.g., "0", "1a")
            include_prefixes: Include c/p/yp variants
            
        Returns:
            List of TNM strings
        """
        variants = set()
        
        # Normalize
        t = t.lower().strip()
        n = n.lower().strip()
        m = m.lower().strip()
        
        # Base form
        base = f"T{t}N{n}M{m}"
        variants.add(base)
        variants.add(base.upper())
        
        # With spaces
        variants.add(f"T{t} N{n} M{m}")
        
        if include_prefixes:
            # Clinical
            variants.add(f"cT{t}N{n}M{m}")
            variants.add(f"cT{t}cN{n}cM{m}")
            variants.add(f"cT{t} cN{n} cM{m}")
            
            # Pathologic
            variants.add(f"pT{t}N{n}M{m}")
            variants.add(f"pT{t}pN{n}pM{m}")
            variants.add(f"pT{t} pN{n} pM{m}")
            
            # Post-neoadjuvant
            variants.add(f"ypT{t}N{n}M{m}")
            variants.add(f"ypT{t}ypN{n}ypM{m}")
        
        # Individual components (for partial matching)
        variants.add(f"T{t}")
        variants.add(f"N{n}")
        variants.add(f"M{m}")
        
        return list(variants)
    
    def generate_stage_group_variants(self, stage_group: str) -> List[str]:
        """
        Generate all stage group notation variants.
        
        Args:
            stage_group: Stage group (e.g., "IIB", "3A")
            
        Returns:
            List of stage group strings
        """
        variants = set()
        
        # Normalize to Roman numeral form
        stage = stage_group.upper().strip()
        stage = re.sub(r'^STAGE\s*', '', stage)
        
        # Convert Arabic to Roman if needed
        arabic_to_roman = {'0': '0', '1': 'I', '2': 'II', '3': 'III', '4': 'IV'}
        for arabic, roman in arabic_to_roman.items():
            if stage.startswith(arabic):
                stage = roman + stage[1:]
                break
        
        # Extract base and suffix
        match = re.match(r'^(0|I{1,3}V?|IV)([ABC]?\d?)$', stage)
        if match:
            base = match.group(1)
            suffix = match.group(2) or ""
        else:
            base = stage
            suffix = ""
        
        # Roman numeral forms
        variants.add(f"Stage {base}{suffix}")
        variants.add(f"stage {base}{suffix}")
        variants.add(f"STAGE {base}{suffix}")
        variants.add(f"{base}{suffix}")
        
        # With various separators
        if suffix:
            variants.add(f"Stage {base} {suffix}")
            variants.add(f"Stage {base}-{suffix}")
        
        # Arabic numeral forms
        roman_to_arabic = {'0': '0', 'I': '1', 'II': '2', 'III': '3', 'IV': '4'}
        if base in roman_to_arabic:
            arabic_base = roman_to_arabic[base]
            variants.add(f"Stage {arabic_base}{suffix}")
            variants.add(f"stage {arabic_base}{suffix}")
            variants.add(f"{arabic_base}{suffix}")
            if suffix:
                variants.add(f"Stage {arabic_base} {suffix}")
        
        # Lowercase suffix variants
        if suffix:
            variants.add(f"Stage {base}{suffix.lower()}")
            variants.add(f"stage {base}{suffix.lower()}")
        
        return list(variants)
    
    def expand_staging_for_search(
        self, 
        query: str, 
        cancer_type: Optional[str] = None
    ) -> StagingSearchTerms:
        """
        Expand staging in query for comprehensive search.
        
        This is the main method for query expansion. It:
        1. Extracts staging from query
        2. Infers cancer type if not provided
        3. Performs bidirectional lookup (TNM ↔ Stage Group)
        4. Generates all notation variants
        
        Args:
            query: Search query text
            cancer_type: Cancer type (optional, will be inferred if not provided)
            
        Returns:
            StagingSearchTerms with all expansion information
        """
        result = StagingSearchTerms(original_query=query)
        
        # Extract staging
        staging = self.extract_staging_from_text(query)
        result.extracted_staging = staging
        
        # Infer cancer type
        if cancer_type:
            result.inferred_cancer_type = self.tables.resolve_cancer_type(cancer_type)
        else:
            result.inferred_cancer_type = self.infer_cancer_type(query)
        
        if not result.inferred_cancer_type:
            result.expansion_notes.append("Could not determine cancer type - stage group inference limited")
        
        all_terms: Set[str] = set()
        
        # If we have TNM components
        if staging.has_tnm():
            # Generate TNM variants
            if staging.is_complete_tnm():
                tnm_variants = self.generate_tnm_variants(
                    staging.t_stage, staging.n_stage, staging.m_stage
                )
                result.tnm_variants = tnm_variants
                all_terms.update(tnm_variants)
                
                # Look up stage group
                if result.inferred_cancer_type:
                    stage, confidence, alternatives = self.tables.lookup_stage_group(
                        result.inferred_cancer_type,
                        staging.t_stage,
                        staging.n_stage,
                        staging.m_stage
                    )
                    
                    if stage:
                        result.inferred_stage_groups.append((stage, confidence))
                        for alt in alternatives:
                            result.inferred_stage_groups.append((alt, confidence * 0.8))
                        
                        # Generate stage group variants
                        for sg, _ in result.inferred_stage_groups:
                            sg_variants = self.generate_stage_group_variants(sg)
                            result.stage_group_variants.extend(sg_variants)
                            all_terms.update(sg_variants)
                        
                        result.expansion_notes.append(
                            f"Inferred stage group {stage} from TNM "
                            f"(confidence: {confidence:.0%})"
                        )
            else:
                # Partial TNM - still generate variants for what we have
                partial_variants = []
                if staging.t_stage:
                    partial_variants.append(f"T{staging.t_stage}")
                    partial_variants.append(f"cT{staging.t_stage}")
                    partial_variants.append(f"pT{staging.t_stage}")
                if staging.n_stage:
                    partial_variants.append(f"N{staging.n_stage}")
                    partial_variants.append(f"cN{staging.n_stage}")
                    partial_variants.append(f"pN{staging.n_stage}")
                if staging.m_stage:
                    partial_variants.append(f"M{staging.m_stage}")
                result.tnm_variants = partial_variants
                all_terms.update(partial_variants)
        
        # If we have stage group
        if staging.stage_group:
            # Generate stage group variants
            sg_variants = self.generate_stage_group_variants(staging.stage_group)
            result.stage_group_variants.extend(sg_variants)
            all_terms.update(sg_variants)
            
            # Look up possible TNM combinations
            if result.inferred_cancer_type:
                tnm_combos = self.tables.lookup_tnm_from_stage(
                    result.inferred_cancer_type,
                    staging.stage_group
                )
                
                if tnm_combos:
                    result.inferred_tnm = tnm_combos
                    
                    # Generate TNM variants for each combination
                    for t, n, m in tnm_combos:
                        if t != "any" and n != "any" and m != "any":
                            tnm_variants = self.generate_tnm_variants(t, n, m)
                            result.tnm_variants.extend(tnm_variants)
                            all_terms.update(tnm_variants)
                    
                    result.expansion_notes.append(
                        f"Found {len(tnm_combos)} possible TNM combinations for "
                        f"stage {staging.stage_group}"
                    )
        
        # Deduplicate
        result.tnm_variants = list(set(result.tnm_variants))
        result.stage_group_variants = list(set(result.stage_group_variants))
        result.all_search_terms = list(all_terms)
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Module-level singleton
_expander: Optional[StagingSearchExpander] = None


def get_staging_expander() -> StagingSearchExpander:
    """Get singleton staging expander instance."""
    global _expander
    if _expander is None:
        _expander = StagingSearchExpander()
    return _expander


def expand_query_with_staging(
    query: str, 
    cancer_type: Optional[str] = None
) -> StagingSearchTerms:
    """
    Convenience function to expand staging in a query.
    
    Args:
        query: Search query text
        cancer_type: Optional cancer type hint
        
    Returns:
        StagingSearchTerms with all expansion information
    """
    return get_staging_expander().expand_staging_for_search(query, cancer_type)


def get_staging_search_conditions(
    query: str, 
    cancer_type: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Get staging search conditions for structured_study_matcher.py integration.
    
    Returns dict suitable for adding to PostgreSQL ILIKE/regex conditions.
    
    Args:
        query: Search query text
        cancer_type: Optional cancer type hint
        
    Returns:
        Dict with 'tnm_patterns' and 'stage_patterns' lists
    """
    terms = expand_query_with_staging(query, cancer_type)
    
    return {
        'tnm_patterns': terms.tnm_variants,
        'stage_patterns': terms.stage_group_variants,
        'all_patterns': terms.all_search_terms,
    }


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    expander = StagingSearchExpander()
    
    test_queries = [
        "T2N1M0 breast cancer treatment options",
        "Stage IIB breast cancer prognosis",
        "cT3N2M0 lung adenocarcinoma",
        "Stage IIIA NSCLC concurrent chemoradiation",
        "pT1aN0M0 renal cell carcinoma surveillance",
        "Stage IB cervical cancer after hysterectomy",
        "T4N0M0 laryngeal cancer",
        "Stage II colorectal cancer adjuvant therapy",
    ]
    
    print("=" * 80)
    print("STAGING SEARCH EXPANDER DEMO")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n{'─' * 80}")
        print(f"QUERY: {query}")
        print(f"{'─' * 80}")
        
        result = expander.expand_staging_for_search(query)
        
        print(f"Cancer type: {result.inferred_cancer_type or 'Unknown'}")
        
        if result.extracted_staging:
            s = result.extracted_staging
            if s.has_tnm():
                print(f"Extracted TNM: T{s.t_stage} N{s.n_stage} M{s.m_stage} ({s.staging_type})")
            if s.stage_group:
                print(f"Extracted Stage: {s.stage_group}")
        
        if result.inferred_stage_groups:
            stages = ", ".join([f"{s} ({c:.0%})" for s, c in result.inferred_stage_groups])
            print(f"Inferred stage groups: {stages}")
        
        if result.inferred_tnm:
            tnms = ", ".join([f"T{t}N{n}M{m}" for t, n, m in result.inferred_tnm[:5]])
            if len(result.inferred_tnm) > 5:
                tnms += f" (+{len(result.inferred_tnm) - 5} more)"
            print(f"Inferred TNM combos: {tnms}")
        
        print(f"\nTNM variants ({len(result.tnm_variants)}): {result.tnm_variants[:5]}...")
        print(f"Stage variants ({len(result.stage_group_variants)}): {result.stage_group_variants[:5]}...")
        print(f"Total search terms: {len(result.all_search_terms)}")
        
        for note in result.expansion_notes:
            print(f"  ℹ️  {note}")
    
    print("\n" + "=" * 80)
    print("END DEMO")
    print("=" * 80)