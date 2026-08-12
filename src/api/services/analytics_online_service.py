"""
Online Analytics Service — PubMed & ClinicalTrials.gov chart data.

Accepts a natural-language request, uses GPT-4o-mini to parse the intent
into a search query + metric + grouping, fetches data from the chosen source,
then extracts quantitative values for Chart.js rendering.
"""

import json
from typing import Optional, Dict, Any

from src.api.services.literature_search_service import LiteratureSearchService

_PARSE_SYSTEM = (
    "You are a clinical research assistant. Parse a natural-language analytics request "
    "into a structured JSON object with exactly these fields:\n"
    '- "search_query": concise keyword string for database search (no natural-language phrasing)\n'
    '- "metric": the quantitative measure to extract or chart (e.g. "overall survival rate", '
    '"enrollment count", "hazard ratio", "PFS months")\n'
    '- "group_by": dimension to group results by, or null if not specified '
    '(e.g. "treatment arm", "cancer stage", "year", "phase")\n\n'
    "Return ONLY valid JSON."
)

_EXTRACT_SYSTEM = (
    "You are a clinical data extractor. Given medical study records, extract quantitative "
    "values for a specified metric and return a JSON object.\n\n"
    "Return ONLY valid JSON with exactly these fields:\n"
    '- "labels": array of string labels (one per data point, max 12)\n'
    '- "values": array of numbers matching labels\n'
    '- "unit": string unit for the metric (e.g., "%", "months", "HR", "patients")\n'
    '- "title": concise chart title (60 chars max)\n'
    '- "n_studies": integer — number of records that contributed data\n'
    '- "caveats": 1-2 sentence note on data quality/heterogeneity, or empty string\n\n'
    "Skip records where the metric is absent. If the metric is not found anywhere, "
    "return empty arrays with an explanatory caveat."
)


class AnalyticsOnlineService:
    """Extract chartable aggregate data from PubMed or ClinicalTrials.gov using LLM."""

    def __init__(self):
        self.lit_svc = LiteratureSearchService()

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _get_openai(self):
        from openai import OpenAI
        from src.core.config import get_settings
        return OpenAI(api_key=get_settings().openai_api_key)

    def _parse_nl_input(self, nl_input: str, client) -> Dict[str, Any]:
        """Use GPT-4o-mini to turn a free-text request into search_query/metric/group_by."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PARSE_SYSTEM},
                {"role": "user", "content": nl_input},
            ],
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        return {
            "search_query": parsed.get("search_query", nl_input),
            "metric": parsed.get("metric", "count"),
            "group_by": parsed.get("group_by") or None,
        }

    def _extract_chart_data(self, records_block: str, metric: str, group_by: Optional[str], client) -> Dict[str, Any]:
        """Run LLM extraction on a block of study records."""
        group_instruction = (
            f'Group results by "{group_by}".' if group_by else
            "Group by whatever natural category makes the most sense (e.g. phase, year, arm)."
        )
        user_prompt = f"Metric to extract: {metric}\n{group_instruction}\n\nRecords:\n{records_block}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=700,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        return json.loads(resp.choices[0].message.content.strip())

    # ── PubMed path ────────────────────────────────────────────────────────────

    def aggregate_from_pubmed(
        self,
        nl_input: Optional[str] = None,
        search_query: Optional[str] = None,
        metric: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search PubMed and extract a metric for charting.

        Accepts either nl_input (parsed by LLM) or explicit search_query + metric.
        Raises ValueError if fewer than 3 abstracts are found.
        """
        client = self._get_openai()

        # 1. Resolve parameters — skip LLM parsing when structured fields provided
        if search_query and metric:
            # Structured mode: use fields directly
            pass
        else:
            parsed = self._parse_nl_input(nl_input or "", client)
            search_query = parsed["search_query"]
            metric = parsed["metric"]
            group_by = group_by if group_by is not None else parsed["group_by"]

        # 2. Fetch abstracts — enrich query to target results-bearing clinical trial papers
        _OUTCOME_MAP = {
            "survival": "overall survival months",
            "os": "overall survival",
            "pfs": "progression-free survival",
            "response": "objective response rate ORR",
            "hazard": "hazard ratio",
            "mortality": "mortality rate",
            "remission": "complete remission rate",
            "efficacy": "efficacy results",
        }
        metric_lower = metric.lower()
        extra_terms = " ".join(v for k, v in _OUTCOME_MAP.items() if k in metric_lower)
        enriched_query = f"{search_query} {extra_terms} randomized trial" if extra_terms else f"{search_query} clinical trial results"
        articles = self.lit_svc.search_pubmed(enriched_query, max_results=18)
        articles_with_abstract = [a for a in articles if a.get("abstract")]
        if len(articles_with_abstract) < 3:
            # Fallback: try broader search without metric keywords
            articles = self.lit_svc.search_pubmed(search_query, max_results=18)
            articles_with_abstract = [a for a in articles if a.get("abstract")]
        if not articles_with_abstract:
            raise ValueError(
                f'No PubMed abstracts found for "{search_query}". Try a broader description.'
            )
        # Previously hard-failed below 3 abstracts, which meant most narrow
        # or patient-specific queries (niche cancer subtype + biomarker
        # combos) errored out with no chart at all. 1-2 studies is thin but
        # still worth showing — the LLM extraction step adds a caveat about
        # small sample size in that case (see _EXTRACT_SYSTEM prompt).
        low_confidence = len(articles_with_abstract) < 3

        # 3. Build records block
        records_block = ""
        for i, a in enumerate(articles_with_abstract[:15], 1):
            records_block += (
                f"[{i}] Title: {a.get('title', 'N/A')}\n"
                f"    Year: {a.get('year', 'N/A')} | Journal: {a.get('journal', 'N/A')}\n"
                f"    Abstract: {(a.get('abstract') or '')[:800]}\n\n"
            )

        # 4. Extract metric
        extracted = self._extract_chart_data(records_block, metric, group_by, client)

        # 5. Build sources
        sources = [
            {"title": a.get("title", ""), "pmid": a.get("pmid"), "year": a.get("year"), "journal": a.get("journal", "")}
            for a in articles_with_abstract[:15]
        ]

        caveats = extracted.get("caveats", "")
        if low_confidence:
            small_n_note = (
                f"Based on only {len(articles_with_abstract)} matching abstract"
                f"{'s' if len(articles_with_abstract) != 1 else ''} — treat this as directional, not definitive."
            )
            caveats = f"{small_n_note} {caveats}".strip()

        return {
            "labels": extracted.get("labels", []),
            "values": extracted.get("values", []),
            "unit": extracted.get("unit", ""),
            "title": extracted.get("title", metric),
            "n_studies": extracted.get("n_studies", len(articles_with_abstract)),
            "caveats": caveats,
            "sources": sources,
            "source_type": "pubmed",
            "parsed_query": search_query,
            "parsed_metric": metric,
        }

    # ── ClinicalTrials.gov path ────────────────────────────────────────────────

    def aggregate_from_clinical_trials(
        self,
        nl_input: Optional[str] = None,
        search_query: Optional[str] = None,
        metric: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search ClinicalTrials.gov and extract a metric for charting.

        Accepts either nl_input (parsed by LLM) or explicit search_query + metric.
        Raises ValueError if fewer than 3 studies are found.
        """
        client = self._get_openai()

        # 1. Resolve parameters
        if search_query and metric:
            pass
        else:
            parsed = self._parse_nl_input(nl_input or "", client)
            search_query = parsed["search_query"]
            metric = parsed["metric"]
            group_by = group_by if group_by is not None else parsed["group_by"]

        # 2. Fetch studies
        studies = self.lit_svc.search_clinical_trials_by_query(search_query, max_results=20)
        if not studies:
            raise ValueError(
                f'No ClinicalTrials.gov studies found for "{search_query}". Try a broader description.'
            )
        # See aggregate_from_pubmed for why this no longer hard-fails below 3.
        low_confidence = len(studies) < 3

        # 3. Build records block from CT.gov study objects
        records_block = ""
        sources = []
        for i, s in enumerate(studies[:15], 1):
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            desc = proto.get("descriptionModule", {})

            nct_id = ident.get("nctId", "N/A")
            title = ident.get("briefTitle", "N/A")
            status = status_mod.get("overallStatus", "N/A")
            phases = design.get("phases", [])
            phase_str = ", ".join(phases) if phases else "N/A"
            enrollment = (design.get("enrollmentInfo") or {}).get("count", "N/A")
            summary = (desc.get("briefSummary") or "")[:500]
            start_year = (status_mod.get("startDateStruct") or {}).get("date", "")[:4] or "N/A"

            records_block += (
                f"[{i}] NCT: {nct_id} | Title: {title}\n"
                f"    Status: {status} | Phase: {phase_str} | Enrollment: {enrollment} | Start: {start_year}\n"
                f"    Summary: {summary}\n\n"
            )
            sources.append({"title": title, "nct_id": nct_id, "status": status, "phase": phase_str})

        # 4. Extract metric — CT.gov has protocol data only (no outcomes/survival).
        #    If the requested metric sounds like a clinical outcome, pivot to enrollment count
        #    or phase distribution which CT.gov can actually provide.
        _OUTCOME_KEYWORDS = {"survival", "os", "pfs", "response rate", "hazard", "median", "mortality"}
        metric_lower = metric.lower()
        is_outcome_metric = any(kw in metric_lower for kw in _OUTCOME_KEYWORDS)
        effective_metric = metric
        if is_outcome_metric:
            effective_metric = "enrollment count by trial phase"
            group_by = "phase"

        extracted = self._extract_chart_data(records_block, effective_metric, group_by, client)

        caveats = extracted.get("caveats", "")
        if is_outcome_metric:
            caveats = (
                f"ClinicalTrials.gov contains protocol data, not outcomes. "
                f"Showing enrollment by phase instead of '{metric}'. "
                f"For outcome data (OS, PFS, response rates), use the PubMed source."
                + (f" {caveats}" if caveats else "")
            )
        if low_confidence:
            small_n_note = (
                f"Based on only {len(studies)} matching stud{'y' if len(studies) == 1 else 'ies'} "
                f"— treat this as directional, not definitive."
            )
            caveats = f"{small_n_note} {caveats}".strip()

        return {
            "labels": extracted.get("labels", []),
            "values": extracted.get("values", []),
            "unit": extracted.get("unit", ""),
            "title": extracted.get("title", effective_metric),
            "n_studies": extracted.get("n_studies", len(studies)),
            "caveats": caveats,
            "sources": sources,
            "source_type": "clinicaltrials",
            "parsed_query": search_query,
            "parsed_metric": effective_metric,
        }
