"""
Analytics Service

Aggregate queries against materialized views for:
- Average survival by study phase / cancer type
- Dose regimen distributions
- Technique frequency counts
- Outcomes by staging group
- Meta-analysis forest plots (inverse-variance HR pooling)

Designed for flat schema where outcomes, patient_characteristics, etc.
are columns on the `studies` table (parsed into mv_study_summary).
"""

import math
import asyncpg
from typing import Any, Dict, List, Optional, Tuple
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

# ── Connection Pool ───────────────────────────────────────────────
_pool = None


async def _get_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_database,
            min_size=2,
            max_size=10,
            timeout=30,
        )
    return _pool


# ── Overview ──────────────────────────────────────────────────────

async def get_overview() -> Dict[str, Any]:
    """Quick stats for the analytics landing card."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                COUNT(*)::int                                      AS total_studies,
                COUNT(DISTINCT cancer_type)::int                   AS cancer_types,
                ROUND(AVG(os_rate_percent)::numeric, 1)            AS avg_os,
                ROUND(AVG(pfs_rate_percent)::numeric, 1)           AS avg_pfs,
                ROUND(AVG(median_followup_months)::numeric, 1)     AS avg_followup,
                ROUND(AVG(number_of_patients)::numeric, 0)         AS avg_n
            FROM mv_study_summary
        """)
        rad = await conn.fetchrow("""
            SELECT
                COUNT(*)::int                              AS total_rad_records,
                COUNT(DISTINCT normalized_technique)::int  AS unique_techniques,
                ROUND(AVG(best_dose_gy)::numeric, 1)       AS avg_dose
            FROM mv_radiation_summary
            WHERE best_dose_gy IS NOT NULL
        """)
        phases = await conn.fetch("""
            SELECT normalized_phase AS phase, COUNT(*)::int AS n
            FROM mv_study_summary
            WHERE normalized_phase IS NOT NULL
            GROUP BY normalized_phase
            ORDER BY n DESC
        """)
    return {
        "total_studies": row["total_studies"],
        "cancer_types": row["cancer_types"],
        "avg_os_percent": float(row["avg_os"]) if row["avg_os"] else None,
        "avg_pfs_percent": float(row["avg_pfs"]) if row["avg_pfs"] else None,
        "avg_followup_months": float(row["avg_followup"]) if row["avg_followup"] else None,
        "avg_patients_per_study": int(row["avg_n"]) if row["avg_n"] else None,
        "total_radiation_records": rad["total_rad_records"],
        "unique_techniques": rad["unique_techniques"],
        "avg_dose_gy": float(rad["avg_dose"]) if rad["avg_dose"] else None,
        "studies_by_phase": [{"phase": r["phase"], "count": r["n"]} for r in phases],
    }


# ── Generic Aggregate ─────────────────────────────────────────────

_METRIC_COLS = {
    "os_rate_percent", "os_median_months",
    "pfs_rate_percent", "pfs_median_months",
    "dfs_rate_percent", "dfs_median_months",
    "lc_rate_percent",
    "median_followup_months",
    "number_of_patients",
    "median_age_numeric",
}

_GROUP_COLS = {
    "normalized_phase": "mv_study_summary",
    "cancer_type": "mv_study_summary",
    "study_type_normalized": "mv_study_summary",
    "is_randomized": "mv_study_summary",
    "is_multi_center": "mv_study_summary",
    "country": "mv_study_summary",
}

_FILTER_COLS = {
    "cancer_type", "normalized_phase", "study_type_normalized",
    "is_randomized", "is_multi_center", "country",
}


async def aggregate_metric(
    metric: str,
    group_by: str,
    agg: str = "avg",
    filters: Optional[Dict[str, str]] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    if metric not in _METRIC_COLS:
        return {"error": f"Invalid metric. Allowed: {sorted(_METRIC_COLS)}"}
    if group_by not in _GROUP_COLS:
        return {"error": f"Invalid group_by. Allowed: {sorted(_GROUP_COLS.keys())}"}
    if agg not in ("avg", "median", "min", "max", "sum"):
        return {"error": "Invalid agg. Allowed: avg, median, min, max, sum"}

    table = _GROUP_COLS[group_by]

    if agg == "median":
        agg_expr = f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {metric})"
    else:
        agg_expr = f"{agg.upper()}({metric})"

    where_parts = [f"{metric} IS NOT NULL", f"{group_by} IS NOT NULL"]
    params: list = []
    idx = 1

    if filters:
        for col, val in filters.items():
            if col in _FILTER_COLS:
                where_parts.append(f"{col} = ${idx}")
                params.append(val)
                idx += 1

    where_clause = " AND ".join(where_parts)

    query = f"""
        SELECT
            {group_by}::text AS grp,
            ROUND({agg_expr}::numeric, 2) AS val,
            COUNT(*)::int AS n
        FROM {table}
        WHERE {where_clause}
        GROUP BY {group_by}
        HAVING COUNT(*) >= 2
        ORDER BY val DESC
        LIMIT {int(limit)}
    """

    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    unit = _guess_unit(metric)
    return {
        "metric": metric,
        "group_by": group_by,
        "agg": agg,
        "unit": unit,
        "count": len(rows),
        "labels": [r["grp"] for r in rows],
        "values": [float(r["val"]) if r["val"] else 0 for r in rows],
        "study_counts": [r["n"] for r in rows],
    }


def _guess_unit(metric: str) -> str:
    if "percent" in metric or "rate" in metric:
        return "%"
    if "months" in metric or "followup" in metric:
        return " months"
    if "patients" in metric:
        return " patients"
    if "age" in metric:
        return " years"
    if "dose" in metric:
        return " Gy"
    return ""


# ── Dose Distribution ─────────────────────────────────────────────

async def dose_distribution(
    cancer_type: Optional[str] = None,
    technique: Optional[str] = None,
    bin_width: float = 5.0,
) -> Dict[str, Any]:
    where_parts = ["best_dose_gy IS NOT NULL", "best_dose_gy > 0"]
    params: list = []
    idx = 1
    if cancer_type:
        where_parts.append(f"cancer_type = ${idx}")
        params.append(cancer_type)
        idx += 1
    if technique:
        where_parts.append(f"normalized_technique ILIKE ${idx}")
        params.append(f"%{technique}%")
        idx += 1

    where_clause = " AND ".join(where_parts)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT
                (FLOOR(best_dose_gy / {bin_width}) * {bin_width})::numeric AS bin_start,
                COUNT(*)::int AS n
            FROM mv_radiation_summary
            WHERE {where_clause}
            GROUP BY bin_start
            ORDER BY bin_start
        """, *params)

    labels = [f"{float(r['bin_start']):.0f}-{float(r['bin_start']) + bin_width:.0f} Gy" for r in rows]
    values = [r["n"] for r in rows]
    return {
        "title": "Radiation Dose Distribution",
        "labels": labels,
        "values": values,
        "unit": " studies",
        "bin_width": bin_width,
        "filters": {"cancer_type": cancer_type, "technique": technique},
    }


# ── Technique Frequency ──────────────────────────────────────────

async def technique_frequency(
    cancer_type: Optional[str] = None,
    limit: int = 15,
) -> Dict[str, Any]:
    where_parts = ["normalized_technique IS NOT NULL"]
    params: list = []
    idx = 1
    if cancer_type:
        where_parts.append(f"cancer_type = ${idx}")
        params.append(cancer_type)
        idx += 1

    where_clause = " AND ".join(where_parts)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT normalized_technique AS technique, COUNT(*)::int AS n
            FROM mv_radiation_summary
            WHERE {where_clause}
            GROUP BY normalized_technique
            ORDER BY n DESC
            LIMIT {int(limit)}
        """, *params)

    return {
        "title": "Radiation Technique Frequency",
        "labels": [r["technique"] for r in rows],
        "values": [r["n"] for r in rows],
        "unit": " studies",
        "filters": {"cancer_type": cancer_type},
    }


# ── Outcomes by Stage ─────────────────────────────────────────────

async def outcomes_by_stage(
    cancer_type: Optional[str] = None,
    metric: str = "os_rate_percent",
) -> Dict[str, Any]:
    if metric not in _METRIC_COLS:
        return {"error": f"Invalid metric. Allowed: {sorted(_METRIC_COLS)}"}

    where_parts = [f"m.{metric} IS NOT NULL", "sd.stage_category IS NOT NULL"]
    params: list = []
    idx = 1
    if cancer_type:
        where_parts.append(f"m.cancer_type = ${idx}")
        params.append(cancer_type)
        idx += 1

    where_clause = " AND ".join(where_parts)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'stage_distribution')")
        if not exists:
            return {
                "title": f"{_metric_label(metric)} by Stage",
                "metric": metric,
                "labels": [], "values": [], "study_counts": [],
                "unit": _guess_unit(metric),
                "message": "stage_distribution table not found",
            }

        rows = await conn.fetch(f"""
            SELECT
                sd.stage_category AS stage,
                ROUND(AVG(m.{metric})::numeric, 1) AS avg_val,
                COUNT(DISTINCT m.study_id)::int AS n
            FROM stage_distribution sd
            JOIN mv_study_summary m ON m.study_id = sd.study_id
            WHERE {where_clause}
            GROUP BY sd.stage_category
            HAVING COUNT(DISTINCT m.study_id) >= 2
            ORDER BY sd.stage_category
        """, *params)

    unit = _guess_unit(metric)
    return {
        "title": f"{_metric_label(metric)} by Stage",
        "metric": metric,
        "labels": [r["stage"] for r in rows],
        "values": [float(r["avg_val"]) for r in rows],
        "study_counts": [r["n"] for r in rows],
        "unit": unit,
        "filters": {"cancer_type": cancer_type},
    }


def _metric_label(m: str) -> str:
    return (
        m.replace("_rate_percent", " Rate")
         .replace("_median_months", " Median")
         .replace("_", " ")
         .title()
    )


# ── Filter Dropdowns ──────────────────────────────────────────────

async def list_cancer_types() -> List[str]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT cancer_type, COUNT(*) AS n
            FROM mv_study_summary
            WHERE cancer_type IS NOT NULL
            GROUP BY cancer_type
            HAVING COUNT(*) >= 3
            ORDER BY n DESC
        """)
    return [r["cancer_type"] for r in rows]


async def list_techniques() -> List[str]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT normalized_technique, COUNT(*) AS n
            FROM mv_radiation_summary
            WHERE normalized_technique IS NOT NULL
            GROUP BY normalized_technique
            HAVING COUNT(*) >= 2
            ORDER BY n DESC
        """)
    return [r["normalized_technique"] for r in rows]


async def list_phases() -> List[str]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT DISTINCT normalized_phase
            FROM mv_study_summary
            WHERE normalized_phase IS NOT NULL
            ORDER BY normalized_phase
        """)
    return [r["normalized_phase"] for r in rows]


# ── Meta-Analysis Forest Plot ─────────────────────────────────────

async def meta_analysis_forest(
    cancer_type: Optional[str] = None,
    metric: str = "os",
) -> Dict[str, Any]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        # Check if HR columns exist on studies
        hr_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'studies' AND column_name = 'os_hr'
            )
        """)
        if hr_exists:
            return await _forest_from_hr(conn, cancer_type, metric)
        else:
            return await _forest_from_rates(conn, cancer_type, metric)


async def _forest_from_rates(conn, cancer_type, metric):
    """
    When no HR data exists, show each study's OS/PFS rate as a
    horizontal bar chart — a 'rate forest' for comparing studies.
    """
    rate_col = f"{metric}_rate_percent"
    median_col = f"{metric}_median_months"
    raw_col = f"{metric}_raw"

    where_parts = [f"({rate_col} IS NOT NULL OR {median_col} IS NOT NULL)"]
    params: list = []
    idx = 1
    if cancer_type:
        where_parts.append(f"cancer_type = ${idx}")
        params.append(cancer_type)
        idx += 1

    where_clause = " AND ".join(where_parts)

    rows = await conn.fetch(f"""
        SELECT
            study_name, document_name, cancer_type, normalized_phase,
            number_of_patients,
            {rate_col} AS rate,
            {median_col} AS median_months,
            {raw_col} AS raw_text
        FROM mv_study_summary
        WHERE {where_clause}
        ORDER BY {rate_col} DESC NULLS LAST
        LIMIT 30
    """, *params)

    if not rows:
        return {
            "title": f"{metric.upper()} Across Studies",
            "studies": [],
            "pooled": None,
            "message": f"No studies with parseable {metric.upper()} data found.",
            "plot_type": "rates",
        }

    studies = []
    rates = []
    for r in rows:
        label = r["study_name"] or r["document_name"] or "Unknown"
        label = label[:50]
        rate = float(r["rate"]) if r["rate"] else None
        median = float(r["median_months"]) if r["median_months"] else None

        studies.append({
            "label": label,
            "rate": rate,
            "median_months": median,
            "n": r["number_of_patients"] or 0,
            "phase": r["normalized_phase"],
            "raw": (r["raw_text"] or "")[:80],
        })
        if rate is not None:
            rates.append(rate)

    pooled = None
    if rates:
        avg_rate = sum(rates) / len(rates)
        pooled = {
            "label": f"Average across {len(rates)} studies",
            "rate": round(avg_rate, 1),
            "n_studies": len(rates),
        }

    return {
        "title": f"{metric.upper()} Rate Across Studies",
        "metric": metric,
        "studies": studies,
        "pooled": pooled,
        "plot_type": "rates",
        "filters": {"cancer_type": cancer_type},
    }


async def _forest_from_hr(conn, cancer_type, metric):
    """Traditional forest plot with inverse-variance pooling when HR/CI columns exist."""
    prefix_map = {
        "os": ("os_hr", "os_hr_ci_lower", "os_hr_ci_upper"),
        "pfs": ("pfs_hr", "pfs_hr_ci_lower", "pfs_hr_ci_upper"),
    }
    if metric not in prefix_map:
        metric = "os"

    hr_col, ci_lo_col, ci_hi_col = prefix_map[metric]

    where_parts = [
        f"{hr_col} IS NOT NULL", f"{ci_lo_col} IS NOT NULL",
        f"{ci_hi_col} IS NOT NULL", f"{hr_col} > 0", f"{ci_lo_col} > 0",
    ]
    params: list = []
    idx = 1
    if cancer_type:
        where_parts.append(f"cancer_type = ${idx}")
        params.append(cancer_type)
        idx += 1

    where_clause = " AND ".join(where_parts)

    rows = await conn.fetch(f"""
        SELECT study_name, document_name, number_of_patients,
            {hr_col} AS hr, {ci_lo_col} AS ci_low, {ci_hi_col} AS ci_high
        FROM studies
        WHERE {where_clause}
        ORDER BY study_name
    """, *params)

    if not rows:
        return {
            "title": f"{metric.upper()} Hazard Ratio Forest Plot",
            "studies": [], "pooled": None,
            "message": "No studies with HR + CI data found.",
            "plot_type": "forest",
        }

    studies = []
    sum_w = 0.0
    sum_w_lnhr = 0.0

    for r in rows:
        hr = float(r["hr"])
        ci_low = float(r["ci_low"])
        ci_high = float(r["ci_high"])
        ln_hr = math.log(hr)
        se = (math.log(ci_high) - math.log(ci_low)) / 3.92
        if se <= 0:
            continue
        weight = 1.0 / (se * se)
        sum_w += weight
        sum_w_lnhr += weight * ln_hr

        label = (r["study_name"] or r["document_name"] or "Unknown")[:50]
        studies.append({
            "label": label, "hr": round(hr, 3),
            "ciLow": round(ci_low, 3), "ciHigh": round(ci_high, 3),
            "weight": round(weight, 2), "n": r["number_of_patients"] or 0,
        })

    pooled = None
    if sum_w > 0:
        pooled_ln_hr = sum_w_lnhr / sum_w
        pooled_se = 1.0 / math.sqrt(sum_w)
        pooled = {
            "label": "Pooled (Fixed Effect)",
            "hr": round(math.exp(pooled_ln_hr), 3),
            "ciLow": round(math.exp(pooled_ln_hr - 1.96 * pooled_se), 3),
            "ciHigh": round(math.exp(pooled_ln_hr + 1.96 * pooled_se), 3),
            "weight": round(sum_w, 2),
        }

    return {
        "title": f"{metric.upper()} Hazard Ratio Forest Plot",
        "metric": metric, "studies": studies, "pooled": pooled,
        "plot_type": "forest", "filters": {"cancer_type": cancer_type},
    }