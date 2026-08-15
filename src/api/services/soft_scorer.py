"""
Soft Scorer — continuous axis-level scoring for patient-study matching.

Replaces binary MATCH/MISMATCH with graduated scoring across multiple
clinical axes.  Gated behind ``settings.enable_soft_scorer``.

Scoring scale per axis:
    exact       = 1.0
    compatible  = 0.7
    partial     = 0.3 – 0.5
    not_available = 0.0  (no penalty)
    mismatch    = -0.2   (soft penalty, not hard exclusion)

Axis weights (total soft budget = 75):
    histology        15
    stage            15
    biomarkers       20
    prior_treatments 15
    treatment_setting 10
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.core.config import settings


# ── Verdict → score mapping ──────────────────────────────────────────

VERDICT_SCORES: Dict[str, float] = {
    "MATCH": 1.0,
    "EXACT": 1.0,
    "COMPATIBLE": 0.7,
    "PARTIAL": 0.4,
    "NOT_AVAILABLE": 0.0,
    "MISMATCH": -0.2,
}

# Axis name aliases coming from the eligibility JSON → canonical axis key
_AXIS_ALIASES: Dict[str, str] = {
    "histology": "histology",
    "stage": "stage",
    "biomarkers": "biomarkers",
    "prior_therapies": "prior_treatments",
    "prior_treatments": "prior_treatments",
    "treatment_setting": "treatment_setting",
}


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class SoftScore:
    """Continuous score for a study on a single axis."""
    axis: str
    score: float        # raw score (–0.2 … 1.0)
    weight: float       # axis weight
    weighted: float     # score * weight
    reason: str         # human-readable explanation


@dataclass
class SoftScoreResult:
    """Aggregated soft scores for a study."""
    study_id: str
    axis_scores: List[SoftScore]
    total_weighted: float
    max_possible: float
    normalized: float   # total_weighted / max_possible * 100  (0–100)


# ── SoftScorer ───────────────────────────────────────────────────────

class SoftScorer:
    """Scores studies on patient-level axes with continuous values.

    Accepts per-axis eligibility verdicts (from PatientEligibility) and
    maps them to graduated scores.

    Usage::

        scorer = SoftScorer()
        result = scorer.score(
            study_id="NCT00001",
            axis_verdicts={"histology": "MATCH", "biomarkers": "COMPATIBLE", ...},
        )
        print(result.normalized)  # 0–100
    """

    AXIS_WEIGHTS: Dict[str, float] = {
        "histology": 15,
        "stage": 15,
        "biomarkers": 20,
        "prior_treatments": 15,
        "treatment_setting": 10,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        study_id: str,
        axis_verdicts: Dict[str, str],
    ) -> SoftScoreResult:
        """Score a study against the patient profile on all soft axes.

        Parameters
        ----------
        study_id:
            Identifier for the study being scored.
        axis_verdicts:
            Mapping of axis name → verdict string.  Verdict strings are
            those produced by ``PatientEligibility`` (MATCH, COMPATIBLE,
            MISMATCH, NOT_AVAILABLE, PARTIAL).  Unknown axes are ignored;
            missing axes default to NOT_AVAILABLE (0 points, no penalty).

        Returns
        -------
        SoftScoreResult with per-axis breakdown and normalized total.
        """
        if not settings.enable_soft_scorer:
            # Feature-gated: return neutral result
            return self._neutral_result(study_id)

        axis_scores: List[SoftScore] = []
        for axis, weight in self.AXIS_WEIGHTS.items():
            raw_verdict = self._resolve_verdict(axis, axis_verdicts)
            score_val, reason = self._score_axis(axis, raw_verdict)
            axis_scores.append(SoftScore(
                axis=axis,
                score=score_val,
                weight=weight,
                weighted=round(score_val * weight, 4),
                reason=reason,
            ))

        total_weighted = sum(s.weighted for s in axis_scores)
        max_possible = sum(self.AXIS_WEIGHTS.values())
        normalized = round(total_weighted / max_possible * 100, 1) if max_possible else 0.0

        result = SoftScoreResult(
            study_id=study_id,
            axis_scores=axis_scores,
            total_weighted=round(total_weighted, 4),
            max_possible=max_possible,
            normalized=normalized,
        )

        print(
            f"[SoftScore] study={study_id} "
            f"total={result.normalized} "
            f"axes={{{', '.join(f'{s.axis}={s.score}' for s in axis_scores)}}}"
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_verdict(
        self,
        axis: str,
        axis_verdicts: Dict[str, str],
    ) -> str:
        """Look up the verdict for *axis*, handling alias mapping.

        If the axis is not present in *axis_verdicts* (under any alias),
        returns ``"NOT_AVAILABLE"``.
        """
        # Direct lookup
        if axis in axis_verdicts:
            return axis_verdicts[axis].upper().strip()

        # Reverse-alias lookup: check if any alias maps to this axis
        for alias, canonical in _AXIS_ALIASES.items():
            if canonical == axis and alias in axis_verdicts:
                return axis_verdicts[alias].upper().strip()

        return "NOT_AVAILABLE"

    @staticmethod
    def _score_axis(axis: str, verdict: str) -> Tuple[float, str]:
        """Map a verdict string to a numeric score and reason.

        Returns
        -------
        (score, reason) tuple.
        """
        verdict_upper = verdict.upper().strip()
        score = VERDICT_SCORES.get(verdict_upper)

        if score is not None:
            return score, f"{axis}: {verdict_upper} → {score}"

        # Fallback: treat unrecognised verdicts as NOT_AVAILABLE
        return 0.0, f"{axis}: unknown verdict '{verdict}' → 0.0"

    def _neutral_result(self, study_id: str) -> SoftScoreResult:
        """Return a neutral (50/100) result when the scorer is disabled."""
        max_possible = sum(self.AXIS_WEIGHTS.values())
        half = max_possible / 2.0
        axis_scores = [
            SoftScore(
                axis=axis,
                score=0.5,
                weight=weight,
                weighted=round(0.5 * weight, 4),
                reason=f"{axis}: scorer disabled → 0.5",
            )
            for axis, weight in self.AXIS_WEIGHTS.items()
        ]
        return SoftScoreResult(
            study_id=study_id,
            axis_scores=axis_scores,
            total_weighted=round(half, 4),
            max_possible=max_possible,
            normalized=round(half / max_possible * 100, 1) if max_possible else 0.0,
        )
