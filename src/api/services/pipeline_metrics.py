"""
Per-call pipeline metrics collector.

Each pipeline entry point calls `start(pipeline_name)` at the top of its
main method. Helpers elsewhere in the request (eligibility checker,
numerical validator, retrieval sources, etc.) call `current().incr(...)`
to bump counters without threading an object through every signature.

The collector is stored in a `contextvars.ContextVar`, so concurrent
requests under asyncio don't see each other's counts — each coroutine
chain inherits its own copy.

Pipelines call `current().summary_line()` right before returning, so the
golden-fixture tests in `tests/pipeline_golden/` can scrape the
`[PipelineMetrics]` prefix out of captured stdout and assert on counts
(source breakdown, eligibility verdict counts, safety-trigger counts).
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Dict, List, Optional


_current: contextvars.ContextVar[Optional["PipelineMetrics"]] = contextvars.ContextVar(
    "_pipeline_metrics", default=None
)


@dataclass
class PipelineMetrics:
    """
    Counters for a single pipeline run.

    Bucket conventions:
        source_counts       : {"qdrant": int, "pg": int, "pto": int, "both": int}
                              — count of distinct doc_ids contributed by each lane
                              during Phase 1 dispatch (all candidates)
        source_counts_final : {"qdrant": int, "pg": int, ...}
                              — count of distinct doc_ids that survived through
                              eligibility filtering (post-Phase 3)
        eligibility         : {"MATCH": int, "POSSIBLE": int, "NO_MATCH": int,
                               "DEMOTED": int, "RESTORED": int}
        safety              : {"numerical_stripped": int,
                               "citations_stripped": int,
                               "gate_rejected": int,
                               "gate_passed": int}
    """

    pipeline: str
    source_counts: Dict[str, int] = field(default_factory=dict)
    source_counts_final: Dict[str, int] = field(default_factory=dict)
    eligibility: Dict[str, int] = field(default_factory=dict)
    safety: Dict[str, int] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    def incr(self, bucket: str, key: str, n: int = 1) -> None:
        target = getattr(self, bucket, None)
        if target is None:
            return
        target[key] = target.get(key, 0) + n

    def record_final_source(self, source: str, n: int = 1) -> None:
        """Increment the post-eligibility source count for *source*."""
        self.source_counts_final[source] = self.source_counts_final.get(source, 0) + n

    def event(self, name: str) -> None:
        self.events.append(name)

    def _fmt(self, d: Dict[str, int]) -> str:
        return ",".join(f"{k}:{v}" for k, v in sorted(d.items()))

    def summary_line(self) -> str:
        parts = [f"pipeline={self.pipeline}"]
        if self.source_counts:
            parts.append("sources_phase1=" + self._fmt(self.source_counts))
        if self.source_counts_final:
            parts.append("sources_final=" + self._fmt(self.source_counts_final))
        if self.eligibility:
            parts.append("elig=" + self._fmt(self.eligibility))
        if self.safety:
            parts.append("safety=" + self._fmt(self.safety))
        if self.events:
            parts.append("events=" + ",".join(self.events))
        return "[PipelineMetrics] " + " ".join(parts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "pipeline": self.pipeline,
            "source_counts": dict(self.source_counts),
            "source_counts_final": dict(self.source_counts_final),
            "eligibility": dict(self.eligibility),
            "safety": dict(self.safety),
            "events": list(self.events),
        }


def start(pipeline: str) -> PipelineMetrics:
    """Create and install a fresh metrics object for this request context."""
    m = PipelineMetrics(pipeline=pipeline)
    _current.set(m)
    return m


def current() -> Optional[PipelineMetrics]:
    """Return the metrics object for this request, or None if not started."""
    return _current.get()


def incr(bucket: str, key: str, n: int = 1) -> None:
    """Convenience — bump the active collector (no-op if none installed)."""
    m = _current.get()
    if m is not None:
        m.incr(bucket, key, n)
