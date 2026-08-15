"""
Unit tests for PipelineMetrics stage-labeled source counts.

The original ``summary_line()`` outputs ``sources=qdrant:5,pg:2`` without
indicating which pipeline stage the counts refer to.  ``source_counts`` is
incremented during Phase 1 dispatch (all candidates), while
``source_provenance`` in the bundle is populated after Phase 3 filtering
(surviving studies only).  They measure different stages and should be
labeled accordingly.

After the fix:
- ``source_counts`` (Phase 1) is rendered as ``sources_phase1=...``
- A new ``source_counts_final`` bucket (post-eligibility) is rendered as
  ``sources_final=...``
- ``record_final_source(source)`` increments the final bucket

**Validates: Requirements 2.10**
"""

import pytest

from src.api.services.pipeline_metrics import PipelineMetrics


# ======================================================================
# Stage-labeled summary line
# ======================================================================

class TestSummaryLineStageLabeling:
    """summary_line() must label source counts by pipeline stage."""

    def test_phase1_sources_labeled(self):
        """source_counts should appear as 'sources_phase1=' in summary."""
        m = PipelineMetrics(pipeline="test")
        m.incr("source_counts", "qdrant", 3)
        m.incr("source_counts", "pg", 2)

        line = m.summary_line()
        assert "sources_phase1=" in line, (
            f"Expected 'sources_phase1=' in summary line, got: {line}"
        )

    def test_final_sources_labeled(self):
        """source_counts_final should appear as 'sources_final=' in summary."""
        m = PipelineMetrics(pipeline="test")
        m.record_final_source("qdrant")
        m.record_final_source("qdrant")
        m.record_final_source("pg")

        line = m.summary_line()
        assert "sources_final=" in line, (
            f"Expected 'sources_final=' in summary line, got: {line}"
        )

    def test_both_stages_present(self):
        """When both buckets have data, both labels appear."""
        m = PipelineMetrics(pipeline="test")
        m.incr("source_counts", "qdrant", 5)
        m.record_final_source("qdrant")

        line = m.summary_line()
        assert "sources_phase1=" in line
        assert "sources_final=" in line

    def test_summary_line_contains_stage_labeled_counts(self):
        """The formatted counts should follow the stage label."""
        m = PipelineMetrics(pipeline="test")
        m.incr("source_counts", "both", 3)
        m.incr("source_counts", "qdrant", 1)

        line = m.summary_line()
        # Should contain the formatted counts after the label
        assert "sources_phase1=both:3,qdrant:1" in line


# ======================================================================
# source_counts_final bucket
# ======================================================================

class TestSourceCountsFinal:
    """A separate source_counts_final bucket must exist."""

    def test_source_counts_final_exists(self):
        """PipelineMetrics should have a source_counts_final attribute."""
        m = PipelineMetrics(pipeline="test")
        assert hasattr(m, "source_counts_final")
        assert isinstance(m.source_counts_final, dict)

    def test_record_final_source_increments(self):
        """record_final_source() should increment source_counts_final."""
        m = PipelineMetrics(pipeline="test")
        m.record_final_source("qdrant")
        m.record_final_source("qdrant")
        m.record_final_source("pg")

        assert m.source_counts_final == {"qdrant": 2, "pg": 1}

    def test_record_final_source_independent_of_phase1(self):
        """source_counts and source_counts_final are independent buckets."""
        m = PipelineMetrics(pipeline="test")
        m.incr("source_counts", "qdrant", 5)
        m.record_final_source("qdrant")

        assert m.source_counts == {"qdrant": 5}
        assert m.source_counts_final == {"qdrant": 1}

    def test_to_dict_includes_final_counts(self):
        """to_dict() should include source_counts_final."""
        m = PipelineMetrics(pipeline="test")
        m.record_final_source("pg")

        d = m.to_dict()
        assert "source_counts_final" in d
        assert d["source_counts_final"] == {"pg": 1}


# ======================================================================
# Backward compatibility
# ======================================================================

class TestBackwardCompatibility:
    """Existing fields and behavior must not break."""

    def test_pipeline_prefix_preserved(self):
        """Summary line still starts with [PipelineMetrics]."""
        m = PipelineMetrics(pipeline="test")
        line = m.summary_line()
        assert line.startswith("[PipelineMetrics] ")

    def test_eligibility_and_safety_unchanged(self):
        """Non-source fields render the same as before."""
        m = PipelineMetrics(pipeline="test")
        m.incr("eligibility", "MATCH", 2)
        m.incr("safety", "gate_passed", 1)

        line = m.summary_line()
        assert "elig=MATCH:2" in line
        assert "safety=gate_passed:1" in line

    def test_empty_metrics_summary(self):
        """Empty metrics still produce a valid summary line."""
        m = PipelineMetrics(pipeline="test")
        line = m.summary_line()
        assert line == "[PipelineMetrics] pipeline=test"

    def test_incr_still_works_for_source_counts(self):
        """incr('source_counts', ...) still works as before."""
        m = PipelineMetrics(pipeline="test")
        m.incr("source_counts", "qdrant", 3)
        assert m.source_counts == {"qdrant": 3}
