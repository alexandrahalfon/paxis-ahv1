"""
Tests for the evidence ingestion front-half added 2026-08-12:
content_extractor.py, section_chunker.py, metadata_classifier.py's
grounding/sanitize step, and evidence_ingestion_service.py's
deterministic-id helpers.

These cover the parts of the pipeline that don't require a live
Postgres/Qdrant/OpenAI connection — the HTML-cleaning heuristics, the
heading-based chunker, and the "don't let the model invent specificity"
grounding logic, which is exactly the part most worth having a
regression test for since it silently degrades (over- or under-tagging)
rather than throwing on failure.
"""

import pytest

from src.api.services.evidence.content_extractor import extract_html
from src.api.services.evidence.section_chunker import chunk_document, Chunk
from src.api.services.evidence.metadata_classifier import sanitize
from src.api.services.evidence.evidence_ingestion_service import (
    stable_id, content_hash_of, unique_section_texts,
)


NCI_STYLE_HTML = b"""
<!DOCTYPE html>
<html>
<head><title>Eating Hints - Taste Changes | National Cancer Institute</title></head>
<body>
<header><nav class="main-nav"><ul><li><a href="/">Home</a></li></ul></nav></header>
<div class="cookie-banner">We use cookies. <button>Accept</button></div>
<div id="content">
  <main>
    <article>
      <h1>Eating Hints Before, During, and After Cancer Treatment</h1>
      <p>Cancer treatment can cause side effects that make it hard to eat well.</p>
      <h2>Changes in Taste</h2>
      <p>Cancer treatment can cause foods to taste different than they did before treatment.
      Some people say food tastes metallic during chemotherapy.</p>
      <h3>Things you can try</h3>
      <ul>
        <li>Use plastic utensils if food tastes metallic.</li>
        <li>Try tart foods unless you have mouth sores.</li>
      </ul>
      <table>
        <tr><th>Symptom</th><th>Suggested action</th></tr>
        <tr><td>Metallic taste</td><td>Use plastic utensils</td></tr>
      </table>
    </article>
  </main>
  <div class="related-articles"><h2>Related</h2><ul><li><a href="/x">Other</a></li></ul></div>
  <div class="social-share">Share: <a href="#">FB</a></div>
</div>
<footer class="site-footer"><p>&copy; NCI. Privacy policy.</p></footer>
</body>
</html>
"""


class TestContentExtractor:
    def test_strips_chrome(self):
        doc = extract_html(NCI_STYLE_HTML, source_url="https://cancer.gov/eating-hints")
        leaked = [kw for kw in ("cookie", "Related", "Share:", "Privacy policy", "main-nav")
                  if kw in doc.plain_text]
        assert leaked == [], f"chrome leaked into extracted content: {leaked}"

    def test_extracts_heading_hierarchy(self):
        doc = extract_html(NCI_STYLE_HTML)
        headings = [s.heading for s in doc.sections]
        assert "Changes in Taste" in headings
        assert "Things you can try" in headings

    def test_preserves_table_content(self):
        doc = extract_html(NCI_STYLE_HTML)
        assert "Metallic taste" in doc.plain_text
        assert "Use plastic utensils" in doc.plain_text

    def test_title_strips_site_suffix(self):
        doc = extract_html(NCI_STYLE_HTML)
        assert "National Cancer Institute" not in doc.title

    def test_is_usable_true_for_real_content(self):
        assert extract_html(NCI_STYLE_HTML).is_usable()

    def test_is_usable_false_for_near_empty_page(self):
        thin = b"<html><body><main><p>Not found.</p></main></body></html>"
        assert not extract_html(thin).is_usable()


class TestSectionChunker:
    def test_chunks_addressed_by_heading(self):
        doc = extract_html(NCI_STYLE_HTML)
        chunks = chunk_document(doc)
        taste_chunks = [c for c in chunks if c.section_title and "taste" in c.section_title.lower()]
        assert len(taste_chunks) == 1
        assert "metallic" in taste_chunks[0].text.lower()

    def test_falls_back_to_fixed_window_when_no_sections(self):
        from src.api.services.evidence.content_extractor import ExtractedDocument
        doc = ExtractedDocument(title="Plain text doc", sections=[], plain_text="word " * 1000)
        chunks = chunk_document(doc)
        assert len(chunks) > 1
        assert all(c.section_title is None for c in chunks)

    def test_empty_document_yields_no_chunks(self):
        from src.api.services.evidence.content_extractor import ExtractedDocument
        assert chunk_document(ExtractedDocument(title="Empty", sections=[], plain_text="")) == []


class TestMetadataClassifierGrounding:
    """The model's raw output should never be trusted verbatim — anything
    that doesn't resolve against clinical_normalization.py's controlled
    vocabulary must be dropped, and an empty cancer_types list must
    become ["all"] rather than staying empty (which would mean 'matches
    nothing' instead of 'matches everyone')."""

    def test_drops_unrecognized_intent(self):
        result = sanitize({"intents": ["nutrition", "not_a_real_intent"]})
        assert result.intents == ["nutrition"]

    def test_drops_unrecognized_treatment_modality(self):
        result = sanitize({"treatment_modalities": ["chemotherapy", "acupuncture"]})
        assert result.treatment_modalities == ["chemotherapy"]

    def test_empty_cancer_types_defaults_to_all(self):
        result = sanitize({"cancer_types": []})
        assert result.cancer_types == ["all"]

    def test_drops_fabricated_regimen(self):
        result = sanitize({"regimens": ["FOLFOX", "NotARealRegimen"]})
        assert result.regimens == ["FOLFOX"]

    def test_normalizes_and_dedupes_drug_aliases(self):
        result = sanitize({"drugs": ["Keytruda", "pembrolizumab"]})
        assert result.drugs == ["pembrolizumab"]

    def test_drops_unrecognized_drug(self):
        result = sanitize({"drugs": ["SomeMadeUpDrugName"]})
        assert result.drugs == []

    def test_symptom_synonyms_normalize_to_same_canonical_term(self):
        result = sanitize({"symptoms": ["metallic taste", "everything tastes weird"]})
        assert result.symptoms == ["dysgeusia"]

    def test_drops_unrecognized_symptom(self):
        result = sanitize({"symptoms": ["an entirely novel symptom nobody has heard of"]})
        assert result.symptoms == []

    def test_invalid_content_type_falls_back_to_patient_education(self):
        result = sanitize({"content_type": "not_a_real_type"})
        assert result.content_type == "patient_education"


class TestUniqueSectionTexts:
    """Chunk-level metadata classification (2026-08-12): each unique
    section gets classified once, using the section's full parent_text
    when a long section was split into overlapping child chunks, so
    unrelated sections of one document don't inherit each other's
    applicability tags (see evidence_ingestion_service.py's
    _classify_sections)."""

    def test_one_entry_per_unique_heading_from_real_document(self):
        doc = extract_html(NCI_STYLE_HTML)
        chunks = chunk_document(doc)
        sections = unique_section_texts(chunks)
        assert "Changes in Taste" in sections
        assert "Things you can try" in sections
        assert "metallic" in sections["Changes in Taste"].lower()

    def test_split_children_of_one_section_collapse_to_one_entry(self):
        long_text = "word " * 500  # forces child-window splitting
        chunks = [
            Chunk(text=f"Long Section\n{long_text[:1200]}", section_title="Long Section",
                  chunk_index=0, parent_text=long_text),
            Chunk(text=f"Long Section\n{long_text[1050:2250]}", section_title="Long Section",
                  chunk_index=1, parent_text=long_text),
        ]
        sections = unique_section_texts(chunks)
        assert list(sections.keys()) == ["Long Section"]
        assert sections["Long Section"] == long_text  # classified against the FULL section

    def test_headingless_chunks_are_skipped(self):
        chunks = [Chunk(text="some fallback text", section_title=None, chunk_index=0)]
        assert unique_section_texts(chunks) == {}

    def test_first_seen_wins_for_duplicate_headings(self):
        chunks = [
            Chunk(text="A", section_title="Repeated", chunk_index=0, parent_text="first"),
            Chunk(text="B", section_title="Repeated", chunk_index=1, parent_text="second"),
        ]
        assert unique_section_texts(chunks) == {"Repeated": "first"}


class TestDeterministicIds:
    def test_stable_id_is_deterministic(self):
        a = stable_id("document", "nci", "https://cancer.gov/eating-hints")
        b = stable_id("document", "nci", "https://cancer.gov/eating-hints")
        assert a == b

    def test_stable_id_differs_for_different_inputs(self):
        a = stable_id("document", "nci", "https://cancer.gov/page-a")
        b = stable_id("document", "nci", "https://cancer.gov/page-b")
        assert a != b

    def test_stable_id_is_valid_uuid_format(self):
        import uuid
        parsed = uuid.UUID(stable_id("document", "nci", "https://cancer.gov/x"))
        assert str(parsed) == stable_id("document", "nci", "https://cancer.gov/x")

    def test_content_hash_deterministic_and_sensitive_to_change(self):
        assert content_hash_of("some text") == content_hash_of("some text")
        assert content_hash_of("some text") != content_hash_of("some text!")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
