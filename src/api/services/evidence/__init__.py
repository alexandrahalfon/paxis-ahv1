"""
Patient-facing evidence layer (Phase 3 + Phase 4 of the consumer-platform
redesign).

    source_registry.py          evidence_sources / evidence_documents CRUD
    evidence_ingestion_service.py  chunk + embed + upsert one document's
                                    already-fetched text into a named
                                    Qdrant collection (does not fetch from
                                    the web itself — see its module docstring)
    multi_corpus_retriever.py   parallel search across literature +
                                 patient-education + medication + guideline
                                 collections, degrading gracefully when a
                                 collection is empty or missing
    applicability_scorer.py     patient-state x evidence-candidate scoring
    retrieval_planner.py        intent -> which corpora, hard constraints,
                                 soft boosts
    evidence_packet_builder.py  assembles the final packet handed to
                                 generation
    patient_context_service.py intent classification + patient state load

Hard boundary (see CLAUDE.md-adjacent architecture review, section 14 and
33): nothing in this package ever reads or writes patient PHI tables
directly except to read the already-built patient_state_snapshot for
context selection. Nothing in the patient/ package (Phase 0/1/2) imports
from this package. Community content (Phase 7) never enters retrieval
here either — see community/ for that boundary.
"""
