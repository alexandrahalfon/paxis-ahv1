# Product Overview

Paxis is an AI-powered clinical evidence platform for oncology. It ingests medical literature (PDFs), stores structured chunks in Qdrant (vector DB) and PostgreSQL (relational metadata), and answers complex clinical queries by retrieving and synthesising evidence.

## Core Use Case

A clinician submits a complex patient profile (e.g., age, cancer type, stage, biomarkers, treatment history) and receives matched clinical studies with relevant evidence, eligibility analysis, and treatment recommendations.

## Key Capabilities

- **RAG Query Engine** — natural-language clinical questions answered with cited evidence from peer-reviewed literature
- **Patient-to-Trial Matching** — extracts patient profiles from free text and matches against internal knowledge base and ClinicalTrials.gov
- **Virtual Tumor Board** — multi-agent architecture with specialty-specific retrieval (radiation oncology, medical oncology, etc.)
- **Treatment & Study Comparison** — side-by-side comparisons with outcomes data and visualizations
- **Document Upload & Processing** — PDF ingestion with Mistral OCR, study profile extraction, and vector embedding
- **Patient Portal** — patient-facing interface for education, symptom tracking, and clinician connection
- **Physician Beta** — physician-specific RAG pipeline with evidence grounding layer (feature-flagged)

## Users

- Oncologists and radiation oncologists (primary)
- Clinical researchers
- Patients (via patient portal)

## Domain

Oncology — specifically radiation oncology, head & neck cancers, immunotherapy, and clinical trial eligibility.

## License

Proprietary — all rights reserved.
