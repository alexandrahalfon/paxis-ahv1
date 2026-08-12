# Paxis RAG Pipeline - Architecture Diagrams

Detailed step-by-step diagrams for each phase of the Paxis retrieval-augmented generation pipeline.

---

## Phase 1: Document Ingestion & Processing

```mermaid
flowchart TD
    A[PDF Upload] --> B{Source?}
    B -->|Knowledge Base| C[Batch Ingestion]
    B -->|User Upload| D[User Upload Service]

    C --> E[CompleteDocumentProcessor]
    D --> F{Existing Doc?}
    F -->|DOI match in Qdrant| G[Link to User Account]
    F -->|New document| E

    E --> P1[Phase 1: Mistral OCR]
    P1 --> P1a[Upload PDF to Mistral API]
    P1a --> P1b[Extract structured content by heading]
    P1b --> P1c[Separate references section]

    P1c --> P2[Phase 2: Pixtral Vision]
    P2 --> P2a[Convert PDF pages to images - 200 DPI]
    P2a --> P2b[Send up to 10 page images to Pixtral LLM]
    P2b --> P2c[Extract detailed text + visual content]

    P2c --> P3[Phase 3: Document Indexing]
    P3 --> P3a[Merge OCR + Pixtral content]
    P3a --> P3b[Create structured document_index JSON]
    P3b --> P3c[Organize by section/paragraph]

    P3c --> P4[Phase 4: Table & Figure Extraction]
    P4 --> P4a[Each page image → Pixtral]
    P4a --> P4b[Extract tables as structured JSON]
    P4b --> P4c[Extract figures - survival curves, charts]
    P4c --> P4d[Save tables as CSV]

    P4d --> P5[Phase 5: Study Profile Extraction]
    P5 --> P5a[GPT-4o-mini extracts structured metadata]
    P5a --> P5b[Cancer type, staging, outcomes, treatment arms]
    P5b --> P5c[Store profile in PostgreSQL]

    P5c --> P6[Phase 6: GCP Sync]
    P6 --> P6a[Upload processed outputs to GCS bucket]

    style A fill:#e1f5fe
    style E fill:#fff3e0
    style P1 fill:#f3e5f5
    style P2 fill:#f3e5f5
    style P3 fill:#f3e5f5
    style P4 fill:#f3e5f5
    style P5 fill:#f3e5f5
    style P6 fill:#f3e5f5
```

### Processing Outputs Table

| Output | Format | Description |
|--------|--------|-------------|
| Structured Content | JSON | OCR-extracted text organized by heading |
| Pixtral Content | Text | Vision-extracted detailed content |
| Document Index | JSON | Merged structured index (section → paragraphs) |
| Tables | JSON + CSV | Extracted tabular data with headers |
| Figures | JSON | Survival curves, charts, diagrams |
| Study Profile | JSON → PostgreSQL | Structured metadata (cancer type, outcomes, arms) |
| Summary Report | JSON | Processing statistics and file manifest |

---

## Phase 2: Chunking + Embedding + Database Upsert

### 2a. Chunking Strategy

```mermaid
flowchart TD
    A[Document Index JSON] --> B{Content Type?}

    B -->|Paragraphs| C[Paragraph Chunking]
    B -->|Tables| D[Table Row Chunking]

    C --> C1[Group paragraphs by doc_id + section]
    C1 --> C2[Sort by section_paragraph_num]
    C2 --> C3[Keyword Tagging]
    C3 --> C3a[Match 1200+ medical terms against text]
    C3a --> C3b[Tag: keywords_flat + keyword_matches]
    C3b --> C4[Section Windowing]
    C4 --> C4a[Concatenate paragraphs in order]
    C4a --> C4b{Cumulative tokens > 600?}
    C4b -->|No| C4c[Add next paragraph to window]
    C4c --> C4a
    C4b -->|Yes| C4d[Flush window as paragraph_window chunk]
    C4d --> C4e[Union keyword metadata across window]
    C4e --> C4f[Start new window]
    C4f --> C4a

    D --> D1[Each table row → atomic chunk]
    D1 --> D2[Include column headers as context]
    D2 --> D3[Tag: chunk_type = table_row]
    D3 --> D4[No windowing applied]

    C4d --> E[Final Chunks JSONL]
    D4 --> E

    style A fill:#e1f5fe
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#f3e5f5
```

### Chunking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max tokens per chunk | 600 | Section window token limit |
| Tokenizer | tiktoken o200k_base | Token counting |
| Min paragraph length | 20 chars | Skip very short paragraphs |
| Keyword source | extractor_keywords.json | 1200+ medical terms |
| Keyword matching | Case-insensitive substring | Against chunk text |
| Keywords in embedding | No | Clean text only for embeddings |

### 2b. Embedding

```mermaid
flowchart TD
    A[Section-Window Chunks JSONL] --> B[Batch Reader]
    B --> C[Batch of 64 chunks]
    C --> D[Extract clean text - no keywords]
    D --> E[OpenAI Embeddings API]
    E --> F[text-embedding-3-large]
    F --> G[3072-dimensional vectors]
    G --> H[Pair: chunk + embedding]
    H --> I{More batches?}
    I -->|Yes| B
    I -->|No| J[All embeddings ready]

    style A fill:#e1f5fe
    style E fill:#fff3e0
    style F fill:#f3e5f5
    style J fill:#e8f5e9
```

### Embedding Configuration

| Parameter | Value |
|-----------|-------|
| Model | text-embedding-3-large |
| Dimensions | 3072 |
| Batch size | 64 chunks per API call |
| Input | Clean text only (no keyword augmentation) |
| Point ID | UUID5 deterministic from chunk_id |

### 2c. Database Upsert

```mermaid
flowchart TD
    A[Embedded Chunks] --> B[Build Qdrant Points]
    B --> B1[point_id = UUID5 from chunk_id]
    B1 --> B2[vector = 3072-dim embedding]
    B2 --> B3[payload = full chunk dict]
    B3 --> B3a[text, doc_id, category, section]
    B3a --> B3b[metadata.keywords_flat]
    B3b --> B3c[doc_meta.citation, year]
    B3c --> B3d[doc_level_* fields for patient matching]

    B3d --> C[Batch Upsert to Qdrant]
    C --> C1[512 points per upsert batch]
    C1 --> C2[Collection: exueed_kb_latest]

    C2 --> D[Build PTO Frames]
    D --> D1[Per-document LLM summary frames]
    D1 --> D2[Per-section embeddings]
    D2 --> D3[Validate against source text]
    D3 --> D4[Upsert PTO points to Qdrant]

    A --> E[Study Profile → PostgreSQL]
    E --> E1[Structured metadata per study]
    E1 --> E2[Cancer type, staging, outcomes]
    E2 --> E3[Lookup tables for filtering]

    style A fill:#e1f5fe
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#f3e5f5
```

### Storage Architecture

| Store | Content | Purpose |
|-------|---------|---------|
| Qdrant (exueed_kb_latest) | Chunk vectors + full payload + doc_level_* metadata | Semantic search + patient matching |
| Qdrant (PTO frames) | Per-doc summary vectors | Document-level retrieval |
| PostgreSQL (display-study-details) | Study profiles | Structured filtering + eligibility criteria |
| PostgreSQL (study-profiles) | Normalized profiles + lookup tables | Faceted search |
| PostgreSQL (exueed_cache) | User data, uploads, saved cases | Application state |

---

## Phase 3: Query Processing + Retrieval + Answer Extraction

### 3a. Query Processing (Phases 0-3)

```mermaid
flowchart TD
    A[User Query] --> P0[Phase 0: Preprocessing]
    P0 --> P0a[Fetch user preferences]
    P0a --> P0b[Clinical entity extraction - regex]
    P0b --> P0c[Trial mention detection]

    P0c --> P1[Phase 1: Classification + Structuring]
    P1 --> P1a{Regex patterns match?}
    P1a -->|Yes - high confidence| P1b[Assign type: dose/staging/treatment/etc.]
    P1a -->|No - ambiguous| P1c[GPT-4o-mini LLM classifier]
    P1c --> P1b
    P1b --> P1d[Fast query structuring - regex]
    P1d --> P1e[Extract: PatientContext, CancerContext, TreatmentContext, ClinicalHistory]
    P1e --> P1f[Merge accumulated conversation context]

    P1f --> P2[Phase 2: Query Expansion - Bidirectional]
    P2 --> P2a[Forward: abbreviation → full term]
    P2a --> P2a1["NSCLC → non-small cell lung cancer"]
    P2 --> P2b[Reverse: full term → abbreviation]
    P2b --> P2b1["radiation therapy → RT"]
    P2 --> P2c[Staging synonyms]
    P2c --> P2c1["pN1 → pathologic N1 → node positive"]
    P2 --> P2d[Clinical concept synonyms]
    P2d --> P2d1["adjuvant → postoperative"]
    P2 --> P2e[Cancer ontology expansion]
    P2e --> P2e1["breast cancer → mammary carcinoma + subtypes"]
    P2 --> P2f[Drug brand ↔ generic ↔ class]
    P2f --> P2f1["Keytruda → pembrolizumab → anti-PD-1"]
    P2 --> P2g[AJCC staging table lookups]
    P2g --> P2g1["T2N1 → Stage III for head & neck"]
    P2 --> P2h[Ontology resolver tokens]
    P2h --> P2h1["Canonical labels injected into embedding input"]

    P2a1 --> D[Expanded Query]
    P2b1 --> D
    P2c1 --> D
    P2d1 --> D
    P2e1 --> D
    P2f1 --> D
    P2g1 --> D
    P2h1 --> D

    D --> P3[Phase 3: Embedding + Parallel LLM Extraction]
    P3 --> P3a[Generate query embedding - text-embedding-3-large]
    P3 --> P3b{Complex query?}
    P3b -->|>150 chars OR >4 commas OR keyword hits| P3c[LLM Extraction - GPT-4o - parallel]
    P3b -->|Simple| P3d[Use regex results only]
    P3c --> P3e[Build ClinicalProfile via SynonymIndex normalization]
    P3e --> P3f[Enrich profile from QueryStructure - fallback gaps]
    P3f --> P3g[Apply profile to structure - fill category/histology/biomarkers]
    P3d --> P3h[Final Structured Query + ClinicalProfile]
    P3g --> P3h

    style A fill:#e1f5fe
    style P0 fill:#fff3e0
    style P1 fill:#e8f5e9
    style P2 fill:#f3e5f5
    style P3 fill:#e1f5fe
```

### Query Types

| Type | Example | Prompt Template |
|------|---------|-----------------|
| treatment_recommendation | "What is the standard treatment for..." | Recommended Treatment / Dose / Guideline Level |
| dose_question | "What dose for prostate SBRT?" | Prescribed Dose / Fractionation / Target Volume |
| indication_question | "When is adjuvant RT indicated?" | Indication / Patient Selection / Contraindications |
| trial_results | "What did RTOG 0617 show?" | Trial Design / Key Outcomes / Clinical Relevance |
| staging | "What stage is T3N1M0 lung?" | Stage Classification / Criteria / Prognosis |
| side_effects | "Toxicity of concurrent chemoRT?" | Acute/Late Effects / Grading / Management |
| comparison | "SBRT vs conventional fractionation?" | Head-to-head / Outcomes / Patient Selection |
| workup | "Workup for newly diagnosed NSCLC?" | Imaging / Labs / Pathology / Staging |
| general | Other queries | Evidence Summary / Key Findings |

### 3b. Retrieval (Phases 3.5-9)

```mermaid
flowchart TD
    A[Expanded Query + Structure + ClinicalProfile] --> P35[Phase 3.5: Qdrant Filter Construction]
    P35 --> P35a[Build category filter from ClinicalProfile]
    P35a --> P35b[Category variant matching - multiple spellings]
    P35b --> P35c[Priority: explicit > prefilter > query_structure.filter_category]

    P35c --> P4[Phase 4: Three-Source Parallel Retrieval]

    P4 --> S1[Source 1: Qdrant Vector Search]
    S1 --> S1a[Cosine similarity - top N candidates]
    S1a --> S1b[Category filter with should-clause variants]
    S1b --> S1c[Return candidates with scores]

    P4 --> S2[Source 2: PostgreSQL Structured Match]
    S2 --> S2a[Build SQL from QueryStructure axes]
    S2a --> S2b[Match: cancer_type, location, stage, histology]
    S2b --> S2c[Score threshold ≥ 0.35]
    S2c --> S2d[Return matching study doc_ids + scores]

    P4 --> S3[Source 3: PTO Frame Index]
    S3 --> S3a[Document-level semantic search]
    S3a --> S3b[Score threshold ≥ 0.28]
    S3b --> S3c[Return doc_ids from PTO matches]

    S1c --> P5[Phase 5: Candidate Conversion + PG Boost]
    S2d --> P5
    S3c --> P5
    P5 --> P5a[Merge candidates from all sources]
    P5a --> P5b[Boost PG-matched candidates - factor 0.3]

    P5b --> P6[Phase 6: BM25 Lexical Scoring + Preference Filters]
    P6 --> P6a[BM25 scores on dense pool]
    P6a --> P6b[Apply user preference filters]

    P6b --> P7[Phase 7: RRF Fusion]
    P7 --> P7a[Reciprocal Rank Fusion - k=60]
    P7a --> P7b[Combine dense + lexical scores]

    P7b --> P8[Phase 8: Structure-Aware Rerank]
    P8 --> P8a[Rerank with clinical profile matching]
    P8a --> P8b[Boost chunks matching patient axes]

    P8b --> P9[Phase 9: Cross-Encoder Rerank]
    P9 --> P9a[Query distillation - build_reranker_query]
    P9a --> P9b[Compact keyword string from QueryStructure]
    P9b --> P9c[ms-marco-MiniLM-L-6-v2 scoring]
    P9c --> P9d[Re-sort by cross-encoder score]

    style A fill:#e1f5fe
    style P4 fill:#fff3e0
    style P8 fill:#f3e5f5
    style P9 fill:#e8f5e9
```

### 3c. Post-Retrieval Processing (Phases 10-13)

```mermaid
flowchart TD
    A[Cross-Encoder Reranked Results] --> P10[Phase 10: Score Boosting Pipeline]
    P10 --> P10a[Dose chunk boost - for dose queries]
    P10a --> P10b[Trial name boost - detected trial mentions]
    P10b --> P10c[Lane separation - trials vs guidelines/landmarks]
    P10c --> P10c1[Trial lane: up to 12 docs]
    P10c --> P10c2[Guideline lane: up to 5 docs - capped, not inflated]
    P10c1 --> P10d[Module-specific boost]
    P10c2 --> P10d
    P10d --> P10d1[general_knowledge / patient_specific / evidence_exploration]

    P10d1 --> P11[Phase 11: Dedup + Caps]
    P11 --> P11a[Max 2 chunks per doc]
    P11a --> P11b[Max 1 chunk per section]
    P11b --> P11c[Max 2 table rows per table]
    P11c --> P11d[Sort preference boost]

    P11d --> P12[Phase 12: Evidence Packs]
    P12 --> P12a[Build evidence packs with neighbor windows]
    P12a --> P12b[Include: title, citation, year, category]

    P12b --> P13[Phase 13: NCCN Gap Detection]
    P13 --> P13a[Detect missing guideline coverage]

    P13a --> POST[Post-Retrieval Processing]
    POST --> POST1[Patient-match scoring per source - 0-100]
    POST1 --> POST2[Cancer-type post-filter - remove wrong-cancer studies]
    POST2 --> POST3[Patient eligibility hard filter + boost]
    POST3 --> POST3a[LLM-based per-study eligibility - 8 axes]
    POST3a --> POST3b[Hard-drop: cancer_type, disease_status, surgical_candidacy]
    POST3b --> POST3c[Hard-drop: study_exclusions_violated]
    POST3c --> POST4[Context document boost - conversation continuity]

    POST4 --> FINAL[Final Evidence Bundle]

    style A fill:#e1f5fe
    style P10 fill:#fff3e0
    style P11 fill:#f3e5f5
    style POST fill:#e8f5e9
    style FINAL fill:#e1f5fe
```

### Retrieval Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Qdrant candidates | Top 100 | Initial vector search pool |
| PostgreSQL threshold | 0.35 | Minimum structured match score |
| PTO threshold | 0.28 | Minimum PTO frame score |
| PG boost factor | 0.3 | Additive boost for PG-matched candidates |
| RRF k | 60 | Reciprocal Rank Fusion constant |
| Rerank pool | Top 50 | Candidates sent to structure-aware rerank |
| Cross-encoder | ms-marco-MiniLM-L-6-v2 | Neural reranker (distilled query) |
| Trial lane cap | 12 docs | Maximum trial documents in lane |
| Guideline lane cap | 5 docs | Maximum guideline/landmark documents |
| Max per doc | 2 chunks | Dedup cap per document |
| Max per section | 1 chunk | Dedup cap per section |

### Patient-Match Scoring (per source)

```mermaid
flowchart TD
    A[ClinicalProfile from query] --> B[For each cited study]
    B --> C[Read doc_level_* metadata from Qdrant payload]
    C --> D[Score 7 weighted axes]
    D --> D1["cancer_type (weight: 25)"]
    D --> D2["cancer_sites (weight: 15)"]
    D --> D3["histologies (weight: 15)"]
    D --> D4["biomarkers (weight: 15)"]
    D --> D5["stages (weight: 10)"]
    D --> D6["prior_treatments (weight: 10)"]
    D --> D7["disease_status (weight: 10)"]

    D1 --> E{Study has data for axis?}
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    D6 --> E
    D7 --> E
    E -->|Yes| F[overlap_ratio = patient ∩ study / patient values]
    E -->|No - NA| G[Axis excluded from denominator]
    F --> H[weighted_sum += weight * ratio]
    G --> H

    H --> I{Cancer type mismatch?}
    I -->|Yes| J[Cap score at 35]
    I -->|No| K[score = 100 * weighted_sum / denominator]
    J --> L[Final 0-100 score per study]
    K --> L

    style A fill:#e1f5fe
    style D fill:#fff3e0
    style L fill:#e8f5e9
```

### Patient Eligibility Hard Filter

| Axis | Hard Drop? | Description |
|------|-----------|-------------|
| cancer_type | Yes | Wrong cancer → remove study |
| histology | No | Mismatch penalizes score only |
| stage | No | Mismatch penalizes score only |
| prior_therapies | No | Mismatch penalizes score only |
| biomarkers | No | Mismatch penalizes score only |
| disease_status | Yes | Wrong trajectory → remove study |
| surgical_candidacy | Yes | Incompatible surgical status → remove |
| study_exclusions_violated | Yes (inverted) | Patient violates exclusion → remove |

---

## Phase 4: Generation

```mermaid
flowchart TD
    A[Assembled Prompt + Evidence] --> B[GPT-4o Generation]
    B --> B1[Query-type-specific system prompt - 9 templates]
    B1 --> B2[Structured evidence as user context]
    B2 --> B3[Staging context from normalizer]
    B3 --> B4[Patient eligibility verification]
    B4 --> B5[Conversation history - up to 10 entries]
    B5 --> B6[Generate comprehensive answer]

    B6 --> C[Post-Generation Validation]
    C --> C1[Numerical Validation]
    C1 --> C1a[Extract all numbers from answer]
    C1a --> C1b[Cross-check against source evidence]
    C1b --> C1c{Number found in sources?}
    C1c -->|Yes| C1d[Keep number]
    C1c -->|No| C1e[Strip unvalidated number]

    C1d --> C2[Statistical Enrichment]
    C1e --> C2
    C2 --> C2a[Add confidence intervals from evidence]
    C2a --> C2b[Add p-values from evidence]
    C2b --> C2c[Add hazard ratios from evidence]

    C2c --> D[Structured Output Parsing]
    D --> D1[Parse into layered structure]
    D1 --> D2[brief_answer: 1-2 sentence summary]
    D2 --> D3[structured_details: key-value pairs by section]
    D3 --> D4[explanation: full narrative with citations]

    D4 --> E[Follow-Up Generation]
    E --> E1[Generate 3-5 contextual follow-up questions]
    E1 --> E2[Based on query type + evidence gaps]

    E2 --> F[Citation Assembly]
    F --> F1[Map cited studies to full references]
    F1 --> F2[Fetch citation counts - Semantic Scholar API]
    F2 --> F3[Include DOI links where available]

    F3 --> G[Final Response]
    G --> G1[answer: synthesized text]
    G1 --> G2[sources: cited studies with metadata + patient_match_score]
    G2 --> G3[follow_ups: suggested next questions]
    G3 --> G4[query_type: classified type]
    G4 --> G5[structured_output: layered response object]
    G5 --> G6[conversation_context: for session continuity]
    G6 --> G7[module_name: routing classification]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style F fill:#fff3e0
    style G fill:#e1f5fe
```

### Generation Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | GPT-4o | Primary generation model |
| Temperature | 0.1 | Low for factual accuracy |
| Prompt templates | 9 types | Query-type-specific formatting |
| Numerical validation | Post-hoc | Strip numbers not in evidence |
| Citation format | Author (Year) | Inline with full reference list |
| Follow-ups | 3-5 | Contextual suggestions |

### Response Structure

| Field | Type | Description |
|-------|------|-------------|
| answer | string | Full synthesized narrative |
| brief_answer | string | 1-2 sentence summary |
| structured_details | object | Key-value pairs by section header |
| sources | array | Cited studies with title, citation, year, doc_id, patient_match_score |
| follow_ups | array | Suggested follow-up questions |
| query_type | string | Classified query type |
| module_name | string | Module classification (general_knowledge/patient_specific/evidence_exploration) |
| metadata | object | Timings, retrieval stats, model used |
| conversation_context | object | Doc IDs + structure for session continuity |

---

## Phase 5: PDF Report Generation

```mermaid
flowchart TD
    A[Query/Match/Comparison Result] --> B{Report Type?}
    B -->|Patient Match| C[Patient Match Report]
    B -->|Treatment Comparison| D[Treatment Comparison Report]
    B -->|Query Result| E[Query Report]

    C --> C1[Patient summary header]
    C1 --> C2[Match table: study, score, rationale, treatment]

    D --> D1[Side-by-side comparison table]
    D1 --> D2[Efficacy / Safety / Dosing / Outcomes columns]
    D2 --> D3[Comparison summary]

    E --> E1{Format?}
    E1 -->|standard| E2[Question + Answer + Details + Sources table]
    E1 -->|patient_handout| E3[Patient-friendly language + key takeaways]
    E1 -->|clinic_note| E4[Clinical question + Recommendation + Evidence + Next steps]

    C2 --> F[ReportLab PDF Generation]
    D3 --> F
    E2 --> F
    E3 --> F
    E4 --> F
    F --> G[PDF bytes returned via API]

    style A fill:#e1f5fe
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#f3e5f5
    style G fill:#e1f5fe
```

### Report Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/report/patient-match` | POST | PDF from patient matching result |
| `/api/report/treatment-comparison` | POST | PDF from treatment comparison |
| `/api/report/query` | POST | PDF from query result (3 formats) |

---

## End-to-End Pipeline Overview

```mermaid
flowchart LR
    subgraph Ingestion ["Phase 1: Ingestion"]
        I1[PDF] --> I2[Mistral OCR + Pixtral Vision]
        I2 --> I3[Document Index]
    end

    subgraph Indexing ["Phase 2: Chunking + Embedding"]
        X1[Section Windowing] --> X2[Keyword Tagging]
        X2 --> X3[text-embedding-3-large]
        X3 --> X4[Qdrant + PostgreSQL]
    end

    subgraph Query ["Phase 3: Query + Retrieval"]
        Q1[User Query] --> Q2[Classify + Expand + Structure]
        Q2 --> Q2a[ClinicalProfile via SynonymIndex]
        Q2a --> Q3[3-Source Parallel Retrieval]
        Q3 --> Q4[RRF + Structure Rerank + Cross-Encoder]
        Q4 --> Q5[Lane Separation + Eligibility Filter]
        Q5 --> Q6[Patient-Match Scoring per Source]
    end

    subgraph Generation ["Phase 4: Generation"]
        G1[GPT-4o Synthesis] --> G2[Numerical Validation]
        G2 --> G3[Structured Output + Module Routing]
        G3 --> G4[Final Response]
    end

    subgraph Reports ["Phase 5: Reports"]
        R1[Result Payload] --> R2[ReportLab PDF]
        R2 --> R3[standard / patient_handout / clinic_note]
    end

    Ingestion --> Indexing
    Indexing --> Query
    Query --> Generation
    Generation --> Reports

    style Ingestion fill:#e1f5fe
    style Indexing fill:#e8f5e9
    style Query fill:#fff3e0
    style Generation fill:#f3e5f5
    style Reports fill:#e1f5fe
```
