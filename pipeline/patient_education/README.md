# Paxis patient-education ingestion pipeline

This is the repository-native replacement for the exploratory Colab notebook.
The notebook should be treated only as an interactive wrapper around this code.

## Architectural rule

**Discovery lives here. Evidence storage does not.**

All writes go through the existing Paxis service:

`src.api.services.evidence.evidence_ingestion_service.EvidenceIngestionService`

Therefore this pipeline inherits the application's existing:

- evidence source registry / domain enforcement;
- HTML/PDF extraction;
- document and section-level classification;
- section-aware chunking;
- OpenAI embedding configuration;
- Qdrant collection routing;
- deterministic document/version/point IDs;
- Postgres evidence version/chunk registry;
- idempotent re-ingestion and safe version switching.

## Files

- `sources.py` — crawl/discovery scopes only. `source_key` and domain must match the core Paxis source registry.
- `discovery.py` — robots-aware sitemap discovery, bounded same-domain crawling, URL normalization and coverage buckets.
- `runner.py` — CLI orchestration: `check`, `discover`, `preflight`, `ingest`.

## Run from the repository root

```bash
python -m pipeline.patient_education.runner check
```

Discover NCI + Cancer.Net + ACS without writing databases:

```bash
python -m pipeline.patient_education.runner discover \
  --sources nci,cancer_net,acs
```

Use Paxis's real fetch/extraction code to validate the resulting URLs, still without DB/Qdrant writes:

```bash
python -m pipeline.patient_education.runner preflight
```

Inspect:

```text
artifacts/patient_education/discovery_manifest.csv
artifacts/patient_education/preflight_all.csv
artifacts/patient_education/ingestion_manifest.csv
```

First controlled write, NCI only, capped at 50 pages:

```bash
python -m pipeline.patient_education.runner ingest \
  --approve-sources nci \
  --max-per-source 50
```

After validation, ingest the full approved NCI manifest:

```bash
python -m pipeline.patient_education.runner ingest \
  --approve-sources nci \
  --max-per-source 1000000
```

Then explicitly approve other sources only after content-use/acquisition review:

```bash
python -m pipeline.patient_education.runner ingest \
  --approve-sources cancer_net,acs \
  --max-per-source 50
```

## Safety properties

- `discover` does not require Paxis DB/Qdrant and performs no writes.
- `preflight` uses Paxis's `fetch_url()` and `extract()` but performs no DB/Qdrant writes.
- `ingest` requires an explicit `--approve-sources ...` argument.
- `ingest` never calls Qdrant or Postgres directly; all writes go through `EvidenceIngestionService`.
- A single failed URL is checkpointed and does not kill the entire run.
- Re-running unchanged content is idempotent because the core service owns content hashes/version IDs.
- The `check` command fails if source keys/domains drift from `DEFAULT_SOURCES`.

## Recommended CI smoke test

Run this in CI because it requires no internet or database:

```bash
python -m pipeline.patient_education.runner check
```

Add the included unit test for URL scope behavior to your normal pytest suite.
