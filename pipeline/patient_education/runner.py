from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from pipeline.patient_education.discovery import SourceDiscoverer, bucket_for, coverage_gaps, host_matches
from pipeline.patient_education.sources import SOURCES


logger = logging.getLogger("paxis.pipeline.patient_education")

DEFAULT_OUT_DIR = Path("artifacts/patient_education")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read_json(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _static_default_sources() -> List[dict]:
    """Read DEFAULT_SOURCES without importing patient_db/asyncpg.

    This keeps `check` usable in lightweight CI while still binding the
    pipeline contract to the repository's actual source_registry.py.
    """
    path = Path("src/api/services/evidence/source_registry.py")
    if not path.exists():
        raise RuntimeError(f"Run from the Paxis repository root; missing {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "DEFAULT_SOURCES":
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "DEFAULT_SOURCES" for t in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError("Could not find DEFAULT_SOURCES in source_registry.py")


def assert_repo_compatibility() -> None:
    """Fail before network/database work if discovery config drifts from Paxis registry."""
    default_sources = _static_default_sources()
    registry_keys = {row["source_key"] for row in default_sources}
    unknown = set(SOURCES) - registry_keys
    if unknown:
        raise RuntimeError(
            "patient_education SOURCES contains keys not registered in "
            f"src/api/services/evidence/source_registry.py DEFAULT_SOURCES: {sorted(unknown)}"
        )
    for source_key, scope in SOURCES.items():
        registry_row = next(row for row in default_sources if row["source_key"] == source_key)
        if registry_row["source_type"] != "patient_education":
            raise RuntimeError(f"{source_key} is no longer a patient_education source in Paxis registry")
        if registry_row["domain"] != scope.domain:
            raise RuntimeError(
                f"Domain drift for {source_key}: pipeline={scope.domain!r}, registry={registry_row['domain']!r}"
            )


def discover_sources(
    source_keys: Sequence[str],
    *,
    out_dir: Path,
    max_depth: int,
    max_pages: int,
    delay: float,
) -> Path:
    assert_repo_compatibility()
    discoverer = SourceDiscoverer(
        max_depth=max_depth,
        max_crawl_pages=max_pages,
        delay_seconds=delay,
    )
    rows: List[dict] = []
    try:
        for source_key in source_keys:
            scope = SOURCES[source_key]
            logger.info("Discovering %s (%s)", source_key, scope.domain)
            source_rows = discoverer.discover(scope)
            gaps = coverage_gaps(source_rows, scope)
            if gaps:
                logger.warning("%s coverage gaps: %s", source_key, ", ".join(gaps))
            else:
                logger.info("%s required coverage buckets present", source_key)
            rows.extend(source_rows)
    finally:
        discoverer.close()

    rows.sort(key=lambda r: (r["source_key"], -float(r["importance"]), r["url"]))
    path = out_dir / "discovery_manifest.json"
    _write_json(path, rows)
    _write_csv(out_dir / "discovery_manifest.csv", rows)
    logger.info("Wrote %d discovered URLs to %s", len(rows), path)
    return path


def preflight_manifest(
    manifest_path: Path,
    *,
    out_dir: Path,
    min_chars: int = 500,
    max_per_source: Optional[int] = None,
) -> Path:
    """Use Paxis fetch/extract code without writing Postgres or Qdrant."""
    assert_repo_compatibility()
    from src.api.services.evidence.content_extractor import extract
    from src.api.services.evidence.source_fetcher import fetch_url
    rows = _read_json(manifest_path)
    counts: Dict[str, int] = {}
    out: List[dict] = []

    for row in rows:
        source_key = row["source_key"]
        counts.setdefault(source_key, 0)
        if max_per_source is not None and counts[source_key] >= max_per_source:
            continue
        counts[source_key] += 1

        scope = SOURCES[source_key]
        source_stub = {"source_key": source_key, "domain": scope.domain}
        rec = dict(row)
        try:
            if not host_matches((__import__("urllib.parse", fromlist=["urlsplit"]).urlsplit(rec["url"]).hostname or ""), scope.domain):
                raise ValueError(f"requested URL is outside registered domain {scope.domain}: {rec['url']}")
            fetched = fetch_url(rec["url"])
            if not host_matches((__import__("urllib.parse", fromlist=["urlsplit"]).urlsplit(fetched.final_url).hostname or ""), scope.domain):
                raise ValueError(f"final URL is outside registered domain {scope.domain}: {fetched.final_url}")
            doc = extract(fetched.content, fetched.content_type, source_url=fetched.final_url)
            text = (doc.plain_text or "").strip()
            usable = doc.is_usable() and len(text) >= min_chars
            rec.update({
                "preflight_status": "usable" if usable else "unusable",
                "final_url": fetched.final_url,
                "title": doc.title,
                "content_type": fetched.content_type,
                "chars": len(text),
                "sections": len(doc.sections),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
                "bucket": bucket_for(fetched.final_url, doc.title),
            })
        except Exception as exc:  # per-URL isolation is intentional in an ingestion batch
            rec.update({"preflight_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        out.append(rec)

    # Exact-content dedup across sources. Prefer NCI, then Cancer.Net, then ACS.
    source_priority = {"nci": 0, "cancer_net": 1, "acs": 2}
    usable = [r for r in out if r.get("preflight_status") == "usable" and r.get("content_hash")]
    usable.sort(key=lambda r: (
        r["content_hash"], source_priority.get(r["source_key"], 99), -float(r.get("importance", 0.0))
    ))
    seen_hashes = set()
    selected_keys = set()
    for r in usable:
        h = r["content_hash"]
        if h in seen_hashes:
            r["selection_reason"] = "exact_duplicate"
            continue
        seen_hashes.add(h)
        r["selection_reason"] = "selected"
        selected_keys.add((r["source_key"], r["url"]))

    selected: List[dict] = []
    for r in out:
        key = (r["source_key"], r["url"])
        if key in selected_keys:
            r["selected"] = True
            selected.append(r)
        else:
            r["selected"] = False
            if "selection_reason" not in r:
                r["selection_reason"] = r.get("preflight_status", "unknown")

    _write_json(out_dir / "preflight_all.json", out)
    _write_csv(out_dir / "preflight_all.csv", out)
    selected_path = out_dir / "ingestion_manifest.json"
    _write_json(selected_path, selected)
    _write_csv(out_dir / "ingestion_manifest.csv", selected)
    logger.info("Preflight: %d candidate rows, %d selected", len(out), len(selected))
    return selected_path


async def ingest_manifest(
    manifest_path: Path,
    *,
    out_dir: Path,
    approved_sources: Sequence[str],
    max_per_source: Optional[int] = None,
) -> Path:
    """Only write path: delegates every document to EvidenceIngestionService."""
    assert_repo_compatibility()
    from src.api.services.evidence.source_registry import get_source_registry
    from src.api.services.evidence.evidence_ingestion_service import get_evidence_ingestion_service
    registry = get_source_registry()
    await registry.seed_default_sources()
    service = get_evidence_ingestion_service()

    rows = _read_json(manifest_path)
    counts: Dict[str, int] = {}
    results: List[dict] = []
    checkpoint = out_dir / "ingestion_results.json"

    for row in rows:
        source_key = row["source_key"]
        if source_key not in approved_sources:
            continue
        counts.setdefault(source_key, 0)
        if max_per_source is not None and counts[source_key] >= max_per_source:
            continue
        counts[source_key] += 1

        url = row.get("final_url") or row["url"]
        result_row = {
            "source_key": source_key,
            "url": url,
            "title": row.get("title"),
            "bucket": row.get("bucket"),
        }
        try:
            result = await service.ingest_url(source_key, url)
            result_row.update({
                "status": "unchanged" if result.get("skipped") else "success",
                "document_id": result.get("document_id"),
                "version_id": result.get("version_id"),
                "collection": result.get("collection"),
                "chunks_ingested": result.get("chunks_ingested", 0),
                "reason": result.get("reason"),
            })
        except Exception as exc:  # isolate a bad page from the rest of the run
            result_row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        results.append(result_row)
        _write_json(checkpoint, results)
        _write_csv(out_dir / "ingestion_results.csv", results)

    logger.info("Ingestion complete: %d attempted", len(results))
    return checkpoint


def _source_keys(value: str) -> List[str]:
    keys = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in keys if x not in SOURCES]
    if bad:
        raise argparse.ArgumentTypeError(f"Unknown source(s): {bad}; valid={sorted(SOURCES)}")
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover, audit, and ingest Paxis patient-education evidence sources."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("discover", help="Discover patient-relevant URLs; no DB/Qdrant writes")
    p.add_argument("--sources", type=_source_keys, default=list(SOURCES), help="comma-separated source keys")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-pages", type=int, default=2500)
    p.add_argument("--delay", type=float, default=0.35)

    p = sub.add_parser("preflight", help="Fetch/extract manifest URLs using Paxis code; no DB/Qdrant writes")
    p.add_argument("--manifest", type=Path, default=DEFAULT_OUT_DIR / "discovery_manifest.json")
    p.add_argument("--min-chars", type=int, default=500)
    p.add_argument("--max-per-source", type=int, default=None)

    p = sub.add_parser("ingest", help="Ingest an approved manifest through EvidenceIngestionService")
    p.add_argument("--manifest", type=Path, default=DEFAULT_OUT_DIR / "ingestion_manifest.json")
    p.add_argument(
        "--approve-sources",
        type=_source_keys,
        required=True,
        help="REQUIRED comma-separated source keys explicitly approved for this write run",
    )
    p.add_argument("--max-per-source", type=int, default=50)

    sub.add_parser("check", help="Validate this pipeline against the current Paxis source registry")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "check":
        assert_repo_compatibility()
        logger.info("Patient-education pipeline is compatible with current Paxis source registry")
        return 0
    if args.command == "discover":
        discover_sources(
            args.sources,
            out_dir=args.out_dir,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            delay=args.delay,
        )
        return 0
    if args.command == "preflight":
        preflight_manifest(
            args.manifest,
            out_dir=args.out_dir,
            min_chars=args.min_chars,
            max_per_source=args.max_per_source,
        )
        return 0
    if args.command == "ingest":
        asyncio.run(ingest_manifest(
            args.manifest,
            out_dir=args.out_dir,
            approved_sources=args.approve_sources,
            max_per_source=args.max_per_source,
        ))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
