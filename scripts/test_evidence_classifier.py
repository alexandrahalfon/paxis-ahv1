"""
Test Evidence Level Classifier - Read-Only Verification
========================================================

This script reads documents from Qdrant and outputs their evidence level
classifications WITHOUT making any updates to the collection.

Use this to verify the classifier is working correctly before running
the actual update.

Usage:
    # Test with sample documents (no Qdrant connection needed)
    python scripts/test_evidence_classifier.py --samples

    # Test against live Qdrant (read-only, no updates)
    python scripts/test_evidence_classifier.py --qdrant --limit 50

    # Test specific document by ID
    python scripts/test_evidence_classifier.py --qdrant --doc-id "some_doc_id"

    # Output to JSON file
    python scripts/test_evidence_classifier.py --qdrant --limit 100 --output results.json

    # Show detailed pattern matches
    python scripts/test_evidence_classifier.py --qdrant --limit 20 --verbose
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try multiple import locations for the classifier
try:
    from src.utils.evidence_level_classifier_complete import (
        EvidenceLevelClassifier,
        ClassificationResult,
        EVIDENCE_LEVEL_NAMES,
        ComprehensiveEvidenceKeywords,
    )
except ImportError:
    try:
        from evidence_level_classifier_complete import (
            EvidenceLevelClassifier,
            ClassificationResult,
            EVIDENCE_LEVEL_NAMES,
            ComprehensiveEvidenceKeywords,
        )
    except ImportError:
        print("Error: Cannot find evidence_level_classifier_complete.py")
        print("Please save the classifier to one of these locations:")
        print("  - src/utils/evidence_level_classifier_complete.py")
        print("  - evidence_level_classifier_complete.py (project root)")
        sys.exit(1)


class ReadOnlyClassificationTester:
    """
    Test evidence level classification against Qdrant documents
    in READ-ONLY mode - no updates are made.
    """

    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None, collection: str = None):
        self.classifier = EvidenceLevelClassifier()
        self.qdrant = None
        self.collection = collection

        if qdrant_url and qdrant_api_key:
            try:
                from qdrant_client import QdrantClient
                self.qdrant = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=120
                )
                print(f"✓ Connected to Qdrant: {qdrant_url}")
                print(f"✓ Collection: {collection}")
            except Exception as e:
                print(f"✗ Failed to connect to Qdrant: {e}")
                self.qdrant = None

    def test_samples(self) -> Dict:
        """Test classifier against built-in sample documents."""
        print("\n" + "=" * 70)
        print("TESTING CLASSIFIER WITH SAMPLE DOCUMENTS")
        print("=" * 70)

        samples = [
            # Level 1 samples
            {
                "doc_id": "sample_001",
                "doc_meta": {
                    "title": "NCCN Clinical Practice Guidelines in Oncology: Breast Cancer Version 4.2024",
                    "citation": "NCCN Guidelines, 2024",
                },
                "expected_level": 1,
            },
            {
                "doc_id": "sample_002",
                "doc_meta": {
                    "title": "Meta-analysis of hypofractionated whole breast irradiation versus conventional fractionation",
                    "citation": "Int J Radiat Oncol Biol Phys, 2023",
                },
                "expected_level": 1,
            },
            {
                "doc_id": "sample_003",
                "doc_meta": {
                    "title": "Systematic review and meta-analysis of adjuvant radiotherapy for early breast cancer",
                    "citation": "Lancet Oncol, 2022",
                },
                "expected_level": 1,
            },
            # Level 2 samples
            {
                "doc_id": "sample_004",
                "doc_meta": {
                    "title": "FAST-Forward Trial: 5-year efficacy and late normal tissue effects of hypofractionated radiotherapy",
                    "citation": "Lancet, 2020",
                },
                "expected_level": 2,
            },
            {
                "doc_id": "sample_005",
                "doc_meta": {
                    "title": "RTOG 0617: A randomized phase III comparison of standard-dose versus high-dose conformal radiotherapy",
                    "citation": "JAMA Oncol, 2017",
                },
                "expected_level": 2,
            },
            {
                "doc_id": "sample_006",
                "doc_meta": {
                    "title": "Double-blind, placebo-controlled, randomized phase III trial of pembrolizumab",
                    "citation": "N Engl J Med, 2021",
                },
                "expected_level": 2,
            },
            # Level 3 samples
            {
                "doc_id": "sample_007",
                "doc_meta": {
                    "title": "Phase II study of concurrent chemoradiation with weekly cisplatin for locally advanced cervical cancer",
                    "citation": "Gynecol Oncol, 2019",
                },
                "expected_level": 3,
            },
            {
                "doc_id": "sample_008",
                "doc_meta": {
                    "title": "Prospective evaluation of SBRT for oligometastatic prostate cancer",
                    "citation": "Int J Radiat Oncol Biol Phys, 2020",
                },
                "expected_level": 3,
            },
            # Level 4 samples
            {
                "doc_id": "sample_009",
                "doc_meta": {
                    "title": "Retrospective analysis of outcomes after re-irradiation for recurrent head and neck cancer",
                    "citation": "Head Neck, 2021",
                },
                "expected_level": 4,
            },
            {
                "doc_id": "sample_010",
                "doc_meta": {
                    "title": "SEER database analysis of survival trends in pancreatic cancer",
                    "citation": "Cancer, 2022",
                },
                "expected_level": 4,
            },
            {
                "doc_id": "sample_011",
                "doc_meta": {
                    "title": "Single-institution experience with proton therapy for pediatric brain tumors",
                    "citation": "Pediatr Blood Cancer, 2020",
                },
                "expected_level": 4,
            },
            # Level 5 samples
            {
                "doc_id": "sample_012",
                "doc_meta": {
                    "title": "Case report: Radiation recall dermatitis following pembrolizumab administration",
                    "citation": "Case Rep Oncol, 2021",
                },
                "expected_level": 5,
            },
            {
                "doc_id": "sample_013",
                "doc_meta": {
                    "title": "A 72-year-old man with locally advanced rectal cancer: case presentation",
                    "citation": "J Gastrointest Oncol, 2020",
                },
                "expected_level": 5,
            },
            # Level 6 samples
            {
                "doc_id": "sample_014",
                "doc_meta": {
                    "title": "Editorial: The future of artificial intelligence in radiation oncology",
                    "citation": "Int J Radiat Oncol Biol Phys, 2023",
                },
                "expected_level": 6,
            },
            {
                "doc_id": "sample_015",
                "doc_meta": {
                    "title": "Letter to the Editor: Response to hypofractionation guidelines",
                    "citation": "Radiother Oncol, 2022",
                },
                "expected_level": 6,
            },
            # Edge cases
            {
                "doc_id": "sample_016",
                "doc_meta": {
                    "title": "Treatment of breast cancer",
                    "citation": "Unknown Journal",
                },
                "expected_level": 7,  # Too generic
            },
        ]

        results = self._process_documents(samples, show_expected=True)
        return results

    def test_from_qdrant(
        self,
        limit: int = 50,
        doc_id: str = None,
        verbose: bool = False
    ) -> Dict:
        """
        Test classifier against documents from Qdrant (READ-ONLY).
        No updates are made to the collection.
        """
        if not self.qdrant:
            print("✗ Qdrant not connected. Use --samples for offline testing.")
            return {}

        print("\n" + "=" * 70)
        print("TESTING CLASSIFIER AGAINST QDRANT DOCUMENTS (READ-ONLY)")
        print(f"Limit: {limit} documents")
        print("=" * 70)

        documents = []
        seen_doc_ids = set()

        if doc_id:
            # Fetch specific document
            print(f"\nSearching for doc_id: {doc_id}")
            points, _ = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter={
                    "must": [{"key": "doc_id", "match": {"value": doc_id}}]
                },
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                documents.append(payload)
                seen_doc_ids.add(doc_id)
        else:
            # Scroll through collection
            offset = None
            while len(documents) < limit:
                points, next_offset = self.qdrant.scroll(
                    collection_name=self.collection,
                    limit=min(100, limit - len(documents)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload or {}
                    pid = payload.get("doc_id", str(point.id))

                    # Only process unique documents
                    if pid not in seen_doc_ids:
                        seen_doc_ids.add(pid)
                        documents.append(payload)

                        if len(documents) >= limit:
                            break

                offset = next_offset
                if offset is None:
                    break

        print(f"\nFound {len(documents)} unique documents to classify\n")

        results = self._process_documents(documents, verbose=verbose)
        return results

    def _process_documents(
        self,
        documents: List[Dict],
        show_expected: bool = False,
        verbose: bool = False
    ) -> Dict:
        """Process documents and output classifications."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(documents),
            "classifications": [],
            "summary": {
                "by_level": defaultdict(int),
                "by_type": defaultdict(int),
                "correct": 0,
                "incorrect": 0,
            },
        }

        current_level = None

        for doc in documents:
            doc_id = doc.get("doc_id", "unknown")
            doc_meta = doc.get("doc_meta", {})
            title = doc_meta.get("title", "No title")
            citation = doc_meta.get("citation", "")
            expected = doc.get("expected_level")

            # Classify
            classification = self.classifier.classify(doc)

            # Track results
            results["summary"]["by_level"][classification.level] += 1
            results["summary"]["by_type"][classification.evidence_type] += 1

            # Check correctness if expected is provided
            is_correct = None
            if expected is not None:
                is_correct = classification.level == expected
                if is_correct:
                    results["summary"]["correct"] += 1
                else:
                    results["summary"]["incorrect"] += 1

            # Store classification
            entry = {
                "doc_id": doc_id,
                "title": title[:200],
                "citation": citation[:100] if citation else None,
                "classification": classification.to_dict(),
            }
            if expected is not None:
                entry["expected_level"] = expected
                entry["correct"] = is_correct

            results["classifications"].append(entry)

            # Print output
            if show_expected and expected != current_level:
                current_level = expected
                print(f"\n{'─' * 60}")
                print(f"Expected Level {expected}: {EVIDENCE_LEVEL_NAMES.get(expected, 'Unknown')}")
                print(f"{'─' * 60}")

            # Status indicator
            if is_correct is True:
                status = "✓"
            elif is_correct is False:
                status = "✗"
            else:
                status = "•"

            # Level indicator
            level_str = f"L{classification.level}"

            print(f"\n{status} [{level_str}] {title[:70]}...")
            print(f"   Type: {classification.evidence_type}")
            print(f"   Confidence: {classification.confidence:.2f}")

            if classification.matched_patterns and verbose:
                print(f"   Matched: {classification.matched_patterns[:3]}")

            if classification.nci_pdq_code:
                print(f"   NCI Code: {classification.nci_pdq_code}")

            if is_correct is False:
                print(f"   ⚠️  Expected Level {expected}, got Level {classification.level}")

        # Print summary
        self._print_summary(results)

        # Convert defaultdicts to regular dicts for JSON serialization
        results["summary"]["by_level"] = dict(results["summary"]["by_level"])
        results["summary"]["by_type"] = dict(results["summary"]["by_type"])

        return results

    def _print_summary(self, results: Dict):
        """Print classification summary."""
        print("\n" + "=" * 70)
        print("CLASSIFICATION SUMMARY")
        print("=" * 70)

        total = results["total_documents"]
        print(f"\nTotal documents: {total}")

        # Accuracy if we have expected values
        correct = results["summary"]["correct"]
        incorrect = results["summary"]["incorrect"]
        if correct + incorrect > 0:
            accuracy = correct / (correct + incorrect) * 100
            print(f"Accuracy: {correct}/{correct + incorrect} ({accuracy:.1f}%)")

        # Distribution by level
        print("\n--- Distribution by Level ---")
        by_level = results["summary"]["by_level"]
        for level in sorted(by_level.keys()):
            count = by_level[level]
            pct = count / max(total, 1) * 100
            bar = "█" * int(pct / 2)
            level_name = EVIDENCE_LEVEL_NAMES.get(level, "Unknown").split(" - ")[1][:30]
            print(f"  L{level} ({level_name}): {count:>4} ({pct:>5.1f}%) {bar}")

        # Top evidence types
        print("\n--- Top Evidence Types ---")
        by_type = sorted(results["summary"]["by_type"].items(), key=lambda x: -x[1])
        for etype, count in by_type[:10]:
            print(f"  {etype}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Evidence Level Classifier (Read-Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with built-in samples (no Qdrant needed)
  python scripts/test_evidence_classifier.py --samples

  # Test against Qdrant (read-only)
  python scripts/test_evidence_classifier.py --qdrant --limit 50

  # Test specific document
  python scripts/test_evidence_classifier.py --qdrant --doc-id "doi_10.1234_example"

  # Save results to file
  python scripts/test_evidence_classifier.py --qdrant --limit 100 --output results.json

Environment Variables:
  QDRANT_URL        Qdrant server URL
  QDRANT_API_KEY    Qdrant API key
  QDRANT_COLLECTION Collection name
        """
    )

    parser.add_argument("--samples", action="store_true",
                        help="Test with built-in sample documents")
    parser.add_argument("--qdrant", action="store_true",
                        help="Test against Qdrant collection (read-only)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of documents to test (default: 50)")
    parser.add_argument("--doc-id", type=str,
                        help="Test specific document by doc_id")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed pattern matches")
    parser.add_argument("--output", "-o", type=str,
                        help="Save results to JSON file")
    parser.add_argument("--qdrant-url", type=str,
                        help="Qdrant URL (or set QDRANT_URL env var)")
    parser.add_argument("--qdrant-key", type=str,
                        help="Qdrant API key (or set QDRANT_API_KEY env var)")
    parser.add_argument("--collection", type=str,
                        help="Collection name (or set QDRANT_COLLECTION env var)")

    args = parser.parse_args()

    # Get Qdrant config
    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL")
    qdrant_key = args.qdrant_key or os.getenv("QDRANT_API_KEY")
    collection = args.collection or os.getenv("QDRANT_COLLECTION")

    if args.samples:
        tester = ReadOnlyClassificationTester()
        results = tester.test_samples()

    elif args.qdrant:
        if not all([qdrant_url, qdrant_key, collection]):
            print("Error: Qdrant configuration required.")
            print("Set via arguments or environment variables:")
            print("  QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION")
            sys.exit(1)

        tester = ReadOnlyClassificationTester(qdrant_url, qdrant_key, collection)
        results = tester.test_from_qdrant(
            limit=args.limit,
            doc_id=args.doc_id,
            verbose=args.verbose
        )

    else:
        parser.print_help()
        sys.exit(0)

    # Save results if requested
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
