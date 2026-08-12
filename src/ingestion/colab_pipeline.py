#!/usr/bin/env python3
"""
Complete Colab-style ingestion pipeline.

This is a direct port of the Colab ingestion workflow with proper organization
and configuration management.
"""

import json
import uuid
import re
import copy
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple, Optional
from collections import defaultdict
import tiktoken
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

from ..core.config import get_settings
from .doc_id import normalize_doc_id


class ColabIngestionPipeline:
    """
    Complete ingestion pipeline matching the Colab workflow.
    
    Processed docs → keyword-tagged chunks → section-window chunks
    → OpenAI embeddings → Qdrant
    """
    
    def __init__(self, keyword_json_path: Path = None):
        """Initialize the pipeline."""
        self.settings = get_settings()
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
        self.qdrant_client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
            timeout=120
        )
        
        # Configuration
        self.embed_model = self.settings.embed_model
        self.embed_dim = self.settings.embed_dim
        self.max_tokens_per_chunk = self.settings.max_tokens_per_chunk
        self.embed_batch_size = self.settings.embed_batch_size
        self.qdrant_batch_size = self.settings.qdrant_batch_size
        self.collection_name = self.settings.qdrant_collection
        
        # Tokenizer
        self.encoding = tiktoken.get_encoding("o200k_base")
        
        # Keywords
        self.keyword_json_path = keyword_json_path or Path("data/keywords/extractor_keywords.json")
        self.raw_keywords = {}
        self.extractor_keywords_flat = {}
        self.extractor_keywords_by_subcat = {}
        self.term_to_categories = defaultdict(set)
        
        self._load_keywords()
        
        print("✓ Initialized ColabIngestionPipeline")
        print(f"  Embed model: {self.embed_model}")
        print(f"  Collection: {self.collection_name}")
    
    def _load_keywords(self):
        """Load and flatten keywords from JSON file."""
        if not self.keyword_json_path.exists():
            print(f"⚠️ Keyword file not found: {self.keyword_json_path}")
            return
        
        print(f"📚 Loading keywords from {self.keyword_json_path}...")
        
        with self.keyword_json_path.open("r", encoding="utf-8") as f:
            self.raw_keywords = json.load(f)
        
        self.extractor_keywords_flat, self.extractor_keywords_by_subcat = self._flatten_keyword_json(
            self.raw_keywords
        )
        
        # Build term to categories mapping
        for category, terms in self.extractor_keywords_flat.items():
            for term in terms:
                self.term_to_categories[term.lower()].add(category)
        
        print("✅ Loaded keyword categories (TOTAL TERMS):")
        for cat, terms in self.extractor_keywords_flat.items():
            print(f"  - {cat}: {len(terms)} terms")
    
    def _flatten_keyword_json(
        self, 
        obj: Dict[str, Any]
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
        """
        Flatten nested keyword structure.
        
        Returns:
            - flat_by_category: {category: [all terms across all subcategories]}
            - by_category_subcat: {category: {subcat: [terms...]}}
        """
        flat_by_category: Dict[str, List[str]] = {}
        by_category_subcat: Dict[str, Dict[str, List[str]]] = {}
        
        for category, subcats in obj.items():
            if isinstance(subcats, dict):
                by_category_subcat[category] = {}
                all_terms: List[str] = []
                
                for subcat, terms in subcats.items():
                    if not isinstance(terms, list):
                        continue
                    cleaned = [t for t in terms if isinstance(t, str) and t.strip()]
                    by_category_subcat[category][subcat] = cleaned
                    all_terms.extend(cleaned)
                
                seen = set()
                flat_by_category[category] = [
                    t for t in all_terms
                    if not (t.lower() in seen or seen.add(t.lower()))
                ]
                
            elif isinstance(subcats, list):
                cleaned = [t for t in subcats if isinstance(t, str) and t.strip()]
                flat_by_category[category] = cleaned
                by_category_subcat[category] = {"_": cleaned}
            else:
                flat_by_category[category] = []
                by_category_subcat[category] = {}
        
        return flat_by_category, by_category_subcat
    
    def run_complete_pipeline(
        self,
        input_root: Path,
        output_root: Path,
        recreate_collection: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete Colab-style ingestion pipeline.
        
        Args:
            input_root: Root directory with processed documents
            output_root: Output directory for intermediate files
            recreate_collection: Whether to recreate Qdrant collection
            
        Returns:
            Pipeline statistics
        """
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        
        all_chunks_jsonl = output_root / "all_chunks.jsonl"
        section_windows_jsonl = output_root / "all_chunks_section_windows.jsonl"
        
        print("\n" + "="*70)
        print("COLAB-STYLE INGESTION PIPELINE")
        print("="*70)
        print(f"Input: {input_root}")
        print(f"Output: {output_root}")
        print("="*70)
        
        stats = {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "stages": {}
        }
        
        # Step 1: Normalize all documents to chunks JSONL
        print("\n📦 STEP 1: Normalizing documents to keyword-tagged chunks...")
        normalize_stats = self._normalize_all_to_jsonl(input_root, all_chunks_jsonl)
        stats["stages"]["normalize"] = normalize_stats
        
        # Step 2: Create section windows
        print("\n🪟 STEP 2: Creating section windows...")
        window_stats = self._process_chunks_jsonl(all_chunks_jsonl, section_windows_jsonl)
        stats["stages"]["windowing"] = window_stats
        
        # Step 3: Embed and ingest to Qdrant
        print("\n🚀 STEP 3: Embedding and ingesting to Qdrant...")
        ingest_stats = self._ingest_chunks_to_qdrant(
            section_windows_jsonl, 
            recreate_collection=recreate_collection
        )
        stats["stages"]["ingestion"] = ingest_stats

        # Step 4: Build and upsert PTO frames
        print("\n🧬 STEP 4: Building PTO frames...")
        try:
            pto_stats = self._build_and_upsert_pto_frames(section_windows_jsonl)
            stats["stages"]["pto_frames"] = pto_stats
        except Exception as e:
            print(f"⚠ PTO frame building failed (non-fatal): {e}")
            import traceback
            traceback.print_exc()
            stats["stages"]["pto_frames"] = {"error": str(e)}

        print("\n🎉 Done!")
        print(f"- Keyword-tagged base chunks:      {all_chunks_jsonl}")
        print(f"- Section-window chunks (safe):    {section_windows_jsonl}")
        print(f"- Qdrant collection:              {self.collection_name}")
        print(f"- Qdrant endpoint:                {self.settings.qdrant_url}")

        return stats
    
    def _build_and_upsert_pto_frames(self, section_windows_jsonl: Path) -> Dict[str, Any]:
        """Build PTO frames from section windows and upsert to Qdrant.

        Uses LLM extraction (gpt-4o-mini) for comprehensive field extraction,
        then validates extracted fields against source text before upserting
        per-section embeddings to Qdrant.
        """
        from src.ingestion.pto_frame_builder import (
            PTOFrameBuilder,
            LLMPTOExtractor,
            QdrantFrameUpserter,
            validate_extracted_fields,
        )

        # Load chunks grouped by doc_id
        builder = PTOFrameBuilder()
        chunks_by_doc = builder.load_chunks(section_windows_jsonl)

        # Build regex-based frames first (free)
        frames = builder.build_all_frames(chunks_by_doc, min_confidence="low")

        if not frames:
            print("⚠ No PTO frames could be built from the chunks")
            return {"frames_built": 0}

        # LLM extraction pass (gpt-4o-mini, ~$0.01/study)
        llm_extractor = LLMPTOExtractor(
            openai_api_key=self.settings.openai_api_key,
            model="gpt-4o-mini",
        )

        enriched = 0
        validated = 0
        for frame in frames:
            # Get combined source text for this document
            doc_chunks = chunks_by_doc.get(frame.doc_id, [])
            source_text = " ".join(c.get("text", "") for c in doc_chunks)

            # LLM extraction
            if source_text and len(source_text) > 100:
                llm_result = llm_extractor.extract_from_text(source_text)
                if llm_result:
                    llm_extractor.apply_to_frame(frame, llm_result)
                    enriched += 1

                    # Validation pass
                    frame = validate_extracted_fields(frame, source_text)
                    validated += 1

            # Rebuild frame text and section texts after enrichment + validation
            frame.frame_text = builder._build_frame_text(frame)
            builder._build_section_texts(frame)

        llm_stats = llm_extractor.get_stats()
        print(f"✓ LLM enrichment: {enriched}/{len(frames)} frames, "
              f"{llm_stats['calls']} API calls, {llm_stats['total_tokens']} tokens")
        print(f"✓ Validation: {validated} frames validated against source text")

        # Upsert to Qdrant (per-section embeddings)
        upserter = QdrantFrameUpserter(
            qdrant_url=self.settings.qdrant_url,
            qdrant_api_key=self.settings.qdrant_api_key,
            collection_name=self.collection_name,
            openai_api_key=self.settings.openai_api_key,
            embedding_model=self.embed_model,
        )
        upserter.ensure_payload_index()
        upserter.upsert_frames(frames)

        builder.print_summary(frames)

        return {
            "frames_built": len(frames),
            "frames_enriched_llm": enriched,
            "frames_validated": validated,
            "llm_stats": llm_stats,
        }

    def _normalize_all_to_jsonl(self, input_root: Path, out_jsonl: Path) -> Dict[str, Any]:
        """Normalize all processed documents into keyword-tagged chunks JSONL."""
        if out_jsonl.exists():
            print(f"⚠️ Existing chunks file found at: {out_jsonl}")
            print("   Deleting and rebuilding...")
            out_jsonl.unlink()
        
        print("📦 Normalizing ALL processed documents into keyword-tagged chunks JSONL...")
        
        total_docs = 0
        total_chunks = 0
        doc_dirs = list(self._iterate_doc_folders(input_root))
        
        with out_jsonl.open("a", encoding="utf-8") as f_out:
            for doc_dir in tqdm(doc_dirs, desc="Docs processed"):
                category = doc_dir.parent.name
                doc_id_raw = doc_dir.name
                doc_id_norm = normalize_doc_id(doc_id_raw)
                source_doc_dir_name = doc_dir.name
                
                idx_files = list(doc_dir.glob("*_document_index.json"))
                table_files = list(doc_dir.glob("*_tables.json"))
                structured_files = list(doc_dir.glob("*_structured_content.json"))
                
                if not idx_files:
                    continue
                
                # Load document_index
                try:
                    with idx_files[0].open("r", encoding="utf-8") as f:
                        document_index = json.load(f)
                except Exception as e:
                    print(f"❌ Error reading document_index for {doc_dir}: {e}")
                    continue
                
                # Load structured content for doc_meta (citations)
                doc_meta = {}
                if structured_files:
                    try:
                        with structured_files[0].open("r", encoding="utf-8") as f:
                            structured = json.load(f)
                        doc_meta = self._build_doc_meta_from_structured(structured)
                    except Exception as e:
                        print(f"⚠️ Could not read structured_content for {doc_dir}: {e}")
                        doc_meta = {}
                
                total_docs += 1
                
                # Paragraph chunks
                for chunk in self._build_paragraph_chunks(
                    doc_id_norm, doc_id_raw, category, document_index, 
                    doc_meta, source_doc_dir_name
                ):
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_chunks += 1
                
                # Table-row chunks
                if table_files:
                    try:
                        with table_files[0].open("r", encoding="utf-8") as f:
                            tables_obj = json.load(f)
                    except Exception as e:
                        print(f"❌ Error reading tables for {doc_dir}: {e}")
                        tables_obj = None
                    
                    if tables_obj:
                        for chunk in self._build_table_row_chunks(
                            doc_id_norm, doc_id_raw, category, tables_obj,
                            doc_meta, source_doc_dir_name
                        ):
                            f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                            total_chunks += 1
        
        print("✅ Normalization complete.")
        print(f"   Docs processed: {total_docs}")
        print(f"   Chunks written: {total_chunks}")
        print(f"   Output JSONL:   {out_jsonl}")
        
        return {
            "docs_processed": total_docs,
            "chunks_written": total_chunks,
            "output_file": str(out_jsonl)
        }
    
    def _process_chunks_jsonl(
        self, 
        in_jsonl: Path, 
        out_jsonl: Path, 
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Create section_window chunks for paragraph chunks only.
        Table rows stay atomic.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens_per_chunk
            
        if out_jsonl.exists():
            print(f"⚠️ Existing section-window file found at: {out_jsonl}")
            print("   Deleting and rebuilding...")
            out_jsonl.unlink()
        
        # Load all chunks (stream into per-(doc_id, section) buckets for paragraphs)
        paragraph_buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        table_rows: List[Dict[str, Any]] = []
        
        with in_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                if c.get("chunk_type") == "table_row":
                    table_rows.append(c)
                else:
                    key = (c.get("doc_id", ""), c.get("section", "") or "")
                    paragraph_buckets[key].append(c)
        
        # Sort paragraphs inside each bucket by section_paragraph_num
        for key, chunks in paragraph_buckets.items():
            chunks.sort(key=lambda x: (
                x.get("section_paragraph_num") if x.get("section_paragraph_num") is not None else 10**9,
                x.get("chunk_id", "")
            ))
        
        total_out = 0
        
        with out_jsonl.open("a", encoding="utf-8") as f_out:
            # Write paragraph section windows
            for (doc_id, section), chunks in tqdm(paragraph_buckets.items(), desc="Building section windows"):
                window_idx = 0
                current: List[Dict[str, Any]] = []
                cur_tokens = 0
                start_i = 0
                
                def flush(end_i: int):
                    nonlocal window_idx, total_out, current, cur_tokens, start_i
                    if not current:
                        return
                    
                    combined_text = "\n\n".join([x.get("text", "") for x in current]).strip()
                    if not combined_text:
                        current = []
                        cur_tokens = 0
                        return
                    
                    # Build a stable, unique window chunk id
                    sec_key = self._section_key(section)
                    secwin_id = f"{doc_id}__{sec_key}__secwin_{window_idx}"
                    
                    # Use deepcopy to avoid modifying original chunks
                    base = copy.deepcopy(current[0])
                    base["chunk_id"] = secwin_id
                    base["chunk_type"] = "paragraph_window"
                    base["chunk_granularity"] = "section_window"
                    base["section_window_idx"] = window_idx
                    base["window_start_chunk_idx"] = start_i
                    base["window_end_chunk_idx"] = end_i
                    base["text"] = combined_text
                    
                    # Keep keyword metadata: union across window
                    kw_union = set()
                    kw_matches_union: Dict[str, set] = defaultdict(set)
                    for x in current:
                        md = x.get("metadata", {}) or {}
                        for t in (md.get("keywords_flat") or []):
                            kw_union.add(t)
                        km = md.get("keyword_matches", {}) or {}
                        for cat, terms in km.items():
                            for t in terms:
                                kw_matches_union[cat].add(t)
                    
                    base_md = base.setdefault("metadata", {})
                    base_md["keywords_flat"] = sorted(kw_union)
                    base_md["keyword_matches"] = {k: sorted(v) for k, v in kw_matches_union.items()}
                    
                    # Keep lineage
                    source_chunk_ids = [x.get("chunk_id") for x in current]
                    base["source_chunk_ids"] = source_chunk_ids
                    
                    # text_for_embedding remains clean
                    base["text_for_embedding"] = base["text"]
                    
                    f_out.write(json.dumps(base, ensure_ascii=False) + "\n")
                    total_out += 1
                    window_idx += 1
                    current = []
                    cur_tokens = 0
                    start_i = end_i + 1
                
                for i, ch in enumerate(chunks):
                    t = ch.get("text", "")
                    tlen = self._token_len(t)
                    
                    # If adding would exceed, flush current
                    if current and (cur_tokens + tlen) > max_tokens:
                        flush(i - 1)
                    
                    current.append(ch)
                    cur_tokens += tlen
                
                # Flush remaining
                if current:
                    flush(len(chunks) - 1)
            
            # Write table rows as atomic chunks (no windowing)
            for tr in table_rows:
                tr = copy.deepcopy(tr)
                tr["chunk_granularity"] = "table_row_atomic"
                tr["section_window_idx"] = None
                tr["text_for_embedding"] = tr.get("text", "")
                f_out.write(json.dumps(tr, ensure_ascii=False) + "\n")
                total_out += 1
        
        print(f"✅ Section-window creation complete. Chunks written: {total_out}")
        print(f"   Output JSONL: {out_jsonl}")
        
        return {
            "paragraph_buckets": len(paragraph_buckets),
            "table_rows": len(table_rows),
            "total_output_chunks": total_out,
            "output_file": str(out_jsonl)
        }
    
    def _ingest_chunks_to_qdrant(
        self, 
        jsonl_path: Path, 
        recreate_collection: bool = True
    ) -> Dict[str, Any]:
        """Embed chunks and ingest to Qdrant."""
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Section-window JSONL not found at: {jsonl_path}")
        
        print(f"Using section-window chunks file: {jsonl_path}")
        
        # Ensure collection
        if recreate_collection:
            self._ensure_collection()
        
        total_points = 0
        
        for chunk_batch in tqdm(
            self._jsonl_batches(jsonl_path, self.embed_batch_size),
            desc="Embedding + Ingesting (3-large)"
        ):
            # Embed clean text only (never keyword-augmented)
            texts = [c.get("text", "") for c in chunk_batch]
            embeddings = self._embed_texts(texts)
            
            points: List[PointStruct] = []
            for c, emb in zip(chunk_batch, embeddings):
                original_id = c.get("chunk_id")
                payload = c.copy()
                
                # Store traceable id in payload
                payload["original_chunk_id"] = original_id
                
                # Stable UUID point id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(original_id)))
                points.append(PointStruct(id=point_id, vector=emb, payload=payload))
            
            # Batch upsert to Qdrant
            for i in range(0, len(points), self.qdrant_batch_size):
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points[i:i+self.qdrant_batch_size]
                )
            
            total_points += len(points)
        
        print(f"✅ Ingestion complete. Total points inserted: {total_points}")
        
        return {
            "total_points": total_points,
            "collection_name": self.collection_name,
            "source_file": str(jsonl_path)
        }
    
    # Helper methods (direct ports from Colab)
    
    def _iterate_doc_folders(self, root: Path) -> Iterable[Path]:
        """Iterate over document folders."""
        for category_dir in sorted(root.iterdir()):
            if not category_dir.is_dir():
                continue
            for doc_dir in sorted(category_dir.iterdir()):
                if doc_dir.is_dir():
                    yield doc_dir
    
    def _build_doc_meta_from_structured(self, structured_json: Dict[str, Any]) -> Dict[str, Any]:
        """Build document metadata from structured content."""
        dm = structured_json.get("document_metadata") or {}
        info = dm.get("document_info") or {}
        pub = dm.get("publication_info") or {}
        
        title = self._norm_whitespace(info.get("title", "")) or None
        authors = info.get("authors") or []
        author_et_al = self._format_author_et_al(authors)
        year = self._extract_year(pub)
        journal = self._norm_whitespace(pub.get("journal", "")) or None
        doi = self._extract_doi(pub)
        
        parts = []
        if author_et_al:
            parts.append(author_et_al)
        if year:
            parts.append(f"({year})")
        if title:
            parts.append(title)
        if journal:
            parts.append(journal)
        if doi:
            parts.append(f"doi:{doi}")
        
        citation = " ".join(parts).strip()
        
        return {
            "title": title,
            "authors": authors,
            "author_et_al": author_et_al,
            "year": year,
            "journal": journal,
            "doi": doi,
            "citation": citation,
            "document_id": dm.get("document_id"),
        }
    
    def _build_paragraph_chunks(
        self,
        doc_id_norm: str,
        doc_id_raw: str,
        category: str,
        document_index: Dict[str, Any],
        doc_meta: Dict[str, Any],
        source_doc_dir_name: str
    ) -> Iterable[Dict[str, Any]]:
        """Build paragraph chunks from document index."""
        paragraphs = document_index.get("paragraphs", [])
        doc_metadata = document_index.get("metadata", {})
        
        for p in paragraphs:
            text = p.get("text", "")
            if not text or not text.strip():
                continue
            if len(text.strip()) < 20:
                continue
            
            para_id = p.get("id")
            if isinstance(para_id, int):
                chunk_id = f"{doc_id_norm}_p{para_id:04d}"
            else:
                chunk_id = f"{doc_id_norm}_p{para_id}"
            
            chunk = {
                "chunk_id": chunk_id,
                "doc_id": doc_id_norm,
                "doc_id_raw": doc_id_raw,
                "category": category,
                "source_doc_dir_name": source_doc_dir_name,
                "chunk_type": "paragraph",
                "section": p.get("section"),
                "section_paragraph_num": p.get("section_paragraph_num"),
                "paragraph_ref": p.get("paragraph_ref"),
                "text": text,
                "doc_meta": copy.deepcopy(doc_meta),
                "metadata": {
                    "sentences": p.get("sentences", []),
                    "doc_metadata": doc_metadata,
                },
            }
            
            chunk = self._tag_chunk_keywords(chunk, add_to_text_for_embedding=False)
            yield chunk
    
    def _build_table_row_chunks(
        self,
        doc_id_norm: str,
        doc_id_raw: str,
        category: str,
        tables_obj: Dict[str, Any],
        doc_meta: Dict[str, Any],
        source_doc_dir_name: str
    ) -> Iterable[Dict[str, Any]]:
        """Build table row chunks from tables object."""
        tables = tables_obj.get("tables", [])
        
        for t_idx, tbl in enumerate(tables):
            table_number = tbl.get("table_number", f"Table_{t_idx+1}")
            title = tbl.get("title", "")
            headers = tbl.get("headers") or []
            rows = tbl.get("rows", [])
            page = tbl.get("page")
            
            for r_idx, row in enumerate(rows):
                row_values = [str(v) for v in row]
                if headers and len(headers) == len(row_values):
                    cells_text = "; ".join(f"{h}: {v}" for h, v in zip(headers, row_values))
                else:
                    cells_text = "; ".join(row_values)
                
                full_text = f"{table_number} - {title}. {cells_text}".strip()
                if not full_text or len(full_text) < 20:
                    continue
                
                chunk_id = f"{doc_id_norm}_t{t_idx+1:02d}_r{r_idx+1:03d}"
                
                chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id_norm,
                    "doc_id_raw": doc_id_raw,
                    "category": category,
                    "source_doc_dir_name": source_doc_dir_name,
                    "chunk_type": "table_row",
                    "table_number": table_number,
                    "table_title": title,
                    "row_index": r_idx + 1,
                    "text": full_text,
                    "doc_meta": copy.deepcopy(doc_meta),
                    "metadata": {
                        "page": page,
                        "headers": headers,
                        "raw_row": row_values,
                    },
                }
                
                chunk = self._tag_chunk_keywords(chunk, add_to_text_for_embedding=False)
                yield chunk
    
    def _tag_chunk_keywords(self, chunk: Dict, add_to_text_for_embedding: bool = False) -> Dict:
        """Keyword tagging - does NOT modify embedding text by default."""
        text = chunk.get("text", "") or ""
        metadata = chunk.setdefault("metadata", {})
        
        category_to_terms, flat_terms = self._match_keywords_in_text(text)
        metadata["keyword_matches"] = category_to_terms
        metadata["keywords_flat"] = flat_terms
        
        if add_to_text_for_embedding and flat_terms:
            chunk["text_for_embedding"] = text + "\n\nKeywords: " + ", ".join(flat_terms)
        else:
            chunk["text_for_embedding"] = text
        
        return chunk
    
    def _match_keywords_in_text(self, text: str) -> Tuple[Dict[str, List[str]], List[str]]:
        """Matching is simple substring match (case-insensitive)."""
        text_lower = (text or "").lower()
        category_to_terms: Dict[str, List[str]] = defaultdict(list)
        
        for term_lower, categories in self.term_to_categories.items():
            if term_lower and term_lower in text_lower:
                for cat in categories:
                    category_to_terms[cat].append(term_lower)
        
        for cat, terms in category_to_terms.items():
            category_to_terms[cat] = sorted(set(terms))
        
        flat_terms = sorted({t for ts in category_to_terms.values() for t in ts})
        
        return dict(category_to_terms), flat_terms
    
    def _token_len(self, text: str) -> int:
        """Get token length of text."""
        return len(self.encoding.encode(text or ""))
    
    def _section_key(self, section: str) -> str:
        """Turn section into a stable, short key for IDs."""
        s = (section or "NO_SECTION").strip().lower()
        slug = re.sub(r"\s+", "_", s)
        slug = re.sub(r"[^a-z0-9_]+", "", slug)
        slug = slug[:60] if slug else "no_section"
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        return f"{slug}-{h}"
    
    def _ensure_collection(self):
        """Delete + create Qdrant collection with correct vector size and payload indexes."""
        if self.qdrant_client.collection_exists(self.collection_name):
            print(f"🗑️ Deleting existing collection '{self.collection_name}'...")
            self.qdrant_client.delete_collection(self.collection_name)
        
        print(f"🗂️ Creating collection '{self.collection_name}' with dim={self.embed_dim} ...")
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.embed_dim, distance=models.Distance.COSINE),
        )
        print("✅ Collection ready.")
        
        # Payload indexes for filtering/reranking
        index_specs = [
            ("metadata.keywords_flat", models.PayloadSchemaType.KEYWORD),
            ("chunk_id", models.PayloadSchemaType.KEYWORD),
            ("original_chunk_id", models.PayloadSchemaType.KEYWORD),
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("doc_id_raw", models.PayloadSchemaType.KEYWORD),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("chunk_type", models.PayloadSchemaType.KEYWORD),
            ("chunk_granularity", models.PayloadSchemaType.KEYWORD),
            ("section", models.PayloadSchemaType.KEYWORD),
            ("table_number", models.PayloadSchemaType.KEYWORD),
            ("doc_meta.doi", models.PayloadSchemaType.KEYWORD),
            ("doc_meta.author_et_al", models.PayloadSchemaType.KEYWORD),
            ("section_window_idx", models.PayloadSchemaType.INTEGER),
            ("row_index", models.PayloadSchemaType.INTEGER),
            ("doc_meta.year", models.PayloadSchemaType.INTEGER),
        ]
        
        for field_name, field_schema in index_specs:
            try:
                print(f"↓ Creating payload index for '{field_name}' ({field_schema})...")
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as e:
                print(f"⚠️ Index create failed for {field_name}: {e}")
        
        print("✅ Payload indexing attempted for all configured fields.")
    
    def _jsonl_batches(self, path: Path, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
        """Yield batches from JSONL file."""
        batch: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                batch.append(obj)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        safe_texts: List[str] = []
        for t in texts:
            t = t or ""
            token_ids = self.encoding.encode(t)
            if len(token_ids) > 7000:
                token_ids = token_ids[:7000]
                t = self.encoding.decode(token_ids)
            safe_texts.append(t)
        
        resp = self.openai_client.embeddings.create(model=self.embed_model, input=safe_texts)
        return [d.embedding for d in resp.data]
    
    # Utility methods
    
    def _norm_whitespace(self, s: str) -> str:
        """Normalize whitespace."""
        return re.sub(r"\s+", " ", (s or "").strip())
    
    def _extract_year(self, publication_info: Dict[str, Any]) -> Optional[int]:
        """Extract year from publication info."""
        for key in ("publication_date", "online_date", "citation"):
            val = publication_info.get(key)
            if not val:
                continue
            m = re.search(r"\b(19|20)\d{2}\b", str(val))
            if m:
                return int(m.group(0))
        return None
    
    def _extract_doi(self, publication_info: Dict[str, Any]) -> Optional[str]:
        """Extract DOI from publication info."""
        doi = publication_info.get("doi")
        if not doi:
            return None
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(doi), flags=re.IGNORECASE)
        doi = doi.strip()
        return doi or None
    
    def _clean_author_name(self, name: str) -> str:
        """
        Clean an author name by removing credentials and titles.
        
        Examples:
            "Claus Garbe, MD" -> "Claus Garbe"
            "John Smith, PhD, FACS" -> "John Smith"
        """
        if not name:
            return ""
        
        credentials_pattern = re.compile(
            r',?\s*\b('
            r'MD|M\.D\.|DO|D\.O\.|PhD|Ph\.D\.|'
            r'MBBS|MBChB|FRCP|FRCS|FACS|FACP|'
            r'MPH|MS|MSc|MA|MBA|'
            r'RN|NP|PA|PA-C|'
            r'Jr\.?|Sr\.?|III|IV|'
            r'FRCR|FRANZCR|FASTRO|'
            r'Professor|Prof\.?|Dr\.?'
            r')\b\.?',
            re.IGNORECASE
        )
        
        cleaned = credentials_pattern.sub('', name)
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip(' ,.')
        
        return cleaned
    
    def _extract_last_name(self, full_name: str) -> str:
        """Extract the last name from a full name."""
        if not full_name:
            return ""
        
        name = self._clean_author_name(full_name)
        if not name:
            return ""
        
        if ',' in name:
            parts = name.split(',')
            return parts[0].strip()
        
        parts = name.split()
        if len(parts) == 1:
            return parts[0]
        
        prefixes = {'van', 'von', 'de', 'del', 'della', 'di', 'da', 'le', 'la', 'el', 'al'}
        
        for i, part in enumerate(parts):
            if part.lower() in prefixes and i < len(parts) - 1:
                return ' '.join(parts[i:])
        
        return parts[-1]
    
    def _format_author_et_al(self, authors: List[str]) -> str:
        """Format authors as 'LastName et al.' format with proper credential handling."""
        cleaned_authors = [self._clean_author_name(a) for a in (authors or [])]
        cleaned_authors = [a for a in cleaned_authors if a and str(a).strip()]
        
        if not cleaned_authors:
            return "Unknown author"
        if len(cleaned_authors) == 1:
            return cleaned_authors[0]
        
        last_name = self._extract_last_name(cleaned_authors[0])
        
        if not last_name:
            return "Unknown author"
        
        return f"{last_name} et al."


def main():
    """Main function for running the Colab-style pipeline."""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Usage: python -m src.ingestion.colab_pipeline <input_root> <output_root>")
        print("Example: python -m src.ingestion.colab_pipeline /path/to/processed_docs /path/to/output")
        return
    
    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    
    if not input_root.exists():
        print(f"❌ Input directory not found: {input_root}")
        return
    
    # Initialize and run pipeline
    pipeline = ColabIngestionPipeline()
    
    try:
        stats = pipeline.run_complete_pipeline(
            input_root=input_root,
            output_root=output_root,
            recreate_collection=True
        )
        
        print(f"\n✅ Colab-style pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()