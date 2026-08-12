"""
Document chunking and windowing for vector ingestion.

Converts processed documents into chunks suitable for embedding and vector storage.
Supports both paragraph-level chunks and section-window chunks.
"""

import json
import hashlib
import re
import copy
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable
from collections import defaultdict
from datetime import datetime
import tiktoken

from ..core.config import get_settings


class ChunkProcessor:
    """Process documents into chunks for vector ingestion."""
    
    def __init__(self):
        """Initialize chunk processor."""
        self.settings = get_settings()
        self.encoding = tiktoken.get_encoding("o200k_base")
        
    def process_documents_to_chunks(
        self, 
        input_root: Path, 
        output_jsonl: Path,
        include_references: bool = False
    ) -> Dict[str, int]:
        """
        Process all documents in input_root into normalized chunks.
        
        Args:
            input_root: Root directory containing processed documents
            output_jsonl: Output JSONL file path
            include_references: Whether to include reference sections
            
        Returns:
            Dict with processing statistics
        """
        if output_jsonl.exists():
            print(f"⚠️ Existing chunks file found at: {output_jsonl}")
            print(" Deleting and rebuilding...")
            output_jsonl.unlink()
        
        print("📦 Normalizing ALL processed documents into chunks JSONL...")
        
        total_docs = 0
        total_chunks = 0
        doc_dirs = list(self._iterate_doc_folders(input_root))
        
        with output_jsonl.open("a", encoding="utf-8") as f_out:
            for doc_dir in doc_dirs:
                category = doc_dir.parent.name
                doc_id_raw = doc_dir.name
                doc_id_norm = self._normalize_doc_id(doc_id_raw)
                source_doc_dir_name = doc_dir.name
                
                # Find required files
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
                
                # Process paragraph chunks
                for chunk in self._build_paragraph_chunks(
                    doc_id_norm, doc_id_raw, category, document_index, 
                    doc_meta, source_doc_dir_name
                ):
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_chunks += 1
                
                # Process table-row chunks
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
        
        stats = {
            "docs_processed": total_docs,
            "chunks_created": total_chunks,
            "output_file": str(output_jsonl)
        }
        
        print("✅ Normalization complete.")
        print(f" Docs processed: {total_docs}")
        print(f" Chunks written: {total_chunks}")
        print(f" Output JSONL: {output_jsonl}")
        
        return stats
    
    def create_section_windows(
        self, 
        input_jsonl: Path, 
        output_jsonl: Path,
        max_tokens: int = None
    ) -> Dict[str, int]:
        """
        Create section-window chunks from paragraph chunks.
        
        Args:
            input_jsonl: Input JSONL with paragraph chunks
            output_jsonl: Output JSONL with section windows
            max_tokens: Maximum tokens per window chunk
            
        Returns:
            Dict with processing statistics
        """
        if max_tokens is None:
            max_tokens = self.settings.max_tokens_per_chunk
            
        if output_jsonl.exists():
            print(f"⚠️ Existing section-window file found at: {output_jsonl}")
            print(" Deleting and rebuilding...")
            output_jsonl.unlink()
        
        # Load all chunks into buckets
        paragraph_buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        table_rows: List[Dict[str, Any]] = []
        
        with input_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                if c.get("chunk_type") == "table_row":
                    table_rows.append(c)
                else:
                    key = (c.get("doc_id", ""), c.get("section", "") or "")
                    paragraph_buckets[key].append(c)
        
        # Sort paragraphs within each bucket
        for key, chunks in paragraph_buckets.items():
            chunks.sort(key=lambda x: (
                x.get("section_paragraph_num") if x.get("section_paragraph_num") is not None else 10**9,
                x.get("chunk_id", "")
            ))
        
        total_out = 0
        
        with output_jsonl.open("a", encoding="utf-8") as f_out:
            # Create paragraph section windows
            for (doc_id, section), chunks in paragraph_buckets.items():
                window_chunks = self._create_windows_for_section(
                    doc_id, section, chunks, max_tokens
                )
                for chunk in window_chunks:
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_out += 1
            
            # Write table rows as atomic chunks
            for tr in table_rows:
                tr = copy.deepcopy(tr)
                tr["chunk_granularity"] = "table_row_atomic"
                tr["section_window_idx"] = None
                tr["text_for_embedding"] = tr.get("text", "")
                f_out.write(json.dumps(tr, ensure_ascii=False) + "\n")
                total_out += 1
        
        stats = {
            "paragraph_buckets": len(paragraph_buckets),
            "table_rows": len(table_rows),
            "total_output_chunks": total_out,
            "output_file": str(output_jsonl)
        }
        
        print(f"✅ Section-window creation complete. Chunks written: {total_out}")
        print(f" Output JSONL: {output_jsonl}")
        
        return stats
    
    def _iterate_doc_folders(self, root: Path) -> Iterable[Path]:
        """Iterate over document folders in the root directory."""
        for category_dir in sorted(root.iterdir()):
            if not category_dir.is_dir():
                continue
            for doc_dir in sorted(category_dir.iterdir()):
                if doc_dir.is_dir():
                    yield doc_dir
    
    def _normalize_doc_id(self, raw_name: str) -> str:
        """Create a filesystem-safe, collision-resistant doc_id."""
        raw_name = raw_name or "unknown"
        # Remove problematic characters but keep some readability
        clean = re.sub(r'[^\w\-.]', '_', raw_name.strip())
        clean = re.sub(r'_+', '_', clean).strip('_')  # collapse multiple underscores
        clean = clean[:50]  # truncate for readability
        
        # Add hash of ORIGINAL name to guarantee uniqueness
        h = hashlib.md5(raw_name.encode('utf-8')).hexdigest()[:8]
        return f"{clean}_{h}"
    
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
                
                yield chunk
    
    def _create_windows_for_section(
        self,
        doc_id: str,
        section: str,
        chunks: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Create windowed chunks for a section."""
        windows = []
        window_idx = 0
        current: List[Dict[str, Any]] = []
        cur_tokens = 0
        start_i = 0
        
        def flush(end_i: int):
            nonlocal window_idx, current, cur_tokens, start_i
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
            
            windows.append(base)
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
        
        return windows
    
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
    
    def _norm_whitespace(self, s: str) -> str:
        """Normalize whitespace in string."""
        return re.sub(r"\s+", " ", (s or "").strip())
    
    def _clean_author_name(self, name: str) -> str:
        """
        Clean an author name by removing credentials and titles.
        
        Examples:
            "Claus Garbe, MD" -> "Claus Garbe"
            "John Smith, PhD, FACS" -> "John Smith"
            "Dr. Jane Doe" -> "Jane Doe"
        """
        if not name:
            return ""
        
        # Pattern to match common medical credentials/titles
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
        
        # Remove credentials
        cleaned = credentials_pattern.sub('', name)
        
        # Remove extra commas and whitespace
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip(' ,.')
        
        return cleaned
    
    def _extract_last_name(self, full_name: str) -> str:
        """
        Extract the last name from a full name, handling various formats.
        
        Examples:
            "John Smith" -> "Smith"
            "Smith, John" -> "Smith"
            "John van der Berg" -> "van der Berg"
        """
        if not full_name:
            return ""
        
        # Clean the name first
        name = self._clean_author_name(full_name)
        
        if not name:
            return ""
        
        # Check if it's "LastName, FirstName" format
        if ',' in name:
            parts = name.split(',')
            return parts[0].strip()
        
        # Split by spaces
        parts = name.split()
        
        if len(parts) == 1:
            return parts[0]
        
        # Handle prefixes like "van", "de", "von", etc.
        prefixes = {'van', 'von', 'de', 'del', 'della', 'di', 'da', 'le', 'la', 'el', 'al'}
        
        # Find where the last name starts
        last_name_parts = []
        for i, part in enumerate(parts):
            if part.lower() in prefixes and i < len(parts) - 1:
                last_name_parts = parts[i:]
                break
        
        if last_name_parts:
            return ' '.join(last_name_parts)
        
        # Default: last word is the last name
        return parts[-1]
    
    def _format_author_et_al(self, authors: List[str]) -> str:
        """Format authors as 'LastName et al.' format with proper credential handling."""
        # Clean all author names
        cleaned_authors = [self._clean_author_name(a) for a in (authors or [])]
        cleaned_authors = [a for a in cleaned_authors if a and str(a).strip()]
        
        if not cleaned_authors:
            return "Unknown author"
        if len(cleaned_authors) == 1:
            return cleaned_authors[0]
        
        # Get last name of first author
        last_name = self._extract_last_name(cleaned_authors[0])
        
        if not last_name:
            return "Unknown author"
        
        return f"{last_name} et al."
    
    def _extract_year(self, publication_info: Dict[str, Any]) -> int:
        """Extract publication year from publication info."""
        for key in ("publication_date", "online_date", "citation"):
            val = publication_info.get(key)
            if not val:
                continue
            m = re.search(r"\b(19|20)\d{2}\b", str(val))
            if m:
                return int(m.group(0))
        return None
    
    def _extract_doi(self, publication_info: Dict[str, Any]) -> str:
        """Extract DOI from publication info."""
        doi = publication_info.get("doi")
        if not doi:
            return None
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(doi), flags=re.IGNORECASE)
        doi = doi.strip()
        return doi or None