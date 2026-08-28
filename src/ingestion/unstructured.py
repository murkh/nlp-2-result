"""
Unstructured Ingestion Engine for PDF, DOCX, TXT, and Markdown Documents.
Saves raw files to Blob Storage, performs recursive chunking (800 chars / 150 overlap) with metadata preservation,
computes dense vector embeddings, and enables hybrid RRF search.
"""

import os
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.database.connection import DatabaseManager, get_db_manager
from src.database.models import Dataset, DocumentChunk
from src.ingestion.metadata_extractor import EmbeddingService
from src.storage.blob_store import BlobStorageManager, get_blob_manager


@dataclass
class ParsedSection:
    """A extracted section of text from an unstructured document."""

    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class RecursiveCharacterChunker:
    """
    Splits text into chunks of target size (default 800 chars) with overlap (default 150 chars),
    prioritizing natural semantic boundaries (paragraphs, sentences, words).
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split a single text into overlapping chunks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting implementation."""
        if not text:
            return []

        # Base case: no more separators or text fits
        if len(text) <= self.chunk_size or not separators:
            # Hard chop if still too long
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i : i + self.chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            splits = list(text)
        else:
            splits = text.split(sep)

        chunks = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Overlap handling
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    overlap_text = current_chunk[overlap_start:]
                    current_chunk = f"{overlap_text}{sep}{part}" if overlap_text else part
                else:
                    # Single part is larger than chunk_size, recurse on remaining separators
                    sub_chunks = self._split(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Clean empty chunks
        return [c for c in chunks if c]


class UnstructuredIngestionEngine:
    """
    Engine for loading, chunking, and embedding PDF, DOCX, TXT, and Markdown documents.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        blob_manager: Optional[BlobStorageManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.db_manager = db_manager or get_db_manager()
        self.blob_manager = blob_manager or get_blob_manager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_file(
        self,
        file_input: Union[Path, bytes, str],
        filename: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dataset:
        """
        Full unstructured document ingestion workflow:
        1. Persist raw file in Blob Storage
        2. Parse pages/sections (PDF, DOCX, MD, TXT)
        3. Recursively chunk text (800 chars / 150 overlap) with page and section metadata
        4. Compute 1536-dim vector embeddings
        5. Bulk insert document chunks into document_chunks table
        6. Record in datasets registry
        """
        # 1. Save to blob store
        dataset_id = str(uuid.uuid4())
        d_id, blob_path, file_size_bytes, content_hash = self.blob_manager.save_file(
            file_input=file_input, filename=filename, dataset_id=dataset_id
        )

        file_ext = Path(filename).suffix.lower()
        abs_path = self.blob_manager.get_absolute_path(blob_path)

        # 2. Extract sections & pages
        sections, total_pages = self._extract_sections(abs_path, file_ext)

        # 3. Chunk sections
        chunks_to_embed: List[DocumentChunk] = []
        chunk_index = 0

        for sec in sections:
            text_chunks = self.chunker.split_text(sec.content)
            for text_chunk in text_chunks:
                char_count = len(text_chunk)
                approx_tokens = max(1, char_count // 4)  # ~4 chars per token rule of thumb

                chunk_obj = DocumentChunk(
                    id=str(uuid.uuid4()),
                    dataset_id=d_id,
                    chunk_index=chunk_index,
                    page_number=sec.page_number,
                    section_title=sec.section_title,
                    content=text_chunk,
                    token_count=approx_tokens,
                    char_count=char_count,
                )
                chunks_to_embed.append(chunk_obj)
                chunk_index += 1

        if not chunks_to_embed:
            raise ValueError(f"No text content could be extracted from: {filename}")

        # 4. Generate embeddings
        all_texts = [c.content for c in chunks_to_embed]
        embeddings = self.embedding_service.embed_texts(all_texts)

        for chunk_obj, emb in zip(chunks_to_embed, embeddings):
            chunk_obj.embedding = emb

        # 5. Insert into document_chunks table
        self.db_manager.save_document_chunks(chunks_to_embed)

        # 6. Record dataset in registry
        base_name = Path(filename).stem
        human_name = display_name or base_name.replace("_", " ").title()
        dataset_description = (
            description
            or f"Unstructured document {filename} containing {len(chunks_to_embed)} chunks across {total_pages or 1} pages."
        )

        dataset_record = Dataset(
            id=d_id,
            name=human_name,
            description=dataset_description,
            file_type=file_ext.lstrip("."),
            category="unstructured",
            blob_path=blob_path,
            file_size_bytes=file_size_bytes,
            content_hash=content_hash,
            row_count=len(chunks_to_embed),
            page_count=total_pages,
        )
        self.db_manager.save_dataset(dataset_record)

        return dataset_record

    def _extract_sections(
        self, file_path: Path, file_ext: str
    ) -> Tuple[List[ParsedSection], Optional[int]]:
        """Dispatch document parsing based on file format."""
        if file_ext == ".pdf":
            return self._parse_pdf(file_path)
        elif file_ext in (".docx", ".doc"):
            return self._parse_docx(file_path)
        elif file_ext in (".md", ".markdown"):
            return self._parse_markdown(file_path)
        else:
            return self._parse_plain_text(file_path)

    def _parse_pdf(self, file_path: Path) -> Tuple[List[ParsedSection], int]:
        """Extract text from PDF page by page."""
        sections: List[ParsedSection] = []
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    sections.append(
                        ParsedSection(
                            content=text.strip(),
                            page_number=page_idx + 1,
                            section_title=f"Page {page_idx + 1}",
                        )
                    )
            return sections, total_pages
        except Exception:
            # Basic fallback for binary PDF extraction
            with open(file_path, "rb") as f:
                raw_data = f.read().decode("latin-1", errors="ignore")
            # Extract printable strings
            clean_strings = re.findall(r"\(([\w\s.,!?-]+)\)", raw_data)
            joined = " ".join(clean_strings) if clean_strings else raw_data[:2000]
            return [
                ParsedSection(content=joined, page_number=1, section_title="Extracted Document")
            ], 1

    def _parse_docx(self, file_path: Path) -> Tuple[List[ParsedSection], int]:
        """Extract paragraphs and headers from DOCX."""
        sections: List[ParsedSection] = []
        try:
            import docx

            doc = docx.Document(str(file_path))
            current_heading = "Introduction"
            current_paragraphs: List[str] = []

            for p in doc.paragraphs:
                text = p.text.strip()
                if not text:
                    continue
                if p.style and p.style.name and p.style.name.startswith("Heading"):
                    if current_paragraphs:
                        sections.append(
                            ParsedSection(
                                content="\n\n".join(current_paragraphs),
                                section_title=current_heading,
                            )
                        )
                        current_paragraphs = []
                    current_heading = text
                else:
                    current_paragraphs.append(text)

            if current_paragraphs:
                sections.append(
                    ParsedSection(
                        content="\n\n".join(current_paragraphs),
                        section_title=current_heading,
                    )
                )
            return sections, 1
        except Exception:
            # Fallback for DOCX via direct XML extraction from zip
            try:
                with zipfile.ZipFile(file_path) as z:
                    xml_content = z.read("word/document.xml").decode("utf-8", errors="ignore")
                    text_parts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml_content)
                    joined = " ".join(text_parts)
                    return [ParsedSection(content=joined, section_title="Document")], 1
            except Exception:
                return self._parse_plain_text(file_path)

    def _parse_markdown(self, file_path: Path) -> Tuple[List[ParsedSection], int]:
        """Extract header-organized sections from Markdown."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        sections: List[ParsedSection] = []
        current_title = "Overview"
        current_lines: List[str] = []

        for line in lines:
            if line.startswith("#"):
                if current_lines:
                    text = "".join(current_lines).strip()
                    if text:
                        sections.append(ParsedSection(content=text, section_title=current_title))
                    current_lines = []
                current_title = line.lstrip("#").strip()
            current_lines.append(line)

        if current_lines:
            text = "".join(current_lines).strip()
            if text:
                sections.append(ParsedSection(content=text, section_title=current_title))

        return sections, 1

    def _parse_plain_text(self, file_path: Path) -> Tuple[List[ParsedSection], int]:
        """Extract text from plain text files."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        return [ParsedSection(content=content, section_title="Text Document")], 1

    def search_hybrid(
        self, query: str, top_k: int = 5, dataset_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid dense vector + sparse keyword search with Reciprocal Rank Fusion.
        """
        query_embedding = self.embedding_service.embed_text(query)
        return self.db_manager.hybrid_search_document_chunks(
            query_text=query,
            query_embedding=query_embedding,
            top_k=top_k,
            dataset_id=dataset_id,
        )
