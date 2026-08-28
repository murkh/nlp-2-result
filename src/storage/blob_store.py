"""
Blob Storage Manager for Multi-Agent Knowledge Base Q&A Platform.
Persists raw uploaded structured (CSV, Parquet, Excel) and unstructured (PDF, DOCX, TXT, MD) files.
"""

import hashlib
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.config import get_settings


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and invalid filesystem characters."""
    # Extract only the base name (no directories)
    base_name = os.path.basename(filename)
    # Replace spaces and special characters except dots, hyphens, and underscores
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name)
    # Avoid empty names or leading dot
    if not clean or clean.startswith("."):
        clean = f"file_{clean}"
    return clean[:255]


def sanitize_dataset_id(dataset_id: Optional[str]) -> str:
    """Sanitize dataset_id to prevent path traversal and invalid characters."""
    if not dataset_id:
        return str(uuid.uuid4())
    clean = re.sub(r"[^a-zA-Z0-9_-]", "", str(dataset_id))
    if not clean:
        return str(uuid.uuid4())
    return clean[:255]


def compute_sha256(content: bytes) -> str:
    """Compute SHA-256 hexadecimal hash string for binary content."""
    return hashlib.sha256(content).hexdigest()


class BlobStorageManager:
    """
    Manages filesystem-based blob storage organized by dataset UUIDs.
    Layout: <base_path>/<dataset_id>/<sanitized_filename>
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = Path(base_path) if base_path else get_settings().blob_storage_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_file(
        self,
        file_input: Union[bytes, str, Path],
        filename: str,
        dataset_id: Optional[str] = None,
    ) -> Tuple[str, str, int, str]:
        """
        Persist a file into blob storage.

        Args:
            file_input: Binary bytes, string content, or existing Path on disk.
            filename: Target file name (e.g., 'orders.csv', 'handbook.pdf').
            dataset_id: Optional dataset UUID. If omitted, a new UUID is generated.

        Returns:
            Tuple of (dataset_id, relative_blob_path, file_size_bytes, sha256_hash)
        """
        if isinstance(file_input, Path):
            with open(file_input, "rb") as f:
                content_bytes = f.read()
        elif isinstance(file_input, str):
            content_bytes = file_input.encode("utf-8")
        elif isinstance(file_input, bytes):
            content_bytes = file_input
        else:
            raise TypeError(f"Unsupported file_input type: {type(file_input)}")

        content_hash = compute_sha256(content_bytes)
        file_size_bytes = len(content_bytes)

        d_id = sanitize_dataset_id(dataset_id)
        safe_name = sanitize_filename(filename)

        resolved_base = self.base_path.resolve()
        dataset_dir = (self.base_path / d_id).resolve()
        try:
            if not dataset_dir.is_relative_to(resolved_base):
                raise ValueError(
                    f"Access denied: Path traversal attempt in dataset_id ({dataset_id})"
                )
        except AttributeError:
            if os.path.commonpath([str(dataset_dir), str(resolved_base)]) != str(resolved_base):
                raise ValueError(
                    f"Access denied: Path traversal attempt in dataset_id ({dataset_id})"
                )

        dataset_dir.mkdir(parents=True, exist_ok=True)

        target_path = (dataset_dir / safe_name).resolve()
        try:
            if not target_path.is_relative_to(resolved_base):
                raise ValueError(
                    f"Access denied: Path traversal attempt in target path ({filename})"
                )
        except AttributeError:
            if os.path.commonpath([str(target_path), str(resolved_base)]) != str(resolved_base):
                raise ValueError(
                    f"Access denied: Path traversal attempt in target path ({filename})"
                )

        with open(target_path, "wb") as f:
            f.write(content_bytes)

        relative_blob_path = f"{d_id}/{safe_name}"
        return d_id, relative_blob_path, file_size_bytes, content_hash

    def get_absolute_path(self, blob_path: str) -> Path:
        """Resolve a relative blob path (e.g. 'uuid/filename.csv') to absolute Path."""
        resolved_base = self.base_path.resolve()
        full_path = (self.base_path / blob_path).resolve()
        # Security check: Ensure resolved path is strictly within base_path
        try:
            is_rel = full_path.is_relative_to(resolved_base)
        except AttributeError:
            is_rel = os.path.commonpath([str(full_path), str(resolved_base)]) == str(resolved_base)
        if not is_rel:
            raise ValueError(f"Access denied: Path traversal attempt detected ({blob_path})")
        return full_path

    def read_bytes(self, blob_path: str) -> bytes:
        """Read binary contents of a stored blob file."""
        abs_path = self.get_absolute_path(blob_path)
        if not abs_path.is_file():
            raise FileNotFoundError(f"Blob file not found: {blob_path}")
        with open(abs_path, "rb") as f:
            return f.read()

    def read_text(self, blob_path: str, encoding: str = "utf-8") -> str:
        """Read text contents of a stored blob file."""
        raw = self.read_bytes(blob_path)
        return raw.decode(encoding, errors="replace")

    def exists(self, blob_path: str) -> bool:
        """Check if blob file exists."""
        try:
            abs_path = self.get_absolute_path(blob_path)
            return abs_path.is_file()
        except ValueError:
            return False

    def delete_file(self, blob_path: str) -> bool:
        """Delete a blob file from storage."""
        try:
            abs_path = self.get_absolute_path(blob_path)
            if abs_path.is_file():
                abs_path.unlink()
                # If directory is now empty and strictly inside base_path, remove it
                parent_dir = abs_path.parent.resolve()
                resolved_base = self.base_path.resolve()
                try:
                    is_rel = parent_dir.is_relative_to(resolved_base)
                except AttributeError:
                    is_rel = os.path.commonpath([str(parent_dir), str(resolved_base)]) == str(
                        resolved_base
                    )
                if is_rel and parent_dir != resolved_base and not any(parent_dir.iterdir()):
                    shutil.rmtree(parent_dir, ignore_errors=True)
                return True
            return False
        except Exception:
            return False

    def list_dataset_files(self, dataset_id: str) -> List[str]:
        """List all relative blob paths for a given dataset ID."""
        cleaned_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(dataset_id))
        if not cleaned_id:
            return []
        resolved_base = self.base_path.resolve()
        dataset_dir = (self.base_path / cleaned_id).resolve()
        try:
            if not dataset_dir.is_relative_to(resolved_base):
                return []
        except AttributeError:
            if os.path.commonpath([str(dataset_dir), str(resolved_base)]) != str(resolved_base):
                return []
        if not dataset_dir.is_dir():
            return []
        return [f"{cleaned_id}/{f.name}" for f in dataset_dir.iterdir() if f.is_file()]


# Singleton helper
_blob_manager_instance: Optional[BlobStorageManager] = None


def get_blob_manager(base_path: Optional[Path] = None) -> BlobStorageManager:
    """Get or create singleton BlobStorageManager."""
    global _blob_manager_instance
    if _blob_manager_instance is None or base_path is not None:
        _blob_manager_instance = BlobStorageManager(base_path=base_path)
    return _blob_manager_instance
