"""
Database models and schemas for Multi-Agent Knowledge Base Q&A Platform.
Provides clean dataclass structures with serialization support.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import uuid


def generate_uuid() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())


def current_utc_time() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class Dataset:
    """Metadata record for an uploaded structured or unstructured dataset."""
    name: str
    file_type: str
    category: str
    blob_path: str
    file_size_bytes: int
    content_hash: str
    id: str = field(default_factory=generate_uuid)
    description: Optional[str] = None
    row_count: Optional[int] = None
    page_count: Optional[int] = None
    created_at: datetime = field(default_factory=current_utc_time)
    updated_at: datetime = field(default_factory=current_utc_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat() if self.created_at else None
        d["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return d


@dataclass
class TableMetadata:
    """Metadata record for a dedicated structured table (used in Stage 1 pruning)."""
    dataset_id: str
    table_name: str
    display_name: str
    description: str
    row_count: int = 0
    column_count: int = 0
    id: str = field(default_factory=generate_uuid)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=current_utc_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat() if self.created_at else None
        return d


@dataclass
class ColumnMetadata:
    """Metadata record for a column in a structured table (used in Stage 2 pruning)."""
    table_id: str
    column_name: str
    data_type: str
    description: str
    id: str = field(default_factory=generate_uuid)
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_target_table: Optional[str] = None
    foreign_target_column: Optional[str] = None
    null_percentage: float = 0.0
    distinct_values_count: int = 0
    sample_values: List[Any] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=current_utc_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat() if self.created_at else None
        d["sample_values"] = json.loads(json.dumps(self.sample_values, default=str))
        return d


@dataclass
class DocumentChunk:
    """Text chunk record from unstructured documents with vector embeddings."""
    dataset_id: str
    chunk_index: int
    content: str
    token_count: int
    char_count: int
    id: str = field(default_factory=generate_uuid)
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=current_utc_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat() if self.created_at else None
        return d


@dataclass
class QueryLog:
    """Telemetry log record for user queries and execution metrics."""
    query_text: str
    engine: str
    status: str
    id: str = field(default_factory=generate_uuid)
    session_id: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    generated_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=current_utc_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat() if self.created_at else None
        return d
