"""
Structured Ingestion Engine for CSV, Parquet, and Excel Datasets.
Saves raw files to Blob Storage, creates dedicated PostgreSQL dynamic tables (Strategy A),
and populates TableMetadata & ColumnMetadata for two-stage schema pruning.
"""

import csv
import io
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from src.database.connection import DatabaseManager, get_db_manager
from src.database.models import ColumnMetadata, Dataset, TableMetadata
from src.ingestion.metadata_extractor import MetadataExtractor
from src.storage.blob_store import BlobStorageManager, get_blob_manager


def sanitize_identifier(name: str, max_length: int = 63) -> str:
    """
    Sanitize an identifier (table or column name) to be PostgreSQL-safe.
    Lowercase, replaces non-alphanumeric chars with underscore, ensures valid leading char.
    """
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "col"
    if clean[0].isdigit():
        clean = f"col_{clean}"
    return clean[:max_length]


def sanitize_table_name(dataset_id: str, raw_name: str) -> str:
    """
    Construct dedicated table name format: tbl_{dataset_id_prefix}_{sanitized_name}
    PostgreSQL limits table names to 63 characters.
    """
    id_prefix = dataset_id.replace("-", "")[:8]
    clean_base = sanitize_identifier(raw_name, max_length=45)
    full_name = f"tbl_{id_prefix}_{clean_base}"
    return full_name[:63]


class StructuredIngestionEngine:
    """
    Engine for loading CSV, Parquet, and Excel files into Blob Storage,
    creating dedicated PostgreSQL tables, and indexing metadata embeddings.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        blob_manager: Optional[BlobStorageManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ):
        self.db_manager = db_manager or get_db_manager()
        self.blob_manager = blob_manager or get_blob_manager()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()

    def ingest_file(
        self,
        file_input: Union[Path, bytes, str],
        filename: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dataset:
        """
        Full structured ingestion workflow:
        1. Save raw file to Blob Storage
        2. Parse columns and rows (CSV, Parquet, or Excel)
        3. Create dedicated SQL table (tbl_<id>_<name>)
        4. Bulk insert rows
        5. Extract and embed table & column metadata
        6. Record in datasets registry
        """
        # 1. Save to blob store
        dataset_id = str(uuid.uuid4())
        d_id, blob_path, file_size_bytes, content_hash = self.blob_manager.save_file(
            file_input=file_input, filename=filename, dataset_id=dataset_id
        )

        # 2. Parse file content into columns and rows
        file_ext = Path(filename).suffix.lower()
        abs_path = self.blob_manager.get_absolute_path(blob_path)

        columns, rows = self._parse_file(abs_path, file_ext)
        if not columns:
            raise ValueError(f"No columns found in structured file: {filename}")

        clean_columns = [sanitize_identifier(col) for col in columns]
        # Ensure unique column names
        unique_cols: List[str] = []
        col_seen: Dict[str, int] = {}
        for c in clean_columns:
            if c in col_seen:
                col_seen[c] += 1
                unique_cols.append(f"{c}_{col_seen[c]}")
            else:
                col_seen[c] = 1
                unique_cols.append(c)
        clean_columns = unique_cols

        # 3. Create dedicated table name
        base_name = Path(filename).stem
        table_name = sanitize_table_name(d_id, base_name)
        human_display_name = display_name or base_name.replace("_", " ").title()

        # 4. Extract profile and metadata
        table_meta, col_metas = self.metadata_extractor.extract_and_embed_metadata(
            dataset_id=d_id,
            table_name=table_name,
            column_names=clean_columns,
            rows=rows,
            display_name=human_display_name,
            table_description=description,
        )

        # 5. Create dedicated table in DB
        col_defs = [(c.column_name, c.data_type) for c in col_metas]
        pk_col = next((c.column_name for c in col_metas if c.is_primary_key), None)
        self.db_manager.create_dedicated_table(table_name=table_name, column_defs=col_defs, pkey_col=pk_col)

        # 6. Bulk insert rows into dedicated table
        self.db_manager.insert_table_rows(table_name=table_name, columns=clean_columns, rows=rows)

        # 7. Save metadata into catalog
        self.db_manager.save_table_metadata(table_meta)
        for col_meta in col_metas:
            self.db_manager.save_column_metadata(col_meta)

        # 8. Record in datasets registry
        dataset_record = Dataset(
            id=d_id,
            name=human_display_name,
            description=table_meta.description,
            file_type=file_ext.lstrip("."),
            category="structured",
            blob_path=blob_path,
            file_size_bytes=file_size_bytes,
            content_hash=content_hash,
            row_count=len(rows),
            page_count=None,
        )
        self.db_manager.save_dataset(dataset_record)

        return dataset_record

    def _parse_file(self, file_path: Path, file_ext: str) -> Tuple[List[str], List[List[Any]]]:
        """Dispatch parser based on file extension."""
        if file_ext in (".csv", ".tsv", ".txt"):
            delimiter = "\t" if file_ext == ".tsv" else ","
            return self._parse_csv(file_path, delimiter=delimiter)
        elif file_ext in (".parquet", ".pq"):
            return self._parse_parquet(file_path)
        elif file_ext in (".xlsx", ".xls"):
            return self._parse_excel(file_path)
        else:
            # Attempt CSV parsing as fallback
            return self._parse_csv(file_path)

    def _parse_csv(self, file_path: Path, delimiter: str = ",") -> Tuple[List[str], List[List[Any]]]:
        """Parse CSV/TSV file with dialect sniffing and type conversion."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            if not header:
                return [], []

            columns = [col.strip() for col in header]
            rows: List[List[Any]] = []

            for row in reader:
                if not row or all(str(cell).strip() == "" for cell in row):
                    continue
                # Normalize row length
                typed_row = []
                for cell in row:
                    typed_row.append(self._convert_cell_value(cell))
                # Pad if row is shorter than header
                while len(typed_row) < len(columns):
                    typed_row.append(None)
                rows.append(typed_row[: len(columns)])

            return columns, rows

    def _parse_parquet(self, file_path: Path) -> Tuple[List[str], List[List[Any]]]:
        """Parse Parquet file using pyarrow, pandas, or fastparquet."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            columns = table.column_names
            rows = table.to_pylist()
            # Convert list of dicts to list of lists
            row_lists = [[r.get(c) for c in columns] for r in rows]
            return columns, row_lists
        except ImportError:
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                columns = list(df.columns)
                rows = df.values.tolist()
                return columns, rows
            except ImportError:
                # If pyarrow/pandas not installed, read as binary or raise informative error
                raise RuntimeError("Parquet parsing requires pyarrow or pandas to be installed.")

    def _parse_excel(self, file_path: Path) -> Tuple[List[str], List[List[Any]]]:
        """Parse Excel (.xlsx / .xls) workbook sheet."""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            sheet = wb.active
            rows_iter = sheet.iter_rows(values_only=True)
            header_row = next(rows_iter, None)
            if not header_row:
                return [], []

            columns = [str(c).strip() if c is not None else f"col_{i+1}" for i, c in enumerate(header_row)]
            rows = []
            for r in rows_iter:
                if not r or all(c is None for c in r):
                    continue
                rows.append([self._convert_cell_value(c) for c in r[:len(columns)]])
            return columns, rows
        except ImportError:
            try:
                import pandas as pd
                df = pd.read_excel(file_path)
                columns = list(df.columns)
                rows = df.values.tolist()
                return columns, rows
            except ImportError:
                raise RuntimeError("Excel parsing requires openpyxl or pandas to be installed.")

    def _convert_cell_value(self, val: Any) -> Any:
        """Convert string cell values to typed integers, floats, or booleans where appropriate."""
        if val is None:
            return None
        if isinstance(val, (int, float, bool)):
            return val

        val_str = str(val).strip()
        if val_str == "" or val_str.lower() in ("null", "none", "nan", "na"):
            return None
        if val_str.lower() in ("true", "t"):
            return True
        if val_str.lower() in ("false", "f"):
            return False

        # Check integer
        if re.match(r"^-?\d+$", val_str):
            try:
                return int(val_str)
            except ValueError:
                pass

        # Check float
        if re.match(r"^-?\d+\.\d+$", val_str):
            try:
                return float(val_str)
            except ValueError:
                pass

        return val_str
