"""
Ingestion & Dataset API Routes.
Provides endpoints for file ingestion (CSV, Parquet, Excel, PDF, DOCX, TXT, MD)
and dataset catalog listing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import DatasetListResponse, IngestResponse
from src.database.connection import DatabaseManager, get_db_manager
from src.ingestion.structured import StructuredIngestionEngine
from src.ingestion.unstructured import UnstructuredIngestionEngine

router = APIRouter(tags=["Ingestion"])


@router.post("/ingest", response_model=IngestResponse)
@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
) -> IngestResponse:
    db_manager = get_db_manager()
    if not isinstance(display_name, str):
        display_name = None
    if not isinstance(description, str):
        description = None

    if hasattr(file, "filename") and hasattr(file, "read"):
        filename = file.filename
        read_res = file.read()
        content_bytes = await read_res if hasattr(read_res, "__await__") else read_res
    elif isinstance(file, tuple) and len(file) == 2:
        filename, content_bytes = file
    elif isinstance(file, (str, Path)):
        p = Path(file)
        filename = p.name
        content_bytes = p.read_bytes()
    else:
        raise ValueError("No file provided for ingestion")

    ext = Path(filename).suffix.lower()
    structured_exts = {".csv", ".tsv", ".parquet", ".pq", ".xlsx", ".xls"}
    unstructured_exts = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}

    if ext in structured_exts:
        struct_engine = StructuredIngestionEngine(db_manager=db_manager)
        ds = struct_engine.ingest_file(
            file_input=content_bytes,
            filename=filename,
            display_name=display_name,
            description=description,
        )
        return IngestResponse(
            dataset_id=ds.id,
            name=ds.name,
            file_type=ds.file_type,
            category=ds.category,
            row_count=ds.row_count,
            page_count=None,
            message=f"Structured dataset '{ds.name}' successfully ingested into dedicated table and vector catalog.",
        )
    elif ext in unstructured_exts:
        unstruct_engine = UnstructuredIngestionEngine(db_manager=db_manager)
        ds = unstruct_engine.ingest_file(
            file_input=content_bytes,
            filename=filename,
            display_name=display_name,
            description=description,
        )
        return IngestResponse(
            dataset_id=ds.id,
            name=ds.name,
            file_type=ds.file_type,
            category=ds.category,
            row_count=ds.row_count,
            page_count=ds.page_count,
            message=f"Unstructured dataset '{ds.name}' successfully chunked ({ds.row_count} chunks) and embedded.",
        )
    else:
        raise ValueError(f"Unsupported file format: {ext}")


@router.get("/datasets", response_model=DatasetListResponse)
@router.get("/ingest/datasets", response_model=DatasetListResponse)
async def list_datasets_endpoint(category: Optional[str] = None) -> DatasetListResponse:
    """List all registered datasets, optionally filtered by category ('structured' / 'unstructured')."""
    db_manager = get_db_manager()
    datasets = db_manager.list_datasets(category=category)
    return DatasetListResponse(datasets=[d.to_dict() for d in datasets])


@router.get("/datasets/{dataset_id}")
async def get_dataset_endpoint(dataset_id: str) -> Dict[str, Any]:
    """Retrieve metadata for a specific dataset ID."""
    db_manager = get_db_manager()
    ds = db_manager.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return ds.to_dict()
