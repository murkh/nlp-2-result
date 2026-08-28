"""
Test fixtures and configuration for Milestone 1 unit and integration tests.
Supports both pytest test runner and standard unittest test execution.
"""

import os
from pathlib import Path
import shutil
import tempfile
from typing import Generator

from src.config import Settings
from src.database.connection import DatabaseManager
from src.ingestion.metadata_extractor import EmbeddingService, MetadataExtractor
from src.ingestion.structured import StructuredIngestionEngine
from src.ingestion.unstructured import UnstructuredIngestionEngine
from src.pruning.schema_pruner import TwoStageSchemaPruner
from src.storage.blob_store import BlobStorageManager

# Set hermetic test environment variables before modules initialize
os.environ["OPENAI_API_KEY"] = ""
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["STORAGE_DIR"] = "/tmp/test_blobs_root"

# Conditional pytest import so tests run with stdlib unittest or pytest
try:
    import pytest
    _has_pytest = True
except ImportError:
    _has_pytest = False


def create_test_fixtures():
    """Factory helper to construct test objects for unittest or standalone execution."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_blobs_"))
    test_db = DatabaseManager(in_memory=True)
    blob_mgr = BlobStorageManager(base_path=temp_dir)
    settings = Settings(openai_api_key="", embedding_provider="mock")
    emb_service = EmbeddingService(settings=settings)
    meta_extractor = MetadataExtractor(embedding_service=emb_service)
    struct_engine = StructuredIngestionEngine(
        db_manager=test_db, blob_manager=blob_mgr, metadata_extractor=meta_extractor
    )
    unstruct_engine = UnstructuredIngestionEngine(
        db_manager=test_db, blob_manager=blob_mgr, embedding_service=emb_service, chunk_size=400, chunk_overlap=80
    )
    pruner = TwoStageSchemaPruner(
        db_manager=test_db, blob_manager=blob_mgr, embedding_service=emb_service
    )
    return {
        "temp_dir": temp_dir,
        "test_db": test_db,
        "blob_manager": blob_mgr,
        "settings": settings,
        "embedding_service": emb_service,
        "metadata_extractor": meta_extractor,
        "structured_engine": struct_engine,
        "unstructured_engine": unstruct_engine,
        "schema_pruner": pruner,
    }


SAMPLE_CSV_TEXT = (
    "order_id,customer_id,order_date,status,total_amount,shipping_city\n"
    "101,501,2024-01-10 10:00:00,completed,150.50,New York\n"
    "102,502,2024-01-11 11:30:00,completed,280.00,San Francisco\n"
    "103,501,2024-01-12 14:15:00,shipped,75.25,New York\n"
    "104,503,2024-01-13 09:45:00,cancelled,45.00,Chicago\n"
    "105,504,2024-01-14 16:20:00,completed,510.80,Austin\n"
)

SAMPLE_CUSTOMERS_CSV_TEXT = (
    "customer_id,first_name,last_name,email,country,is_active\n"
    "501,Alice,Smith,alice@example.com,USA,true\n"
    "502,Bob,Jones,bob@example.com,USA,true\n"
    "503,Charlie,Brown,charlie@example.com,Canada,false\n"
    "504,Diana,Prince,diana@example.com,UK,true\n"
)

SAMPLE_MARKDOWN_TEXT = (
    "# Engineering Operations Handbook\n\n"
    "## Deployment Guidelines\n"
    "All deployments must pass continuous integration tests before release. "
    "Canary releases should be monitored for at least 15 minutes before full traffic migration.\n\n"
    "## Incident Response Protocol\n"
    "When a Severity 1 incident occurs, page the on-call engineer and open an incident Slack channel. "
    "A post-mortem document must be published within 48 hours of resolution.\n\n"
    "## Code Review Policy\n"
    "Every pull request requires two approving reviews and zero unresolved comments. "
    "Security-sensitive modules require an explicit sign-off from the Application Security team.\n"
)

if _has_pytest:
    @pytest.fixture
    def temp_blob_dir() -> Generator[Path, None, None]:
        temp_dir = Path(tempfile.mkdtemp(prefix="test_blobs_"))
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_db() -> DatabaseManager:
        return DatabaseManager(in_memory=True)

    @pytest.fixture
    def blob_manager(temp_blob_dir: Path) -> BlobStorageManager:
        return BlobStorageManager(base_path=temp_blob_dir)

    @pytest.fixture
    def embedding_service() -> EmbeddingService:
        settings = Settings()
        settings.embedding_provider = "mock"
        return EmbeddingService(settings=settings)

    @pytest.fixture
    def metadata_extractor(embedding_service: EmbeddingService) -> MetadataExtractor:
        return MetadataExtractor(embedding_service=embedding_service)

    @pytest.fixture
    def structured_engine(
        test_db: DatabaseManager,
        blob_manager: BlobStorageManager,
        metadata_extractor: MetadataExtractor,
    ) -> StructuredIngestionEngine:
        return StructuredIngestionEngine(
            db_manager=test_db,
            blob_manager=blob_manager,
            metadata_extractor=metadata_extractor,
        )

    @pytest.fixture
    def unstructured_engine(
        test_db: DatabaseManager,
        blob_manager: BlobStorageManager,
        embedding_service: EmbeddingService,
    ) -> UnstructuredIngestionEngine:
        return UnstructuredIngestionEngine(
            db_manager=test_db,
            blob_manager=blob_manager,
            embedding_service=embedding_service,
            chunk_size=400,
            chunk_overlap=80,
        )

    @pytest.fixture
    def schema_pruner(
        test_db: DatabaseManager,
        blob_manager: BlobStorageManager,
        embedding_service: EmbeddingService,
    ) -> TwoStageSchemaPruner:
        return TwoStageSchemaPruner(
            db_manager=test_db,
            blob_manager=blob_manager,
            embedding_service=embedding_service,
        )

    @pytest.fixture
    def sample_csv_text() -> str:
        return SAMPLE_CSV_TEXT

    @pytest.fixture
    def sample_customers_csv_text() -> str:
        return SAMPLE_CUSTOMERS_CSV_TEXT

    @pytest.fixture
    def sample_markdown_text() -> str:
        return SAMPLE_MARKDOWN_TEXT
