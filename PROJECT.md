# Project: Multi-Agent Knowledge Base Q&A Platform

## Architecture
Production-grade, token-efficient Multi-Agent Knowledge Base Q&A platform supporting structured (CSV, Parquet, Excel) and unstructured (PDF, DOCX, TXT, MD) datasets. Structured questions run on a sandboxed Python/Pandas engine; document questions on a hybrid dense + sparse RAG engine. Ships dedicated API endpoints, Langfuse observability, Ragas evaluation suites, an interactive Streamlit UI, and Docker Compose deployment.

```
                                  ┌──────────────────────────────┐
                                  │      Streamlit Web UI         │
                                  │       (Ingest / Q&A)          │
                                  └───────────────┬───────────────┘
                                                  │ HTTP (FastAPI)
                                                  ▼
FastAPI Backend
  LangGraph Supervisor Router
    - GREETING_OR_CHITCHAT   -> Immediate conversational reply (0 DB / tool calls)
    - AMBIGUOUS_QUERY        -> Proactive dataset suggestions & clarification questions
    - STRUCTURED_QUERY       -> Dispatches to the Pandas Sandbox
    - UNSTRUCTURED_QUERY     -> Dispatches to the Hybrid RAG Engine

      Pandas Sandbox                          Unstructured Hybrid RAG
      Blob Parquet / CSV                      Dense pgvector
      Subprocess (python -I -S)               + Sparse tsvector / BM25
      AST Whitelist + limits

                       Synthesizer Agent
              (Answers, Tables, Evidence, Citations)

PostgreSQL + pgvector                     Blob Storage
 - datasets registry                       storage/blobs/{dataset_id}/{filename}
 - table_metadata & column_metadata        - raw CSV / Parquet / Excel files
 - document_chunks (HNSW + GIN tsvector)   - source PDF / DOCX / TXT / MD files
 - query_logs
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Ingestion: Structured | Upload CSV, Parquet, Excel to blob store (`storage/blobs/`) and index table/column metadata | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Ingestion: Unstructured | Chunk & embed PDF, DOCX, TXT, MD into PostgreSQL `document_chunks` table with metadata (page, section, offsets) | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Metadata & Vector Embedding Catalog | Extract and embed table/column metadata into `table_metadata` and `column_metadata` with HNSW cosine indexes | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Two-Stage Schema Pruner | Vector-based Stage 1 table retrieval + Stage 2 column retrieval with PK/FK preservation & `LIMIT 20` prompt injection | M1 | ORIGINAL_REQUEST §R1 |
| 7 | Pandas Sandbox Query Engine (`POST /query/pandas-sandbox`) | Sandboxed Python DataFrame execution with AST validation, isolated subprocess (`python -I -S`), 512MB RAM & 5s timeout | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Unstructured Hybrid RAG (`POST /query/unstructured-rag`) | Hybrid dense (`pgvector`) + sparse (`tsvector`/BM25) search with Reciprocal Rank Fusion (RRF) & bracketed citations | M2 | ORIGINAL_REQUEST §R2 |
| 10 | FastAPI Backend & Pydantic V2 Schemas | Modular REST API with full request/response schemas, error handling, and telemetry metadata | M2 | ORIGINAL_REQUEST §R2 |
| 11 | LangGraph Supervisor Router | State machine classifying `GREETING_OR_CHITCHAT`, `AMBIGUOUS_QUERY`, `STRUCTURED_QUERY`, `UNSTRUCTURED_QUERY` | M3 | ORIGINAL_REQUEST §R3 |
| 12 | Synthesizer Agent | Response synthesis formatting natural language answers, markdown tables, citations, and execution telemetry | M3 | ORIGINAL_REQUEST §R3 |
| 13 | Observability with Langfuse | Request root traces, child spans for router/pruner/generator/sandbox/synthesis, token aggregation, local fallback | M3 | ORIGINAL_REQUEST §R4 |
| 14 | Ragas Unstructured Evaluation Suite | Automated evaluation measuring `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` | M4 | ORIGINAL_REQUEST §R4 |
| 15 | Structured Execution Equivalence Suite | Ground-Truth execution equivalence vs golden outputs (`assert_frame_equal`), syntax first-pass rate, cost/latency | M4 | ORIGINAL_REQUEST §R4 |
| 16 | Packaging & Configuration | `pyproject.toml` with `uv`, `pydantic-settings` environment configuration, PostgreSQL init DDL script | M1, M5 | ORIGINAL_REQUEST §R5 |
| 17 | Streamlit Web UI (`frontend/ui.py`) | 2-tab UI: Ingestion Hub, Conversational Q&A with Engine Selector | M5 | ORIGINAL_REQUEST §R5 |
| 18 | Docker Compose Stack | 4-container stack (`postgres-pgvector`, `backend`, `frontend`, `langfuse`) with healthchecks and multi-stage Dockerfiles | M5 | ORIGINAL_REQUEST §R5 |
| 19 | E2E Integration & Verification | 100% test pass on full suite (`uv run pytest`) across all features and acceptance criteria | M6 | ORIGINAL_REQUEST §Acceptance Criteria |
| 20 | Adversarial Coverage Hardening | White-box stress testing, security boundary tests, AST bypass attempt validation, edge-case datasets | M6 | Project Pattern Tier 5 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Suite (Parallel Track) | Design and implement opaque-box test runner & Tiers 1-4 test suite per TEST_INFRA.md | None | DONE (174 tests, TEST_READY.md) |
| M1 | Ingestion, Storage & Schema Pruner | Database DDL, blob storage, structured/unstructured ingestion pipelines, metadata embedding, 2-stage schema pruner | None | DONE |
| M2 | Execution Engines & Query Endpoints | Pandas Sandbox, Unstructured RAG, FastAPI routes | M1 | DONE |
| M3 | Multi-Agent Orchestration & Observability | LangGraph supervisor router (4 intents), Synthesizer agent, Langfuse tracing wrapper & local fallback | M2 | DONE |
| M4 | Evaluation Frameworks | Ragas unstructured evaluation suite + Structured Ground-Truth Execution Equivalence suite | M2, M3 | DONE |
| M5 | Streamlit Web UI & Docker Compose Stack | 2-tab Streamlit UI (`frontend/ui.py`), Dockerfiles, `docker-compose.yml` stack | M2, M3 | IN_PROGRESS |
| M6 | Final Verification & Adversarial Hardening | Pass 100% E2E test suite (Phase 1) + Tier 5 Challenger Adversarial Coverage Hardening (Phase 2) | E2E, M1-M5 | PLANNED |

## Interface Contracts

### 1. Ingestion ↔ Storage Engine (`src/ingestion/`)
- `ingest_file(file_path: Path, filename: str, content_type: str) -> DatasetRecord`:
  - Saves file to `storage/blobs/{dataset_id}/{filename}`
  - If structured: derives the logical table name `tbl_{dataset_id_prefix}_{name}` and generates table & column metadata records with embeddings. The blob is the only copy of the rows.
  - If unstructured: parses text, generates recursive chunks (800 chars / 150 overlap), embeds chunks, inserts into `document_chunks` table with `tsvector`.

### 2. Schema Pruner ↔ Execution Engines (`src/pruning/`)
- `prune_schema(query: str, dataset_ids: Optional[List[str]] = None, top_k_tables: int = 3, max_cols: int = 20) -> PrunedSchemaContext`:
  - Returns `table_names: List[str]`, `ddl_prompt_snippet: str`, `file_paths: Dict[str, str]` (for the Pandas sandbox).
  - Enforces `LIMIT 20` directive in generated schema prompts.

### 3. Execution Engines ↔ FastAPI Endpoints (`src/engines/`)
- Pandas Sandbox: `execute_pandas_sandbox(code: str, dataset_id: str) -> SandboxExecutionResult`
- Unstructured RAG: `execute_hybrid_rag(query: str, top_k: int = 5) -> RAGResult`

### 4. Router & Graph State Machine (`src/agent/`)
- `AgentState`:
  - `query: str`, `session_id: str`, `intent: str`, `candidate_datasets: List[str]`, `pruned_tables: List[Dict]`, `generated_code: Optional[str]`, `execution_result: Optional[List[Dict]]`, `final_answer: Optional[str]`, `telemetry: TelemetryData`

## Code Layout
```
/Users/murkh/Development/nlp-2-result/
├── pyproject.toml              # uv-managed packaging and dependency definitions
├── README.md                   # System documentation, quickstart, and API guide
├── docker-compose.yml          # Multi-container orchestration stack
├── Dockerfile.backend          # Backend container image definition
├── Dockerfile.frontend         # Frontend container image definition
├── scripts/
│   ├── init_db.sql             # PostgreSQL + pgvector schema initialization DDL
│   └── seed_data.py            # Sample datasets creation & initial ingestion
├── data/
│   ├── samples/                # Sample CSV, Parquet, Excel, PDF, DOCX, TXT files
│   └── blob/                   # Local blob storage directory
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entrypoint and lifespan handlers
│   ├── config.py               # Pydantic-settings configuration models
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py       # Async & sync PostgreSQL / pgvector connection pools
│   │   └── models.py           # SQLAlchemy / SQL schema definitions
│   ├── storage/
│   │   ├── __init__.py
│   │   └── blob_store.py       # Blob storage manager
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── structured.py       # CSV, Parquet, Excel parser, blob writer & metadata indexer
│   │   ├── unstructured.py     # PDF, DOCX, TXT, MD parser, chunker & embedder
│   │   └── metadata_extractor.py # Statistical profiler and embedding generator
│   ├── pruning/
│   │   ├── __init__.py
│   │   └── schema_pruner.py    # Two-stage vector schema pruner with LIMIT 20
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── pandas_sandbox/     # Sandboxed Python DataFrame execution
│   │   │   ├── __init__.py
│   │   │   ├── ast_validator.py # AST security analyzer and whitelist enforcement
│   │   │   └── runner.py       # Subprocess runner (python -I -S) with resource limits
│   │   └── hybrid_rag.py       # Unstructured Hybrid RAG engine (dense + sparse RRF)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py            # LangGraph TypedDict state schema
│   │   ├── router.py           # Intent classification (Greeting, Ambiguous, Structured, Unstructured)
│   │   ├── graph.py            # LangGraph StateGraph compiler
│   │   └── nodes/              # LangGraph node implementations
│   │       ├── router_node.py
│   │       ├── chitchat_node.py
│   │       ├── clarify_node.py
│   │       ├── structured_node.py
│   │       ├── unstructured_node.py
│   │       └── synthesizer_node.py
│   ├── observability/
│   │   ├── __init__.py
│   │   └── telemetry.py        # Langfuse tracing wrapper & local fallback
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_suite.py      # Ragas unstructured RAG evaluation suite
│   │   └── structured_equivalence.py # Ground-truth DataFrame equivalence suite
│   └── api/
│       ├── __init__.py
│       ├── schemas.py          # Pydantic V2 request/response models
│       └── routes/
│           ├── ingest.py       # File upload & dataset management endpoints
│           ├── query.py        # Pandas sandbox & Hybrid RAG routes
│           └── agent.py        # Conversational LangGraph Q&A endpoint
├── frontend/
│   ├── __init__.py
│   └── ui.py                   # Streamlit Web UI (Ingestion, Q&A)
└── tests/
    ├── conftest.py             # Test fixtures, mock LLM/embeddings, test DB setup
    ├── test_ingestion.py       # Structured & unstructured ingestion unit tests
    ├── test_schema_pruner.py   # Two-stage schema pruning token efficiency tests
    ├── test_pandas_sandbox.py  # Sandbox security and execution tests
    ├── test_hybrid_rag.py      # Unstructured hybrid search & RRF tests
    ├── test_agent_router.py    # LangGraph routing & intent classification tests
    ├── test_observability.py   # Langfuse tracing & telemetry tests
    ├── test_ragas_suite.py     # Ragas evaluation harness tests
    ├── test_structured_equivalence.py # DataFrame equivalence tests
    └── e2e/                    # Opaque-box E2E test suites (Tiers 1-4)
        ├── test_tier1_features.py
        ├── test_tier2_boundaries.py
        ├── test_tier3_combinations.py
        └── test_tier4_workloads.py
```
