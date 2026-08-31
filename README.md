# Multi-Agent Knowledge Base Q&A Platform

A token-efficient Multi-Agent Knowledge Base Q&A platform supporting structured (CSV, Parquet, Excel) and unstructured (PDF, DOCX, TXT, MD) datasets.

The platform implements and benchmarks three distinct execution strategies with dedicated API endpoints, Langfuse observability, full Docker Compose deployment, RAGAS evaluation suites, and an interactive Streamlit comparison UI.

---

## Architecture Overview

```
                               ┌───────────────────────────────┐
                               │      Streamlit Web UI         │
                               │  (Ingest / Q&A / Benchmark)   │
                               └───────────────┬───────────────┘
                                               │ HTTP (Port 8501)
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FastAPI Backend (Port 8000)                           │
│                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              LangGraph Supervisor Router                              │  │
│  │   - GREETING_OR_CHITCHAT   -> Immediate conversational reply (0 DB / tool tokens)     │  │
│  │   - AMBIGUOUS_QUERY        -> Proactive dataset suggestions & clarification questions │  │
│  │   - STRUCTURED_QUERY       -> Dispatches to Selected Strategy (A, B, or C)            │  │
│  │   - UNSTRUCTURED_QUERY     -> Dispatches to Hybrid RAG Engine                         │  │
│  └──────────────────────────────────────────┬────────────────────────────────────────────┘  │
│                                             │                                               │
│    ┌──────────────────┬─────────────────────┼──────────────────────┬────────────────┐       │
│    ▼                  ▼                     ▼                      ▼                ▼       │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ ┌────────────────┐ ┌──────────┐ │
│ │  Strategy A  │ │  Strategy B  │ │      Strategy C      │ │  Unstructured  │ │Benchmark │ │
│ │(Dedicated DB)│ │   (DuckDB)   │ │   (Pandas Sandbox)   │ │   Hybrid RAG   │ │  Arena   │ │
│ │ PostgreSQL   │ │ Blob Parquet/│ │ Subprocess Sandbox   │ │ Dense pgvector │ │Parallel  │ │
│ │ Text2SQL     │ │ CSV in-memory│ │ AST Whitelist + CPU  │ │+ Sparse ts/BM25│ │A, B, C   │ │
│ └──────┬───────┘ └──────┬───────┘ └─────────┬────────────┘ └────────┬───────┘ └────┬─────┘ │
│        │                │                   │                       │              │        │
│        └────────────────┴───────────┬───────┴───────────────────────┴──────────────┘        │
│                                     ▼                                                       │
│                       ┌───────────────────────────┐                                         │
│                       │     Synthesizer Agent     │                                         │
│                       │ (Answers, Tables, Evidence│                                         │
│                       └─────────────┬─────────────┘                                         │
└─────────────────────────────────────┼───────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
┌───────────────────────────────────────┐   ┌────────────────────────────────────────┐
│     PostgreSQL + pgvector Database    │   │            Blob Storage                │
│ - datasets & table/column metadata    │   │ /app/data/blob/{dataset_id}/{filename} │
│ - document_chunks (HNSW + tsvector)   │   │ - raw CSV / Parquet / Excel files      │
│ - dedicated tables (tbl_<id>)         │   │ - source PDF / DOCX / TXT / MD files   │
└───────────────────────────────────────┘   └────────────────────────────────────────┘
```

---

## The 3 Structured Execution Strategies

| Strategy | Endpoint | Mechanism | Key Advantage |
|---|---|---|---|
| **Strategy A: Dedicated DB** | `POST /query/dedicated-db` | Dedicated PostgreSQL table per file + Text2SQL | Native SQL engine, persistent indexing |
| **Strategy B: In-Memory DuckDB** | `POST /query/duckdb` | Direct vectorized columnar scan over Parquet/CSV in blob | Serverless, zero DB schema pollution, high performance |
| **Strategy C: Pandas Sandbox** | `POST /query/pandas-sandbox` | Python code generation executed in isolated subprocess | Flexible for custom math, non-SQL transforms |
| **Benchmark Arena** | `POST /query/benchmark` | Runs Strategy A, B, and C concurrently | Head-to-head comparison of accuracy, latency, and token cost |

---

## Quickstart with Docker Compose

### 1. Configure Environment
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=sk-your-openai-api-key

# Optional OpenAI-compatible gateway base URL (e.g. bedrock-mantle)
OPENAI_API_URL=

# Optional Langfuse tracing keys (leave empty to use in-memory tracing)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

#### Embeddings
Defaults need no API key: `fastembed` runs `BAAI/bge-small-en-v1.5` (384-dim) locally on ONNX
Runtime. The model (~130MB) is downloaded from HuggingFace on first use and cached — in Docker on
the `fastembed_cache` volume, so it downloads once.

```bash
EMBEDDING_PROVIDER=fastembed   # fastembed (default) | openai | mock
EMBEDDING_MODEL=               # empty = provider default (bge-small-en-v1.5 / text-embedding-3-small)
EMBEDDING_DIM=384              # must match the model AND vector(N) in scripts/init_db.sql
```

`mock` yields deterministic hash-projection vectors for offline runs and the test suite — it does
**not** retrieve semantically. A provider that fails to initialise now raises instead of silently
falling back to `mock`.

Changing `EMBEDDING_DIM` means editing `vector(384)` in `scripts/init_db.sql` and recreating the
database (`docker compose down -v`), then re-ingesting.

### 2. Start Full Stack
```bash
docker-compose up --build -d
```

### 3. Access Services
- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Langfuse Tracing Dashboard**: [http://localhost:3000](http://localhost:3000)

---

## Local Development (with `uv`)

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Run test suite
uv run pytest

# 3. Run FastAPI backend
uv run uvicorn src.main:app --reload --port 8000

# 4. Run Streamlit UI
uv run streamlit run frontend/ui.py --server.port 8501
```

---

## API Endpoints Reference

### Ingestion & Catalog
- `POST /ingest` — Upload and ingest structured (CSV/Parquet/Excel) or unstructured (PDF/DOCX/TXT/MD) file.
- `GET /datasets` — List registered datasets and schemas.
- `GET /datasets/{dataset_id}` — Get dataset metadata.

### Query Execution
- `POST /query/agent` — Conversational Q&A with LangGraph supervisor (greetings, ambiguity clarification, auto-routing).
- `POST /query/dedicated-db` — Execute via Strategy A (PostgreSQL).
- `POST /query/duckdb` — Execute via Strategy B (In-Memory DuckDB).
- `POST /query/pandas-sandbox` — Execute via Strategy C (Python Sandbox).
- `POST /query/unstructured-rag` — Execute via Hybrid Dense+Sparse Document RAG.
- `POST /query/benchmark` — Run Strategy A, B, and C concurrently with comparison telemetry.

---

## License
Apache 2.0
