# Multi-Agent Knowledge Base Q&A Platform

A token-efficient Multi-Agent Knowledge Base Q&A platform supporting structured (CSV, Parquet, Excel) and unstructured (PDF, DOCX, TXT, MD) datasets.

Structured questions are answered by a sandboxed Python/Pandas engine; document questions by a hybrid dense + sparse RAG engine. The platform ships dedicated API endpoints, Langfuse observability, full Docker Compose deployment, RAGAS evaluation suites, and an interactive Streamlit UI.

---

## Architecture Overview

```
                               ┌───────────────────────────────┐
                               │      Streamlit Web UI         │
                               │       (Ingest / Q&A)          │
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
│  │   - STRUCTURED_QUERY       -> Dispatches to the Pandas Sandbox                        │  │
│  │   - UNSTRUCTURED_QUERY     -> Dispatches to Hybrid RAG Engine                         │  │
│  └──────────────────────────────────────────┬────────────────────────────────────────────┘  │
│                                             │                                               │
│                    ┌────────────────────────┴────────────────────────┐                      │
│                    ▼                                                 ▼                      │
│         ┌──────────────────────┐                          ┌────────────────┐                │
│         │    Pandas Sandbox    │                          │  Unstructured  │                │
│         │  Blob Parquet / CSV  │                          │   Hybrid RAG   │                │
│         │ Subprocess Sandbox   │                          │ Dense pgvector │                │
│         │ AST Whitelist + CPU  │                          │+ Sparse ts/BM25│                │
│         └──────────┬───────────┘                          └────────┬───────┘                │
│                    │                                               │                        │
│                    └───────────────────────┬───────────────────────┘                        │
│                                            ▼                                                │
│                              ┌───────────────────────────┐                                  │
│                              │     Synthesizer Agent     │                                  │
│                              │ (Answers, Tables, Evidence│                                  │
│                              └─────────────┬─────────────┘                                  │
└────────────────────────────────────────────┼────────────────────────────────────────────────┘
                                             │
            ┌────────────────────────────────┴──────────────────┐
            ▼                                                   ▼
┌───────────────────────────────────────┐   ┌────────────────────────────────────────┐
│     PostgreSQL + pgvector Database    │   │            Blob Storage                │
│ - datasets & table/column metadata    │   │ /app/data/blob/{dataset_id}/{filename} │
│ - document_chunks (HNSW + tsvector)   │   │ - raw CSV / Parquet / Excel files      │
│ - query_logs                          │   │ - source PDF / DOCX / TXT / MD files   │
└───────────────────────────────────────┘   └────────────────────────────────────────┘
```

---

## Execution Engines

| Engine | Endpoint | Mechanism | Key Advantage |
|---|---|---|---|
| **Pandas Sandbox** | `POST /query/pandas-sandbox` | Python code generation executed in an isolated subprocess over the raw Parquet/CSV blob | Flexible for custom math and non-SQL transforms; no DB schema pollution |
| **Unstructured Hybrid RAG** | `POST /query/unstructured-rag` | Dense pgvector + sparse tsvector retrieval fused with RRF | Grounded answers with bracketed source citations |

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

#### Projection Critic
Guards against answers that return only a row key. A deterministic gate checks whether the
generated SQL projects the columns an analyst needs to verify the result — the entity's
human-readable identifier, the columns being compared, and the measures. When the projection is
already adequate the node makes **no LLM call and no second query**; only a thin projection is
sent for widening.

```bash
PROJECTION_CRITIC_ENABLED=true  # set false to disable the node entirely
CRITIC_MODEL=                   # empty = reuse OPENAI_MODEL
```

Widening a SELECT list against a given DDL is a small task, so `CRITIC_MODEL` can point at a
cheaper/faster model than `OPENAI_MODEL`. It must be a model id your gateway actually serves —
behind bedrock-mantle that is a Bedrock model id, not an OpenAI one. Leaving it empty reuses
`OPENAI_MODEL`, which always works.

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
- `POST /query/pandas-sandbox` — Execute via the sandboxed Python/Pandas engine.
- `POST /query/unstructured-rag` — Execute via Hybrid Dense+Sparse Document RAG.

---

## License
Apache 2.0
