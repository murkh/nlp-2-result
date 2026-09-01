-- =============================================================================
-- Database Initialization DDL for Multi-Agent Knowledge Base Q&A Platform
-- Supports PostgreSQL 15+ with pgvector, pg_trgm, and full-text search
-- =============================================================================

-- Create dedicated database for Langfuse (must be separate from knowledge_base)
SELECT 'CREATE DATABASE langfuse' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- -----------------------------------------------------------------------------
-- 1. Datasets Registry Table
-- Tracks all uploaded structured and unstructured files and storage locations
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_type VARCHAR(32) NOT NULL, -- 'csv', 'parquet', 'excel', 'pdf', 'docx', 'txt', 'md'
    category VARCHAR(32) NOT NULL,  -- 'structured', 'unstructured'
    blob_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    row_count INTEGER,
    page_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_datasets_category ON datasets(category);
CREATE INDEX IF NOT EXISTS idx_datasets_file_type ON datasets(file_type);

-- -----------------------------------------------------------------------------
-- 2. Structured Table Metadata (Stage 1 Schema Pruning)
-- Stores table-level summaries, statistics, and 384-dim vector embeddings
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS table_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    table_name VARCHAR(63) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    row_count BIGINT NOT NULL DEFAULT 0,
    column_count INTEGER NOT NULL DEFAULT 0,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_table_metadata_embedding 
ON table_metadata 
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_table_metadata_dataset_id 
ON table_metadata(dataset_id);

CREATE INDEX IF NOT EXISTS idx_table_metadata_name_trgm 
ON table_metadata 
USING gin (table_name gin_trgm_ops);

-- -----------------------------------------------------------------------------
-- 3. Structured Column Metadata (Stage 2 Schema Pruning)
-- Stores column data types, key constraints, statistical profiles, and embeddings
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS column_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_id UUID NOT NULL REFERENCES table_metadata(id) ON DELETE CASCADE,
    column_name VARCHAR(63) NOT NULL,
    data_type VARCHAR(64) NOT NULL,
    is_primary_key BOOLEAN NOT NULL DEFAULT FALSE,
    is_foreign_key BOOLEAN NOT NULL DEFAULT FALSE,
    foreign_target_table VARCHAR(63),
    foreign_target_column VARCHAR(63),
    null_percentage DOUBLE PRECISION DEFAULT 0.0,
    distinct_values_count BIGINT DEFAULT 0,
    sample_values JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT NOT NULL,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_table_column UNIQUE (table_id, column_name)
);

CREATE INDEX IF NOT EXISTS idx_column_metadata_embedding 
ON column_metadata 
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_column_metadata_table_id 
ON column_metadata(table_id);

CREATE INDEX IF NOT EXISTS idx_column_metadata_keys 
ON column_metadata(table_id, is_primary_key, is_foreign_key);

-- -----------------------------------------------------------------------------
-- 4. Unstructured Document Chunks (Hybrid Dense + Sparse Search)
-- Stores text chunks with 384-dim vector embeddings and stored tsvectors
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    section_title TEXT,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    embedding vector(384),
    content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_document_chunks_tsvector 
ON document_chunks 
USING gin (content_tsvector);

CREATE INDEX IF NOT EXISTS idx_document_chunks_dataset_id 
ON document_chunks(dataset_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_dataset_page 
ON document_chunks(dataset_id, page_number);

-- -----------------------------------------------------------------------------
-- 5. Query Logs & Execution Telemetry
-- Stores latency, token usage, generated code, and status for observability
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(64),
    query_text TEXT NOT NULL,
    engine VARCHAR(32) NOT NULL, -- 'pandas_sandbox', 'unstructured_rag'
    status VARCHAR(32) NOT NULL, -- 'success', 'error', 'clarification'
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    latency_ms DOUBLE PRECISION DEFAULT 0.0,
    generated_code TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_engine ON query_logs(engine);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at DESC);
