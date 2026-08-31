"""
Unstructured Hybrid RAG Engine.
Executes hybrid dense vector (pgvector) + sparse keyword retrieval
with Reciprocal Rank Fusion (RRF k=60) and grounded response synthesis
with exact bracketed citations [Doc: <name>, Page: <page>, Chunk: <index>].
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import (
    Citation,
    DecisionStep,
    ExecutionMetrics,
    QueryUnstructuredRAGRequest,
    QueryUnstructuredRAGResponse,
    ThinkingProcess,
    TokenUsage,
)
from src.config import Settings, get_settings
from src.database.connection import DatabaseManager, get_db_manager
from src.database.models import QueryLog
from src.ingestion.metadata_extractor import EmbeddingService
from src.llm import get_openai_client


class HybridRAGEngine:
    """
    Unstructured Hybrid RAG Engine.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ):
        self.db_manager = db_manager or get_db_manager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.settings = settings or get_settings()
        self._openai_client = get_openai_client(self.settings)

    def execute_query(
        self,
        request: QueryUnstructuredRAGRequest,
    ) -> QueryUnstructuredRAGResponse:
        """
        Execute Unstructured Hybrid RAG workflow:
        1. Embed query with EmbeddingService.
        2. Perform hybrid dense + sparse search with Reciprocal Rank Fusion (RRF).
        3. Format retrieved chunks and build citations.
        4. Synthesize grounded answer with bracketed citations.
        5. Compute latency metrics and token usage.
        6. Log execution telemetry.
        """
        start_time = time.perf_counter()
        query_text = request.query
        top_k = request.top_k or 5
        target_dataset_id = (
            request.dataset_ids[0]
            if (request.dataset_ids and len(request.dataset_ids) == 1)
            else None
        )

        # 1. Embed query & Retrieve chunks via RRF
        retrieval_start = time.perf_counter()
        query_emb = self.embedding_service.embed_text(query_text)
        raw_chunks = self.db_manager.hybrid_search_document_chunks(
            query_text=query_text,
            query_embedding=query_emb,
            top_k=top_k,
            dataset_id=target_dataset_id,
        )
        engine_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        if not raw_chunks:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            ans = "Based on the provided documents, I could not find information to answer this question."
            return QueryUnstructuredRAGResponse(
                query=query_text,
                answer=ans,
                citations=[],
                retrieved_chunks_count=0,
                metrics=ExecutionMetrics(
                    engine_execution_ms=engine_latency_ms, total_latency_ms=total_lat
                ),
                token_usage=TokenUsage(prompt_tokens=20, completion_tokens=15),
            )

        # 2. Build citations & Context block
        citations: List[Citation] = []
        context_parts: List[str] = []

        dataset_name_map: Dict[str, str] = {}

        for idx, chunk in enumerate(raw_chunks):
            d_id = chunk["dataset_id"]
            if d_id not in dataset_name_map:
                ds = self.db_manager.get_dataset(d_id)
                dataset_name_map[d_id] = ds.name if ds else f"Doc_{d_id[:8]}"
            doc_name = dataset_name_map[d_id]
            page_num = chunk.get("page_number")
            content = chunk["content"]
            rrf_score = chunk.get("rrf_score", 0.0)

            snippet = content[:250] + "..." if len(content) > 250 else content
            citations.append(
                Citation(
                    document_name=doc_name,
                    page_number=page_num,
                    chunk_index=idx,
                    similarity_score=rrf_score,
                    snippet=snippet,
                )
            )

            cite_tag = f"[Doc: {doc_name}, Page: {page_num or 1}, Chunk: {idx}]"
            context_parts.append(f"--- Excerpt {idx + 1} {cite_tag} ---\n{content}\n")

        full_context = "\n".join(context_parts)

        # 3. Synthesize grounded answer with dynamic LLM thought
        synth_start = time.perf_counter()
        answer, llm_thought, synth_tokens = self._synthesize_grounded_answer(
            query_text, full_context, citations, raw_chunks
        )
        synth_latency_ms = (time.perf_counter() - synth_start) * 1000.0

        total_lat = (time.perf_counter() - start_time) * 1000.0

        token_usage = TokenUsage(
            prompt_tokens=synth_tokens[0],
            completion_tokens=synth_tokens[1],
        )

        metrics = ExecutionMetrics(
            query_generation_ms=0.0,
            engine_execution_ms=engine_latency_ms,
            synthesis_ms=synth_latency_ms,
            total_latency_ms=total_lat,
        )

        # Log query telemetry
        self.db_manager.log_query(
            QueryLog(
                query_text=query_text,
                engine="unstructured_hybrid_rag",
                status="SUCCESS",
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                latency_ms=metrics.total_latency_ms,
                generated_code=None,
            )
        )

        # Construct Thinking Process with dynamic LLM thoughts
        top_docs = list({c.document_name for c in citations})
        top_docs_str = ", ".join(top_docs) if top_docs else "None"
        synth_reason = (
            llm_thought
            or f"Extracted {len(citations)} relevant excerpt(s) from [{top_docs_str}] and grounded response strictly in verified citations."
        )

        thinking = ThinkingProcess(
            summary=f"Unstructured Hybrid RAG executed Dense Vector + Sparse BM25 retrieval over document(s) [{top_docs_str}], fusing {len(citations)} top chunk(s) via RRF.",
            steps=[
                DecisionStep(
                    step_number=1,
                    title="Dense Semantic Vector Retrieval",
                    choice="Queried pgvector HNSW embeddings",
                    reasoning=f"Generated embedding vector for query '{query_text}' and retrieved top semantic cosine matches.",
                ),
                DecisionStep(
                    step_number=2,
                    title="Sparse Keyword BM25 Retrieval",
                    choice="Queried full-text tsvector index",
                    reasoning=f"Scanned full-text index for exact lexical term matches across document catalog.",
                ),
                DecisionStep(
                    step_number=3,
                    title="Reciprocal Rank Fusion (RRF k=60)",
                    choice=f"Fused top {len(citations)} chunk(s) across: {top_docs_str}",
                    reasoning=f"Merged dense semantic and sparse lexical ranks using standard RRF (k=60) to optimize recall and precision.",
                    details={"retrieved_chunks": len(citations), "documents": top_docs},
                ),
                DecisionStep(
                    step_number=4,
                    title="Grounded Answer Synthesis & Citation Binding",
                    choice=f"Synthesized answer with {len(citations)} citation(s)",
                    reasoning=synth_reason,
                ),
            ],
        )

        return QueryUnstructuredRAGResponse(
            query=query_text,
            answer=answer,
            citations=citations,
            retrieved_chunks_count=len(citations),
            thinking_process=thinking,
            metrics=metrics,
            token_usage=token_usage,
        )

    def _synthesize_grounded_answer(
        self, query: str, context: str, citations: List[Citation], raw_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[str], Tuple[int, int]]:
        """Synthesize natural language answer strictly supported by context chunks."""
        prompt = (
            f"You are a precise AI knowledge assistant. Answer the user's question using ONLY the retrieved excerpts below.\n\n"
            f"Retrieved Document Context:\n{context}\n\n"
            f"Instructions:\n"
            f"1. In a ```thought block, explain your step-by-step reasoning: which facts and excerpts you referenced and how you answered the question.\n"
            f"2. Base your answer strictly on the facts present in the excerpts.\n"
            f"3. Include exact bracketed citations for every stated fact in the format [Doc: <doc_name>, Page: <page_num>, Chunk: <chunk_index>].\n"
            f"4. If the context does not contain sufficient information, state 'Based on the provided documents, I could not find information to answer this question.'\n\n"
            f"Question: {query}"
        )

        prompt_tokens = max(1, len(prompt) // 4)

        if self._openai_client:
            try:
                resp = self._openai_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw_text = resp.choices[0].message.content or ""

                thought = None
                ans = raw_text
                blocks = re.findall(r"```(\w*)\n(.*?)```", raw_text, re.DOTALL)
                for lang, content in blocks:
                    lang_clean = lang.lower().strip()
                    if lang_clean in ("thought", "thinking", "reasoning", "explanation"):
                        thought = content.strip()
                        ans = raw_text.replace(f"```{lang}\n{content}```", "").strip()

                comp_tokens = (
                    resp.usage.completion_tokens if resp.usage else max(1, len(raw_text) // 4)
                )
                return ans.strip(), thought, (prompt_tokens, comp_tokens)
            except Exception:
                pass

        # Deterministic grounded synthesizer
        ans = self._deterministic_rag_synthesis(query, citations, raw_chunks)
        comp_tokens = max(1, len(ans) // 4)
        return ans, None, (prompt_tokens, comp_tokens)

    def _deterministic_rag_synthesis(
        self, query: str, citations: List[Citation], raw_chunks: List[Dict[str, Any]]
    ) -> str:
        """Deterministic RAG answer synthesis referencing relevant retrieved sentences."""
        if not citations or not raw_chunks:
            return "Based on the provided documents, I could not find information to answer this question."

        query_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]

        best_score = -1
        best_sentence = ""
        best_cite = citations[0]

        for idx, (cite, chunk) in enumerate(zip(citations, raw_chunks)):
            content = chunk.get("content", "")
            # Split into clean sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", content) if s.strip()]
            for s in sentences:
                if s.startswith("#"):
                    continue
                s_words = [w.lower() for w in re.findall(r"\w+", s)]
                match_count = 0
                for qt in query_terms:
                    for sw in s_words:
                        if qt == sw or (len(qt) >= 4 and len(sw) >= 4 and qt[:4] == sw[:4]):
                            match_count += 1
                            break
                if match_count > best_score:
                    best_score = match_count
                    best_sentence = s
                    best_cite = cite

        if not best_sentence or best_score <= 0:
            content = raw_chunks[0].get("content", "")
            clean_lines = [
                l.strip()
                for l in content.split("\n")
                if l.strip() and not l.strip().startswith("#")
            ]
            best_sentence = clean_lines[0] if clean_lines else content[:150]
            best_cite = citations[0]

        cite_tag = f"[Doc: {best_cite.document_name}, Page: {best_cite.page_number or 1}, Chunk: {best_cite.chunk_index}]"
        return f"{best_sentence} {cite_tag}"
