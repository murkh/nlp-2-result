"""
Synthesizer Node for Multi-Agent Output Generation.
Formats natural language answers with data evidence, markdown tables,
bracketed citations, and execution telemetry metadata.
"""

from typing import Any, Dict, List, Optional

from src.agent.state import AgentState
from src.observability.telemetry import get_tracer


def format_markdown_table(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None, max_rows: int = 20) -> str:
    """Format row dictionaries into a clean GitHub-flavored Markdown table."""
    if not rows:
        return ""

    cols = columns or list(rows[0].keys())
    if not cols:
        return ""

    header_line = "| " + " | ".join(str(c) for c in cols) + " |"
    separator_line = "| " + " | ".join("---" for _ in cols) + " |"
    
    table_lines = [header_line, separator_line]
    display_rows = rows[:max_rows]
    for r in display_rows:
        row_cells = [str(r.get(c, "")).replace("\n", " ").replace("|", "\\|") for c in cols]
        table_lines.append("| " + " | ".join(row_cells) + " |")

    table_md = "\n".join(table_lines)
    if len(rows) > max_rows:
        table_md += f"\n\n*Showing top {max_rows} rows (total {len(rows)} rows, truncated).* "
    return table_md


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizer Node:
    Takes tabular execution results, generated code, or retrieved document chunks
    and synthesizes a well-structured final answer with evidence, tables, citations, and telemetry.
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    intent = state.get("intent", "STRUCTURED_QUERY")
    strategy = state.get("suggested_strategy") or "duckdb"
    generated_code = state.get("generated_code")
    execution_result = state.get("execution_result") or []
    execution_columns = state.get("execution_columns") or []
    execution_error = state.get("execution_error")
    retrieved_chunks = state.get("retrieved_chunks") or []
    citations = list(state.get("citations") or [])
    raw_answer = state.get("final_answer")
    telemetry = state.get("telemetry", {}) or {}

    tracer = get_tracer()

    with tracer.start_trace(name="synthesizer_node", session_id=session_id) as trace:
        span = trace.start_span(
            "synthesize_final_response",
            input_data={
                "intent": intent,
                "strategy": strategy,
                "rows_count": len(execution_result),
                "chunks_count": len(retrieved_chunks),
                "has_error": bool(execution_error),
            },
        )

        final_answer = ""

        if execution_error:
            final_answer = (
                f"I encountered an issue executing your query: {execution_error}\n\n"
                "Please verify your query parameters or check the dataset schema."
            )
        elif intent == "STRUCTURED_QUERY":
            # 1. Base answer
            if raw_answer:
                base_text = raw_answer.strip()
            elif execution_result and len(execution_result) == 1 and len(execution_result[0]) == 1:
                # Single scalar result
                val = list(execution_result[0].values())[0]
                key = list(execution_result[0].keys())[0]
                base_text = f"The result for **{query}** is **{val}** ({key})."
            elif execution_result:
                base_text = f"Query executed successfully via {strategy.replace('_', ' ').title()}. Found {len(execution_result)} matching record(s)."
            else:
                base_text = "Query executed successfully, but returned zero rows."

            # 2. Markdown Table for multi-row or multi-column data
            table_md = ""
            if execution_result and not (len(execution_result) == 1 and len(execution_result[0]) == 1):
                table_md = format_markdown_table(execution_result, columns=execution_columns, max_rows=20)

            parts = [base_text]
            if table_md:
                parts.append(table_md)
            
            final_answer = "\n\n".join(parts)

            # Ensure citations are recorded
            if not citations:
                strat_label = "DuckDB (In-Memory)" if strategy == "duckdb" else ("PostgreSQL (Dedicated DB)" if strategy == "dedicated_db" else "Pandas Sandbox")
                citations.append(f"[Engine: {strat_label}, Rows: {len(execution_result)}]")

        elif intent == "UNSTRUCTURED_QUERY":
            if raw_answer:
                final_answer = raw_answer.strip()
            elif retrieved_chunks:
                # Combine top chunk snippets
                snippets = [f"- {c.get('content', c.get('snippet', ''))}" for c in retrieved_chunks[:3]]
                final_answer = f"Based on the relevant documentation:\n\n" + "\n\n".join(snippets)
            else:
                final_answer = "No matching documentation or knowledge chunks were found for your query."

            # If citations exist and not embedded in answer, add citation list
            if citations and not any(c in final_answer for c in citations):
                cit_text = "\n\n**Sources & Citations:**\n" + "\n".join(f"- {c}" for c in citations)
                final_answer += cit_text

        else:
            # Fallback for chitchat or clarification if routed here
            final_answer = raw_answer or "How can I assist you with your knowledge base today?"

        prompt_tokens = max(15, len(query.split()) + len(str(execution_result)[:200].split()))
        completion_tokens = max(20, len(final_answer.split()))
        span.record_tokens(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        span.end(output_data={"final_answer_length": len(final_answer), "citations_count": len(citations)})

        telemetry.update({
            "trace_id": trace.trace_id,
            "route": intent,
            "strategy_used": strategy if intent == "STRUCTURED_QUERY" else ("hybrid_rag" if intent == "UNSTRUCTURED_QUERY" else None),
            "prompt_tokens": telemetry.get("prompt_tokens", 0) + prompt_tokens,
            "completion_tokens": telemetry.get("completion_tokens", 0) + completion_tokens,
            "total_tokens": telemetry.get("total_tokens", 0) + prompt_tokens + completion_tokens,
            "latency_ms": round(trace.latency_ms, 2),
            "execution_success": execution_error is None,
        })

        return {
            "final_answer": final_answer,
            "citations": citations,
            "telemetry": telemetry,
        }
