"""
Streamlit Web UI for Multi-Agent Knowledge Base Q&A Platform.
Provides a 3-tab interactive web interface:
1. Ingestion Hub (Structured & Unstructured file upload, dataset catalog preview)
2. Conversational Q&A (LangGraph supervisor routing, engine selector, telemetry panels)
3. Benchmark Arena (Parallel 3-way Strategy A vs B vs C head-to-head comparison)
"""

import os
import uuid
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from frontend.client import BackendClient

# -----------------------------------------------------------------------------
# Streamlit App Configuration & Custom CSS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Knowledge Base Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2E3440;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #8892B0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #ECEFF4;
        margin-top: 4px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-greeting { background-color: #3B82F6; color: white; }
    .badge-ambiguous { background-color: #F59E0B; color: white; }
    .badge-structured { background-color: #10B981; color: white; }
    .badge-unstructured { background-color: #8B5CF6; color: white; }
    .badge-success { background-color: #10B981; color: white; }
    .badge-failed { background-color: #EF4444; color: white; }

    /* Strategy Columns in Benchmark */
    .benchmark-col-header {
        background-color: #2D3748;
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# State Initialization
# -----------------------------------------------------------------------------

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "benchmark_history" not in st.session_state:
        st.session_state["benchmark_history"] = []
    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = os.getenv("BACKEND_URL", "http://localhost:8000")


init_session_state()
client = BackendClient(base_url=st.session_state["backend_url"])


# -----------------------------------------------------------------------------
# Sidebar: System Status & Settings
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 System Control")
    
    # Backend Connectivity
    health_info = client.health()
    is_healthy = health_info.get("status") == "healthy"
    
    if is_healthy:
        st.success(f"🟢 Backend: Connected ({health_info.get('version', '0.1.0')})")
    else:
        st.error(f"🔴 Backend: Disconnected ({health_info.get('error', 'Unreachable')})")
        if st.button("🔄 Retry Connection"):
            st.rerun()

    st.divider()

    # Query Parameters
    st.subheader("⚙️ Query Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    rag_top_k = st.slider("RAG Top-K Chunks", min_value=1, max_value=15, value=5, step=1)
    
    st.divider()

    # Session Management
    st.subheader("🧵 Session")
    st.caption(f"Session ID: `{st.session_state['session_id'][:8]}...`")
    if st.button("🧹 Reset Session & Chat"):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.rerun()

    # Dataset Catalog Summary in Sidebar
    st.divider()
    datasets = client.list_datasets()
    struct_count = sum(1 for d in datasets if d.get("category") == "structured")
    unstruct_count = sum(1 for d in datasets if d.get("category") == "unstructured")
    
    st.subheader("📊 Catalog Stats")
    st.write(f"• **Structured Tables:** {struct_count}")
    st.write(f"• **Unstructured Docs:** {unstruct_count}")
    st.write(f"• **Total Datasets:** {len(datasets)}")


# -----------------------------------------------------------------------------
# Helper Renderers
# -----------------------------------------------------------------------------

def render_intent_badge(intent: str, confidence: float = 1.0, strategy: Optional[str] = None):
    """Render stylized HTML badge for supervisor routing intent."""
    badge_class = {
        "GREETING_OR_CHITCHAT": "badge-greeting",
        "AMBIGUOUS_QUERY": "badge-ambiguous",
        "STRUCTURED_QUERY": "badge-structured",
        "UNSTRUCTURED_QUERY": "badge-unstructured",
    }.get(intent, "badge-structured")

    label = intent.replace("_", " ")
    if strategy:
        label += f" → {strategy}"
    
    st.markdown(
        f'<span class="badge {badge_class}">Intent: {label} ({confidence:.0%})</span>',
        unsafe_allow_html=True,
    )


def render_telemetry_metrics(
    metrics: Dict[str, Any],
    token_usage: Dict[str, Any],
    trace_id: Optional[str] = None,
):
    """Render telemetry metrics in 4 neat metric columns."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lat = metrics.get("total_latency_ms", 0.0)
        st.metric("Latency", client.format_latency(lat))
    with c2:
        prompt_toks = token_usage.get("prompt_tokens", 0)
        compl_toks = token_usage.get("completion_tokens", 0)
        st.metric("Tokens (P / C)", f"{prompt_toks} / {compl_toks}")
    with c3:
        total_toks = token_usage.get("total_tokens", prompt_toks + compl_toks)
        st.metric("Total Tokens", f"{total_toks}")
    with c4:
        cost = token_usage.get("estimated_cost_usd", client.calculate_cost(prompt_toks, compl_toks))
        st.metric("Est. Cost", f"${cost:.5f}")

    if trace_id:
        st.caption(f"🔗 Trace ID: `{trace_id}`")


def render_model_thinking(thinking_data: Optional[Dict[str, Any]], default_expanded: bool = True):
    """Render interactive Model Thinking & Decision Choices step-by-step panel."""
    if not thinking_data:
        return

    steps = thinking_data.get("steps", [])
    summary = thinking_data.get("summary", "")

    if not steps and not summary:
        return

    with st.expander(f"🧠 Model Thinking & Decision Choices ({len(steps)} steps)", expanded=default_expanded):
        if summary:
            st.markdown(f"**Decision Summary:** *{summary}*")
            st.divider()

        for step in steps:
            step_num = step.get("step_number", 1)
            title = step.get("title", f"Step {step_num}")
            choice = step.get("choice", "")
            reasoning = step.get("reasoning", "")
            details = step.get("details", {})

            # Step title with category icon
            icon = "🔵" if step_num == 1 else ("🟢" if step_num == 2 else ("🛡️" if "Security" in title or "Guardrail" in title or "Check" in title else ("⚡" if "Execution" in title or "Query" in title else "✨")))
            st.markdown(f"##### {icon} Step {step_num}: {title}")
            
            if choice:
                st.info(f"**Model Choice:** {choice}")
            if reasoning:
                st.write(f"**Chain of Thought:** {reasoning}")
            
            # Supplementary details
            if details:
                if "retained_columns" in details and details["retained_columns"]:
                    st.caption("📋 **Retained Columns for Token Minimization:**")
                    for tbl, cols in details["retained_columns"].items():
                        st.write(f"- `{tbl}`: `{', '.join(cols)}`")
                elif "sql" in details:
                    st.caption("Formulated SQL:")
                    st.code(details["sql"], language="sql")
                elif "code" in details:
                    st.caption("Formulated Code:")
                    st.code(details["code"], language="python")
                elif "violations" in details and details["violations"]:
                    st.warning(f"⚠️ Security Violations Detected: {', '.join(details['violations'])}")
            
            st.markdown("---")


# -----------------------------------------------------------------------------
# Main Tabs Layout
# -----------------------------------------------------------------------------

tab_ingest, tab_qa, tab_benchmark = st.tabs([
    "📥 Ingestion Hub",
    "💬 Conversational Q&A",
    "⚔️ Benchmark Arena",
])


# =============================================================================
# TAB 1: INGESTION HUB
# =============================================================================

with tab_ingest:
    st.header("📥 Multi-Strategy Ingestion Hub")
    st.write(
        "Upload structured data (CSV, Parquet, Excel) for automated PostgreSQL table creation "
        "and DuckDB/Pandas indexing, or unstructured data (PDF, DOCX, TXT, MD) for hybrid dense + BM25 retrieval."
    )

    # Top KPI Metrics Cards
    all_datasets = client.list_datasets()
    struct_ds = [d for d in all_datasets if d.get("category") == "structured"]
    unstruct_ds = [d for d in all_datasets if d.get("category") == "unstructured"]
    total_rows = sum(d.get("row_count") or 0 for d in all_datasets)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Total Datasets</div>'
            f'<div class="metric-value">{len(all_datasets)}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Structured Tables</div>'
            f'<div class="metric-value">{len(struct_ds)}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Unstructured Documents</div>'
            f'<div class="metric-value">{len(unstruct_ds)}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Total Rows / Chunks</div>'
            f'<div class="metric-value">{total_rows:,}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Upload Section
    u_col1, u_col2 = st.columns([2, 1])
    with u_col1:
        uploaded_file = st.file_uploader(
            "Upload Dataset File",
            type=["csv", "tsv", "parquet", "pq", "xlsx", "xls", "pdf", "docx", "doc", "txt", "md", "markdown"],
            help="Supports CSV, Parquet, Excel, PDF, DOCX, TXT, MD",
        )
    with u_col2:
        disp_name = st.text_input("Display Name (optional)", placeholder="e.g., Q3 Sales Report")
        desc = st.text_area("Description (optional)", placeholder="e.g., Regional sales transactions for 2023", height=70)

    if uploaded_file is not None:
        if st.button("🚀 Ingest File", type="primary"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                file_bytes = uploaded_file.getvalue()
                res = client.ingest_file(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    display_name=disp_name or uploaded_file.name,
                    description=desc,
                )
                if "error" in res:
                    st.error(f"❌ Ingestion Failed: {res['error']}")
                else:
                    st.success(
                        f"✅ Successfully ingested '{res.get('name')}' ({res.get('category')})! "
                        f"ID: `{res.get('dataset_id')}` | Rows/Chunks: {res.get('row_count')}"
                    )
                    st.rerun()

    st.divider()

    # Dataset Catalog Listing
    st.subheader("📚 Ingested Dataset Catalog")
    cat_filter = st.radio("Filter Category:", ["All", "Structured", "Unstructured"], horizontal=True)
    
    filtered_datasets = all_datasets
    if cat_filter == "Structured":
        filtered_datasets = struct_ds
    elif cat_filter == "Unstructured":
        filtered_datasets = unstruct_ds

    if filtered_datasets:
        cat_df = pd.DataFrame([
            {
                "ID": d.get("id", "")[:8] + "...",
                "Full ID": d.get("id", ""),
                "Name": d.get("name", ""),
                "Category": d.get("category", ""),
                "File Type": d.get("file_type", ""),
                "Rows / Chunks": d.get("row_count", 0),
                "Table Name": d.get("table_name", "N/A"),
                "Created At": str(d.get("created_at", ""))[:19],
            }
            for d in filtered_datasets
        ])
        
        st.dataframe(
            cat_df[["ID", "Name", "Category", "File Type", "Rows / Chunks", "Table Name", "Created At"]],
            use_container_width=True,
            hide_index=True,
        )

        # Dataset Inspector drill-down
        st.subheader("🔍 Dataset Inspector")
        ds_options = {f"{d.get('name')} ({d.get('id', '')[:8]})": d for d in filtered_datasets}
        selected_ds_name = st.selectbox("Select Dataset to Inspect:", list(ds_options.keys()))
        
        if selected_ds_name:
            selected_ds = ds_options[selected_ds_name]
            st.json(selected_ds)
    else:
        st.info("No datasets found matching filter. Upload a file above to begin!")


# =============================================================================
# TAB 2: CONVERSATIONAL Q&A
# =============================================================================

with tab_qa:
    st.header("💬 Conversational Multi-Agent Q&A")
    st.write(
        "Ask natural language questions across your data catalog. "
        "The LangGraph Supervisor automatically classifies intent, selects optimal execution strategies, "
        "and handles greetings, clarifications, Text2SQL, sandboxed Python, or hybrid RAG."
    )

    # Engine Selector & Dataset Filter Controls
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        engine_mode = st.selectbox(
            "Execution Engine Selector:",
            [
                "Auto Router (LangGraph Supervisor)",
                "Strategy A: Dedicated PostgreSQL DB",
                "Strategy B: In-Memory DuckDB",
                "Strategy C: Sandboxed Python Pandas",
                "Unstructured Hybrid RAG",
            ],
            index=0,
            help="Select Auto Router for intent classification or pick a specific execution engine.",
        )
    with ctrl_col2:
        dataset_choices = {f"{d.get('name')} ({d.get('category')})": d.get("id") for d in all_datasets}
        selected_ds_labels = st.multiselect(
            "Scope Specific Datasets (Optional):",
            options=list(dataset_choices.keys()),
            help="Restrict query to selected datasets",
        )
        target_dataset_ids = [dataset_choices[k] for k in selected_ds_labels] if selected_ds_labels else None

    st.divider()

    # Render Chat History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                # Assistant Response Components
                if msg.get("intent"):
                    render_intent_badge(
                        intent=msg["intent"],
                        confidence=msg.get("confidence", 1.0),
                        strategy=msg.get("suggested_strategy"),
                    )

                # Transparent Model Thinking & Decision Choices Panel
                if msg.get("thinking_process"):
                    render_model_thinking(msg["thinking_process"], default_expanded=True)

                st.markdown(msg["content"])

                # Clarification suggestion buttons if ambiguous
                if msg.get("clarification_message") and msg.get("candidate_datasets"):
                    st.info(f"💡 Suggestion: Consider querying datasets: {', '.join(msg['candidate_datasets'])}")

                # Expandable Panels
                if msg.get("generated_code"):
                    code_lang = "python" if "pandas" in str(msg.get("suggested_strategy", "")).lower() else "sql"
                    with st.expander("💻 Generated Query / Code", expanded=False):
                        st.code(msg["generated_code"], language=code_lang)

                if msg.get("tabular_result"):
                    tab_df = client.tabular_result_to_dataframe(msg["tabular_result"])
                    if not tab_df.empty:
                        with st.expander("📊 Tabular Result", expanded=True):
                            st.dataframe(tab_df, use_container_width=True)
                            if msg["tabular_result"].get("truncated"):
                                st.caption("⚠️ Result capped by LIMIT 20 safety directive.")

                if msg.get("citations"):
                    with st.expander("📑 Document Citations", expanded=False):
                        for cit in msg["citations"]:
                            if isinstance(cit, dict):
                                doc = cit.get("document_name", "Document")
                                page = f" (Page {cit.get('page_number')})" if cit.get("page_number") else ""
                                score = f" [Score: {cit.get('similarity_score', 0.0):.2f}]" if cit.get("similarity_score") else ""
                                snippet = cit.get("snippet", "")
                                st.markdown(f"- **{doc}**{page}{score}: {snippet}")
                            else:
                                st.markdown(f"- {cit}")

                if msg.get("metrics") or msg.get("token_usage"):
                    with st.expander("⚡ Telemetry & Cost", expanded=False):
                        render_telemetry_metrics(
                            metrics=msg.get("metrics", {}),
                            token_usage=msg.get("token_usage", {}),
                            trace_id=msg.get("trace_id"),
                        )

    # Chat Input Box
    user_prompt = st.chat_input("Ask a question about your structured or unstructured data...")

    if user_prompt:
        # Append and display user message
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Execute Query via Client
        with st.chat_message("assistant"):
            with st.spinner("Processing query..."):
                resp_payload: Dict[str, Any] = {}

                if engine_mode == "Auto Router (LangGraph Supervisor)":
                    resp_payload = client.query_agent(
                        query=user_prompt,
                        session_id=st.session_state["session_id"],
                        dataset_ids=target_dataset_ids,
                        temperature=temperature,
                    )
                elif engine_mode == "Strategy A: Dedicated PostgreSQL DB":
                    resp = client.query_dedicated_db(user_prompt, dataset_ids=target_dataset_ids, temperature=temperature)
                    resp_payload = {
                        "content": resp.get("answer", ""),
                        "generated_code": resp.get("sql_query"),
                        "tabular_result": resp.get("tabular_result"),
                        "thinking_process": resp.get("thinking_process"),
                        "metrics": resp.get("metrics", {}),
                        "token_usage": resp.get("token_usage", {}),
                        "suggested_strategy": "dedicated_db",
                        "intent": "STRUCTURED_QUERY",
                    }
                elif engine_mode == "Strategy B: In-Memory DuckDB":
                    resp = client.query_duckdb(user_prompt, dataset_ids=target_dataset_ids, temperature=temperature)
                    resp_payload = {
                        "content": resp.get("answer", ""),
                        "generated_code": resp.get("sql_query"),
                        "tabular_result": resp.get("tabular_result"),
                        "thinking_process": resp.get("thinking_process"),
                        "metrics": resp.get("metrics", {}),
                        "token_usage": resp.get("token_usage", {}),
                        "suggested_strategy": "duckdb",
                        "intent": "STRUCTURED_QUERY",
                    }
                elif engine_mode == "Strategy C: Sandboxed Python Pandas":
                    resp = client.query_pandas_sandbox(user_prompt, dataset_ids=target_dataset_ids, temperature=temperature)
                    resp_payload = {
                        "content": resp.get("answer", ""),
                        "generated_code": resp.get("python_code"),
                        "tabular_result": resp.get("tabular_result"),
                        "thinking_process": resp.get("thinking_process"),
                        "metrics": resp.get("metrics", {}),
                        "token_usage": resp.get("token_usage", {}),
                        "suggested_strategy": "pandas_sandbox",
                        "intent": "STRUCTURED_QUERY",
                    }
                elif engine_mode == "Unstructured Hybrid RAG":
                    resp = client.query_unstructured_rag(
                        user_prompt,
                        top_k=rag_top_k,
                        dataset_ids=target_dataset_ids,
                        temperature=temperature,
                    )
                    resp_payload = {
                        "content": resp.get("answer", ""),
                        "citations": resp.get("citations", []),
                        "thinking_process": resp.get("thinking_process"),
                        "metrics": resp.get("metrics", {}),
                        "token_usage": resp.get("token_usage", {}),
                        "intent": "UNSTRUCTURED_QUERY",
                    }

                # Construct assistant message record
                assistant_msg = {
                    "role": "assistant",
                    "content": resp_payload.get("answer") or resp_payload.get("content") or "No response returned.",
                    "intent": resp_payload.get("intent"),
                    "confidence": resp_payload.get("confidence", 1.0),
                    "routing_reason": resp_payload.get("routing_reason"),
                    "suggested_strategy": resp_payload.get("suggested_strategy"),
                    "generated_code": resp_payload.get("generated_code"),
                    "tabular_result": resp_payload.get("tabular_result"),
                    "citations": resp_payload.get("citations", []),
                    "clarification_message": resp_payload.get("clarification_message"),
                    "candidate_datasets": resp_payload.get("candidate_datasets", []),
                    "thinking_process": resp_payload.get("thinking_process"),
                    "metrics": resp_payload.get("metrics", {}),
                    "token_usage": resp_payload.get("token_usage", {}),
                    "trace_id": (resp_payload.get("telemetry") or {}).get("trace_id"),
                }

                st.session_state["messages"].append(assistant_msg)
                st.rerun()


# =============================================================================
# TAB 3: BENCHMARK ARENA
# =============================================================================

with tab_benchmark:
    st.header("⚔️ 3-Way Structured Benchmark Arena")
    st.write(
        "Concurrently benchmark **Strategy A (Dedicated PostgreSQL)**, "
        "**Strategy B (In-Memory DuckDB)**, and **Strategy C (Sandboxed Pandas)** on the exact same natural language query. "
        "Evaluates execution equivalence, latency breakdown, token efficiency, and sandbox security."
    )

    # Benchmark Query Input & Presets
    b_col1, b_col2 = st.columns([3, 1])
    preset_queries = [
        "What is the total revenue and count of transactions?",
        "Show top 5 customers ordered by total purchase amount",
        "Calculate the average order value grouped by product category",
        "Count orders per month with status COMPLETED",
        "Custom query...",
    ]
    with b_col2:
        selected_preset = st.selectbox("Preset Query Template:", preset_queries, index=0)
    
    with b_col1:
        default_q = "" if selected_preset == "Custom query..." else selected_preset
        benchmark_query = st.text_area(
            "Benchmark Query:",
            value=default_q,
            placeholder="Enter a complex analytical query to test across all 3 engines...",
            height=70,
        )

    if st.button("🚀 Run 3-Way Head-to-Head Benchmark", type="primary"):
        if not benchmark_query.strip():
            st.warning("Please enter a benchmark query to execute.")
        else:
            with st.spinner("Executing Strategy A, B, and C in parallel..."):
                bench_res = client.query_benchmark(
                    query=benchmark_query,
                    include_raw_data=True,
                    temperature=temperature,
                )

                if "error" in bench_res and not bench_res.get("strategy_a"):
                    st.error(f"❌ Benchmark execution failed: {bench_res['error']}")
                else:
                    st.session_state["benchmark_history"].insert(0, bench_res)

    # Render Latest Benchmark Result
    if st.session_state["benchmark_history"]:
        latest_bench = st.session_state["benchmark_history"][0]
        st.divider()

        # Summary Header & Consensus Status
        summary = latest_bench.get("benchmark_summary", {})
        consensus = summary.get("consensus_reached", False)
        fastest = summary.get("fastest_strategy", "N/A")
        token_winner = summary.get("most_token_efficient_strategy", "N/A")
        total_lat = latest_bench.get("total_arena_latency_ms", 0.0)

        c_cons, c_fast, c_tok, c_lat = st.columns(4)
        with c_cons:
            if consensus:
                st.success("✅ **Consensus: EQUIVALENT**")
            else:
                st.warning("⚠️ **Consensus: DISCREPANCY**")
        with c_fast:
            st.info(f"⚡ **Fastest:** {fastest}")
        with c_tok:
            st.info(f"🪙 **Token Efficient:** {token_winner}")
        with c_lat:
            st.metric("Total Wall-Clock Latency", client.format_latency(total_lat))

        if summary.get("summary_analysis"):
            st.markdown(f"> 📝 *{summary.get('summary_analysis')}*")

        st.subheader("📊 Side-by-Side Head-to-Head Results")

        # 3-Column Arena Display
        col_a, col_b, col_c = st.columns(3)

        strategies = [
            (col_a, "Strategy A (PostgreSQL)", latest_bench.get("strategy_a", {}), "sql"),
            (col_b, "Strategy B (DuckDB)", latest_bench.get("strategy_b", {}), "sql"),
            (col_c, "Strategy C (Pandas Sandbox)", latest_bench.get("strategy_c", {}), "python"),
        ]

        chart_data = []

        for col, title, data, lang in strategies:
            with col:
                st.markdown(f'<div class="benchmark-col-header">{title}</div>', unsafe_allow_html=True)
                status = data.get("status", "UNKNOWN")
                if status == "SUCCESS":
                    st.markdown('<span class="badge badge-success">Status: SUCCESS</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-failed">Status: FAILED</span>', unsafe_allow_html=True)
                    if data.get("error"):
                        st.caption(f"Error: {data.get('error')}")

                # Metrics
                met = data.get("metrics", {})
                tok = data.get("token_usage", {})
                lat_ms = met.get("total_latency_ms", 0.0)
                tot_toks = tok.get("total_tokens", 0)
                
                chart_data.append({
                    "Strategy": title.split()[1],
                    "Latency (ms)": lat_ms,
                    "Total Tokens": tot_toks,
                    "Cost ($)": tok.get("estimated_cost_usd", client.calculate_cost(tok.get("prompt_tokens", 0), tok.get("completion_tokens", 0))),
                })

                st.write(f"⏱️ **Latency:** {client.format_latency(lat_ms)}")
                st.write(f"🪙 **Tokens:** {tot_toks} ({tok.get('prompt_tokens', 0)} / {tok.get('completion_tokens', 0)})")
                
                # Model Thinking & Choices
                if data.get("thinking_process"):
                    render_model_thinking(data["thinking_process"], default_expanded=False)

                # Answer
                if data.get("answer"):
                    st.markdown(f"**Answer:** {data.get('answer')}")

                # Code Generated
                if data.get("code_generated"):
                    st.markdown("**Generated Code:**")
                    st.code(data.get("code_generated"), language=lang)

                # Tabular Result Preview
                if data.get("tabular_result"):
                    df_res = client.tabular_result_to_dataframe(data.get("tabular_result"))
                    if not df_res.empty:
                        st.markdown("**Result Table:**")
                        st.dataframe(df_res.head(10), use_container_width=True)

        # Comparative Charts Section
        st.divider()
        st.subheader("📈 Comparative Telemetry Breakdown")
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data).set_index("Strategy")
            ch_col1, ch_col2 = st.columns(2)
            with ch_col1:
                st.markdown("**⏱️ Latency Comparison (ms)**")
                st.bar_chart(chart_df["Latency (ms)"])
            with ch_col2:
                st.markdown("**🪙 Token Consumption Comparison**")
                st.bar_chart(chart_df["Total Tokens"])
