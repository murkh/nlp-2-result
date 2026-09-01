"""
Streamlit Web UI for Multi-Agent Knowledge Base Q&A Platform.
Provides a 2-tab interactive web interface:
1. Ingestion Hub (Structured & Unstructured file upload, dataset catalog preview)
2. Conversational Q&A (LangGraph supervisor routing, engine selector, telemetry panels)
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

    with st.expander(
        f"🧠 Model Thinking & Decision Choices ({len(steps)} steps)", expanded=default_expanded
    ):
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
            icon = (
                "🔵"
                if step_num == 1
                else (
                    "🟢"
                    if step_num == 2
                    else (
                        "🛡️"
                        if "Security" in title or "Guardrail" in title or "Check" in title
                        else ("⚡" if "Execution" in title or "Query" in title else "✨")
                    )
                )
            )
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
                    st.warning(
                        f"⚠️ Security Violations Detected: {', '.join(details['violations'])}"
                    )

            st.markdown("---")


# -----------------------------------------------------------------------------
# Main Tabs Layout
# -----------------------------------------------------------------------------

tab_ingest, tab_qa = st.tabs(
    [
        "📥 Ingestion Hub",
        "💬 Conversational Q&A",
    ]
)


# =============================================================================
# TAB 1: INGESTION HUB
# =============================================================================

with tab_ingest:
    st.header("📥 Ingestion Hub")
    st.write(
        "Upload structured data (CSV, Parquet, Excel) for Pandas-ready blob storage and metadata "
        "indexing, or unstructured data (PDF, DOCX, TXT, MD) for hybrid dense + BM25 retrieval."
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
        uploaded_files = st.file_uploader(
            "Upload Dataset Files",
            type=[
                "csv",
                "tsv",
                "parquet",
                "pq",
                "xlsx",
                "xls",
                "pdf",
                "docx",
                "doc",
                "txt",
                "md",
                "markdown",
            ],
            help="Supports CSV, Parquet, Excel, PDF, DOCX, TXT, MD. Select multiple files to ingest them in one batch.",
            accept_multiple_files=True,
        )
    single_upload = len(uploaded_files) == 1
    with u_col2:
        if single_upload:
            disp_name = st.text_input(
                "Display Name (optional)", placeholder="e.g., Q3 Sales Report"
            )
            desc = st.text_area(
                "Description (optional)",
                placeholder="e.g., Regional sales transactions for 2023",
                height=70,
            )
        else:
            disp_name, desc = "", ""
            st.caption(
                "Display Name / Description apply to single-file uploads. "
                "In a batch, each dataset is named after its filename."
            )

    if uploaded_files:
        if st.button(f"🚀 Ingest {len(uploaded_files)} File(s)", type="primary"):
            results = []
            bar = st.progress(0.0)
            for idx, uf in enumerate(uploaded_files):
                bar.progress(idx / len(uploaded_files), text=f"Ingesting {uf.name}...")
                res = client.ingest_file(
                    file_bytes=uf.getvalue(),
                    filename=uf.name,
                    display_name=(disp_name or uf.name) if single_upload else uf.name,
                    description=desc if single_upload else None,
                )
                results.append(
                    {
                        "File": uf.name,
                        "Status": "❌ Failed" if "error" in res else "✅ Ingested",
                        "Dataset ID": res.get("dataset_id", ""),
                        "Category": res.get("category", ""),
                        "Rows/Chunks": res.get("row_count", ""),
                        "Error": res.get("error", ""),
                    }
                )
            bar.empty()

            succeeded = sum(1 for r in results if r["Status"].endswith("Ingested"))
            failed = len(results) - succeeded
            (st.success if failed == 0 else st.warning)(
                f"Ingestion complete: {succeeded} succeeded, {failed} failed."
            )
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            if failed == 0:
                st.rerun()

    st.divider()

    # Dataset Catalog Listing
    st.subheader("📚 Ingested Dataset Catalog")
    cat_filter = st.radio(
        "Filter Category:", ["All", "Structured", "Unstructured"], horizontal=True
    )

    filtered_datasets = all_datasets
    if cat_filter == "Structured":
        filtered_datasets = struct_ds
    elif cat_filter == "Unstructured":
        filtered_datasets = unstruct_ds

    if filtered_datasets:
        cat_df = pd.DataFrame(
            [
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
            ]
        )

        st.dataframe(
            cat_df[
                ["ID", "Name", "Category", "File Type", "Rows / Chunks", "Table Name", "Created At"]
            ],
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
        "The LangGraph Supervisor automatically classifies intent and handles greetings, "
        "clarifications, sandboxed Python analysis, or hybrid RAG."
    )

    # Engine Selector & Dataset Filter Controls
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        engine_mode = st.selectbox(
            "Execution Engine Selector:",
            [
                "Auto Router (LangGraph Supervisor)",
                "Sandboxed Python Pandas",
                "Unstructured Hybrid RAG",
            ],
            index=0,
            help="Select Auto Router for intent classification or pick a specific execution engine.",
        )
    with ctrl_col2:
        dataset_choices = {
            f"{d.get('name')} ({d.get('category')})": d.get("id") for d in all_datasets
        }
        selected_ds_labels = st.multiselect(
            "Scope Specific Datasets (Optional):",
            options=list(dataset_choices.keys()),
            help="Restrict query to selected datasets",
        )
        target_dataset_ids = (
            [dataset_choices[k] for k in selected_ds_labels] if selected_ds_labels else None
        )

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
                    st.info(
                        f"💡 Suggestion: Consider querying datasets: {', '.join(msg['candidate_datasets'])}"
                    )

                # Expandable Panels
                if msg.get("generated_code"):
                    with st.expander("💻 Generated Query / Code", expanded=False):
                        st.code(msg["generated_code"], language="python")

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
                                page = (
                                    f" (Page {cit.get('page_number')})"
                                    if cit.get("page_number")
                                    else ""
                                )
                                score = (
                                    f" [Score: {cit.get('similarity_score', 0.0):.2f}]"
                                    if cit.get("similarity_score")
                                    else ""
                                )
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
                elif engine_mode == "Sandboxed Python Pandas":
                    resp = client.query_pandas_sandbox(
                        user_prompt, dataset_ids=target_dataset_ids, temperature=temperature
                    )
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
                    "content": resp_payload.get("answer")
                    or resp_payload.get("content")
                    or "No response returned.",
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
