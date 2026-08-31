"""
LangGraph StateGraph compilers.

Two graphs are built from the same node registry:

- `build_single_pass_graph` is acyclic and generates once. Benchmark and
  evaluation runs use it, so their token and latency numbers stay comparable.
- `build_multi_agent_graph` adds the exploration and self-correction cycle.

Intent branches in both: GREETING_OR_CHITCHAT, AMBIGUOUS_QUERY,
STRUCTURED_QUERY (the structured chain below), UNSTRUCTURED_QUERY.
"""

import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes.chitchat_node import chitchat_node
from src.agent.nodes.clarify_node import clarify_node
from src.agent.nodes.loop import (
    code_executor_node,
    code_generator_node,
    code_validator_node,
    escalation_node,
    explorer_planner_node,
    has_probes,
    has_schema,
    is_valid,
    reflector_node,
    route_after_reflect,
    schema_retriever_node,
    tool_executor_node,
)
from src.agent.nodes.projection_critic import projection_critic_node
from src.agent.nodes.router_node import supervisor_router_node
from src.agent.nodes.synthesizer_node import synthesizer_node
from src.agent.nodes.unstructured_node import unstructured_node
from src.agent.state import AgentState
from src.config import get_settings

INTENT_ROUTES = {
    "GREETING_OR_CHITCHAT": "chitchat",
    "AMBIGUOUS_QUERY": "clarification",
    "STRUCTURED_QUERY": "schema_retriever",
    "UNSTRUCTURED_QUERY": "unstructured_agent",
}


def route_intent(state: AgentState) -> str:
    """Conditional edge router determining the next node from supervisor classification."""
    return INTENT_ROUTES.get(state.get("intent") or "", "chitchat")


def _register_shared_nodes(workflow: StateGraph) -> None:
    workflow.add_node("supervisor_router", supervisor_router_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("clarification", clarify_node)
    workflow.add_node("schema_retriever", schema_retriever_node)
    workflow.add_node("code_generator", code_generator_node)
    workflow.add_node("code_validator", code_validator_node)
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("projection_critic", projection_critic_node)
    workflow.add_node("unstructured_agent", unstructured_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("supervisor_router")
    # route_intent returns node names, so the path map is keyed on those.
    workflow.add_conditional_edges(
        "supervisor_router",
        route_intent,
        {node: node for node in INTENT_ROUTES.values()},
    )

    workflow.add_edge("chitchat", END)
    workflow.add_edge("clarification", END)
    workflow.add_edge("code_generator", "code_validator")
    workflow.add_edge("unstructured_agent", "synthesizer")
    workflow.add_edge("projection_critic", "synthesizer")
    workflow.add_edge("synthesizer", END)


def build_single_pass_graph() -> Any:
    """Acyclic graph: one generation, no probes, no retries."""
    workflow = StateGraph(AgentState)
    _register_shared_nodes(workflow)

    workflow.add_conditional_edges(
        "schema_retriever",
        has_schema,
        {"generate": "code_generator", "no_schema": "clarification"},
    )
    workflow.add_conditional_edges(
        "code_validator",
        is_valid,
        {"execute": "code_executor", "reflect": "projection_critic"},
    )
    workflow.add_edge("code_executor", "projection_critic")

    return workflow.compile()


def build_multi_agent_graph() -> Any:
    """Graph with the bounded exploration and self-correction cycle."""
    workflow = StateGraph(AgentState)
    _register_shared_nodes(workflow)

    workflow.add_node("explorer_planner", explorer_planner_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("escalation", escalation_node)

    workflow.add_conditional_edges(
        "schema_retriever",
        has_schema,
        {"generate": "explorer_planner", "no_schema": "clarification"},
    )
    workflow.add_conditional_edges(
        "explorer_planner",
        has_probes,
        {"probe": "tool_executor", "generate": "code_generator"},
    )
    workflow.add_edge("tool_executor", "code_generator")
    workflow.add_conditional_edges(
        "code_validator",
        is_valid,
        {"execute": "code_executor", "reflect": "reflector"},
    )
    workflow.add_edge("code_executor", "reflector")
    workflow.add_conditional_edges(
        "reflector",
        route_after_reflect,
        {
            "continue": "projection_critic",
            "retry": "explorer_planner",
            "escalate": "escalation",
        },
    )
    workflow.add_edge("escalation", END)

    return workflow.compile()


_compiled_graphs: Dict[bool, Any] = {}


def get_agent_graph(agentic: Optional[bool] = None) -> Any:
    """Compiled graph singleton, keyed on whether the correction loop is on."""
    use_loop = get_settings().structured_loop_enabled if agentic is None else agentic
    if use_loop not in _compiled_graphs:
        _compiled_graphs[use_loop] = (
            build_multi_agent_graph() if use_loop else build_single_pass_graph()
        )
    return _compiled_graphs[use_loop]


def initial_state(
    query: str,
    session_id: Optional[str] = None,
    suggested_strategy: Optional[str] = None,
    dataset_ids: Optional[List[str]] = None,
) -> AgentState:
    """Fresh state for one request."""
    return {
        "query": query,
        "session_id": session_id or str(uuid.uuid4()),
        "messages": [],
        "intent": None,
        "confidence": 0.0,
        "routing_reason": "",
        "suggested_strategy": suggested_strategy,
        "candidate_datasets": dataset_ids or [],
        "pruned_tables": {},
        "schema_ddl": "",
        "retrieved_chunks": [],
        "generated_code": None,
        "execution_result": [],
        "execution_columns": [],
        "execution_error": None,
        "observations": [],
        "loop_iterations": 0,
        "probe_plan": [],
        "reflection_class": None,
        "clarification_message": None,
        "final_answer": None,
        "citations": [],
        "telemetry": {},
    }


def run_agent(
    query: str,
    session_id: Optional[str] = None,
    suggested_strategy: Optional[str] = None,
    dataset_ids: Optional[List[str]] = None,
    agentic: Optional[bool] = None,
) -> AgentState:
    """
    Execute the multi-agent graph on a user query.

    The retry budget is enforced by a conditional edge on `loop_iterations`, so a
    spent budget escalates to the user. `recursion_limit` is only a catastrophic
    breaker and should never be what stops a run.
    """
    settings = get_settings()
    graph = get_agent_graph(agentic)
    config = {"recursion_limit": max(25, settings.structured_loop_max_iters * 3 * 6)}

    return graph.invoke(
        initial_state(query, session_id, suggested_strategy, dataset_ids),
        config=config,
    )
