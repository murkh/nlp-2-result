"""
LangGraph Multi-Agent StateGraph Workflow Compiler.
Defines the multi-agent graph with supervisor routing across 4 intent branches:
1. GREETING_OR_CHITCHAT -> chitchat_node -> END
2. AMBIGUOUS_QUERY -> clarify_node -> END
3. STRUCTURED_QUERY -> structured_node -> synthesizer_node -> END
4. UNSTRUCTURED_QUERY -> unstructured_node -> synthesizer_node -> END
"""

import uuid
from typing import Any, List, Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes.chitchat_node import chitchat_node
from src.agent.nodes.clarify_node import clarify_node
from src.agent.nodes.router_node import supervisor_router_node
from src.agent.nodes.structured_node import structured_node
from src.agent.nodes.synthesizer_node import synthesizer_node
from src.agent.nodes.unstructured_node import unstructured_node
from src.agent.state import AgentState


def route_intent(state: AgentState) -> str:
    """
    Conditional edge router determining the next node from supervisor classification.
    """
    intent = state.get("intent")
    if intent == "GREETING_OR_CHITCHAT":
        return "chitchat"
    elif intent == "AMBIGUOUS_QUERY":
        return "clarification"
    elif intent == "STRUCTURED_QUERY":
        return "structured_agent"
    elif intent == "UNSTRUCTURED_QUERY":
        return "unstructured_agent"
    return "chitchat"


def build_multi_agent_graph() -> Any:
    """
    Build and compile the multi-agent orchestration state graph.
    """
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("supervisor_router", supervisor_router_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("clarification", clarify_node)
    workflow.add_node("structured_agent", structured_node)
    workflow.add_node("unstructured_agent", unstructured_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Set Entry Point
    workflow.set_entry_point("supervisor_router")

    # Add Supervisor Conditional Edges
    workflow.add_conditional_edges(
        "supervisor_router",
        route_intent,
        {
            "chitchat": "chitchat",
            "clarification": "clarification",
            "structured_agent": "structured_agent",
            "unstructured_agent": "unstructured_agent",
        },
    )

    # Add Execution Edges
    workflow.add_edge("chitchat", END)
    workflow.add_edge("clarification", END)
    workflow.add_edge("structured_agent", "synthesizer")
    workflow.add_edge("unstructured_agent", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# Graph Singleton
_compiled_graph: Optional[Any] = None


def get_agent_graph() -> Any:
    """Retrieve compiled multi-agent state graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_multi_agent_graph()
    return _compiled_graph


def run_agent(
    query: str,
    session_id: Optional[str] = None,
    suggested_strategy: Optional[str] = None,
    dataset_ids: Optional[List[str]] = None,
) -> AgentState:
    """
    High-level entrypoint to execute the multi-agent graph on a user query.
    """
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id or str(uuid.uuid4()),
        "messages": [],
        "intent": None,
        "confidence": 0.0,
        "routing_reason": "",
        "suggested_strategy": suggested_strategy,
        "candidate_datasets": dataset_ids or [],
        "pruned_tables": [],
        "retrieved_chunks": [],
        "generated_code": None,
        "execution_result": [],
        "execution_columns": [],
        "execution_error": None,
        "clarification_message": None,
        "final_answer": None,
        "citations": [],
        "telemetry": {},
    }

    graph = get_agent_graph()
    return graph.invoke(initial_state)
