"""
LangGraph Multi-Agent StateGraph Workflow Compiler.
Defines the multi-agent graph with supervisor routing across 4 intent branches:
1. GREETING_OR_CHITCHAT -> chitchat_node -> END
2. AMBIGUOUS_QUERY -> clarify_node -> END
3. STRUCTURED_QUERY -> structured_node -> synthesizer_node -> END
4. UNSTRUCTURED_QUERY -> unstructured_node -> synthesizer_node -> END
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from src.agent.nodes.chitchat_node import chitchat_node
from src.agent.nodes.clarify_node import clarify_node
from src.agent.nodes.router_node import supervisor_router_node
from src.agent.nodes.structured_node import structured_node
from src.agent.nodes.synthesizer_node import synthesizer_node
from src.agent.nodes.unstructured_node import unstructured_node
from src.agent.state import AgentState

try:
    from langgraph.graph import END, StateGraph
    _has_langgraph = True
except ImportError:
    _has_langgraph = False
    END = "__end__"


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


class LocalCompiledGraph:
    """
    Lightweight, robust in-memory compiled state graph.
    Provides identical execution semantics to LangGraph's CompiledGraph
    when running in environments without langgraph installed.
    """

    def __init__(
        self,
        nodes: Dict[str, Callable[[AgentState], Dict[str, Any]]],
        entry_point: str,
        edges: Dict[str, str],
        conditional_edges: Dict[str, Tuple[Callable[[AgentState], str], Dict[str, str]]],
    ):
        self.nodes = nodes
        self.entry_point = entry_point
        self.edges = edges
        self.conditional_edges = conditional_edges

    def invoke(self, input_state: AgentState) -> AgentState:
        """Execute the state graph workflow synchronously."""
        state = dict(input_state)
        current_node = self.entry_point

        max_steps = 20
        step_count = 0

        while current_node and current_node not in (END, "__end__") and step_count < max_steps:
            step_count += 1
            if current_node not in self.nodes:
                break

            node_func = self.nodes[current_node]
            node_output = node_func(state)
            if isinstance(node_output, dict):
                state.update(node_output)

            # Determine next node
            if current_node in self.conditional_edges:
                router_fn, path_map = self.conditional_edges[current_node]
                route_key = router_fn(state)
                current_node = path_map.get(route_key, END)
            elif current_node in self.edges:
                current_node = self.edges[current_node]
            else:
                break

        return state

    async def ainvoke(self, input_state: AgentState) -> AgentState:
        """Async invocation support."""
        return self.invoke(input_state)


class LocalStateGraph:
    """Fallback StateGraph builder mirroring LangGraph API."""

    def __init__(self, state_schema: Any = None):
        self.state_schema = state_schema
        self.nodes: Dict[str, Callable[[AgentState], Dict[str, Any]]] = {}
        self.entry_point: Optional[str] = None
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Tuple[Callable[[AgentState], str], Dict[str, str]]] = {}

    def add_node(self, name: str, func: Callable[[AgentState], Dict[str, Any]]):
        self.nodes[name] = func

    def set_entry_point(self, name: str):
        self.entry_point = name

    def add_edge(self, start_node: str, end_node: str):
        self.edges[start_node] = end_node

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[[AgentState], str],
        path_map: Dict[str, str],
    ):
        self.conditional_edges[source] = (path, path_map)

    def compile(self) -> LocalCompiledGraph:
        if not self.entry_point:
            raise ValueError("Entry point must be set before compiling StateGraph")
        return LocalCompiledGraph(
            nodes=self.nodes,
            entry_point=self.entry_point,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
        )


def build_multi_agent_graph() -> Any:
    """
    Build and compile the multi-agent orchestration state graph.
    Uses native LangGraph when installed, with seamless local fallback.
    """
    if _has_langgraph:
        workflow = StateGraph(AgentState)
    else:
        workflow = LocalStateGraph(AgentState)

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
