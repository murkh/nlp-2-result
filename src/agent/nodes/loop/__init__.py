"""
Bounded ReAct + reflection loop for structured queries.

The loop generates, validates and executes a statement, and feeds execution
failures back into the next attempt. Iterations are bounded by a state counter
and a conditional edge, so a spent budget escalates to the user instead of
crashing the graph.
"""

from src.agent.nodes.loop.generate import (
    code_executor_node,
    code_generator_node,
    code_validator_node,
    is_valid,
)
from src.agent.nodes.loop.reflect import (
    escalation_node,
    reflector_node,
    route_after_reflect,
)
from src.agent.nodes.loop.schema import has_schema, schema_retriever_node

__all__ = [
    "code_executor_node",
    "code_generator_node",
    "code_validator_node",
    "escalation_node",
    "has_schema",
    "is_valid",
    "reflector_node",
    "route_after_reflect",
    "schema_retriever_node",
]
