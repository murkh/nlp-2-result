"""
Schema retrieval node.

Runs the two-stage pruner once per query. Everything downstream reads the pruned
slice from state rather than pruning again.
"""

from typing import Any, Dict

from src.agent.state import AgentState
from src.api.schemas import SchemaContextRef
from src.database.connection import get_db_manager
from src.pruning.schema_pruner import TwoStageSchemaPruner

NO_SCHEMA_MESSAGE = (
    "No structured datasets are available to query. Upload a CSV, Parquet, or Excel "
    "file in the Ingestion Hub first."
)


def schema_retriever_node(state: AgentState) -> Dict[str, Any]:
    """Prune the catalog down to the tables and columns this query needs."""
    pruner = TwoStageSchemaPruner(db_manager=get_db_manager())
    context = pruner.prune_schema(
        query=state.get("query", ""),
        dataset_ids=state.get("candidate_datasets") or None,
    )

    update: Dict[str, Any] = {
        "pruned_tables": SchemaContextRef.from_pruned(context).model_dump(),
        "schema_ddl": context.ddl_prompt_snippet,
        "loop_iterations": 0,
    }

    if not context.table_names:
        update["execution_error"] = NO_SCHEMA_MESSAGE
        update["clarification_message"] = NO_SCHEMA_MESSAGE

    return update


def has_schema(state: AgentState) -> str:
    """Route out of the loop when there is nothing to query."""
    pruned = state.get("pruned_tables") or {}
    return "generate" if pruned.get("table_names") else "no_schema"
