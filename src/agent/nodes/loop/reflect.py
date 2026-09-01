"""
Reflection node.

Deterministic classification first: an execution error, zero rows, or an
all-NULL row are cheap signals that need no model call, and execution feedback
is a stronger correction signal than model introspection. The optional LLM
critique only looks at results that already passed those checks.
"""

import logging
from typing import Any, Dict, List, Tuple

from src.agent.nodes.loop._shared import add_tokens, schema_summary
from src.agent.state import AgentState
from src.config import Settings, get_settings
from src.feedback import CRITIQUE, EMPTY_RESULT, EXECUTION_ERROR, observation
from src.llm import require_openai_client

logger = logging.getLogger(__name__)

OK = "ok"
ERROR = "error"
EMPTY = "empty_result"
DEGENERATE = "degenerate"
CRITIQUE_REJECTED = "critique_rejected"


def _is_all_null(rows: List[Dict[str, Any]]) -> bool:
    return len(rows) == 1 and all(value is None for value in rows[0].values())


def classify_outcome(state: AgentState) -> str:
    """Deterministic verdict on the last execution."""
    if state.get("execution_error"):
        return ERROR

    rows = state.get("execution_result") or []
    if not rows:
        return EMPTY
    if _is_all_null(rows):
        return DEGENERATE
    return OK


def _critique(state: AgentState, settings: Settings) -> Tuple[bool, str, Tuple[int, int]]:
    """Ask the model whether the result actually answers the question."""
    client = require_openai_client(settings)
    rows = state.get("execution_result") or []
    prompt = (
        "Judge whether the result answers the question. Be strict about the question's "
        "conditions, lenient about formatting.\n"
        f"Question: {state.get('query', '')}\n"
        f"Schema:\n{schema_summary(state.get('pruned_tables') or {})}\n"
        f"Statement:\n{state.get('generated_code') or ''}\n"
        f"Result columns: {state.get('execution_columns') or []}\n"
        f"First rows: {rows[:3]}\n\n"
        "Reply with exactly one line: 'OK' if it answers the question, otherwise "
        "'RETRY: <what is wrong>'."
    )
    response = client.chat.completions.create(
        model=settings.critic_model or settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    reply = (response.choices[0].message.content or "").strip()
    completion_tokens = (
        response.usage.completion_tokens if response.usage else max(1, len(reply) // 4)
    )
    tokens = (max(1, len(prompt) // 4), completion_tokens)

    if reply.upper().startswith("OK"):
        return True, "", tokens
    return False, reply.split(":", 1)[-1].strip() or reply, tokens


def reflector_node(state: AgentState) -> Dict[str, Any]:
    """Classify the attempt and record why the loop will or will not continue."""
    settings = get_settings()
    attempt = state.get("loop_iterations") or 1
    outcome = classify_outcome(state)

    update: Dict[str, Any] = {}
    observations: List[Dict[str, Any]] = []
    telemetry = dict(state.get("telemetry") or {})

    budget_remains = attempt < settings.structured_loop_max_iters

    if outcome == EMPTY and budget_remains:
        observations.append(
            observation(EMPTY_RESULT, attempt=attempt, code=state.get("generated_code") or "")
        )

    if outcome == OK and settings.reflection_enabled and budget_remains:
        accepted, critique, tokens = _critique(state, settings)
        telemetry = add_tokens(telemetry, tokens[0], tokens[1])
        if not accepted:
            outcome = CRITIQUE_REJECTED
            observations.append(observation(CRITIQUE, attempt=attempt, critique=critique))

    loop_telemetry = dict(telemetry.get("loop") or {})
    loop_telemetry["iterations"] = attempt
    loop_telemetry["outcome"] = outcome
    loop_telemetry.setdefault("tool_calls", 0)
    loop_telemetry["attempts"] = [
        *(loop_telemetry.get("attempts") or []),
        {"attempt": attempt, "outcome": outcome, "error": state.get("execution_error")},
    ]
    telemetry["loop"] = loop_telemetry
    telemetry["execution_success"] = state.get("execution_error") is None

    update["reflection_class"] = outcome
    update["telemetry"] = telemetry
    if observations:
        update["observations"] = observations

    return update


def route_after_reflect(state: AgentState) -> str:
    """
    Continue, retry, or escalate.

    A spent budget only escalates when no attempt ever executed. An empty result
    that executed cleanly is a real answer, so it is forwarded rather than
    reported as a failure.
    """
    settings = get_settings()
    outcome = state.get("reflection_class") or OK
    attempt = state.get("loop_iterations") or 1

    if outcome == OK:
        return "continue"
    if attempt < settings.structured_loop_max_iters:
        return "retry"
    return "escalate" if state.get("execution_error") else "continue"


def _attempt_errors(observations: List[Dict[str, Any]]) -> List[str]:
    return [
        f"- attempt {entry.get('attempt')} ({entry.get('correction_class')}): {entry.get('error')}"
        for entry in observations
        if entry.get("kind") == EXECUTION_ERROR
    ]


def escalation_node(state: AgentState) -> Dict[str, Any]:
    """
    Hand the query back to the user after the retry budget is spent.

    Reports what was actually tried and what the data actually contains. The
    failure stays a failure in telemetry -- no answer is invented from it.
    """
    observations = list(state.get("observations") or [])
    attempts = state.get("loop_iterations", 0)

    sections = [
        f"I could not build a working query for '{state.get('query', '')}' "
        f"after {attempts} attempt(s), so I am not going to guess at an answer.",
    ]

    errors = _attempt_errors(observations)
    if errors:
        sections.append("**What failed:**\n" + "\n".join(errors))

    sections.append(
        "Could you rephrase the question, or name the table and column you want?"
    )
    message = "\n\n".join(sections)

    telemetry = dict(state.get("telemetry") or {})
    telemetry["execution_success"] = False
    loop_telemetry = dict(telemetry.get("loop") or {})
    loop_telemetry["escalated"] = True
    telemetry["loop"] = loop_telemetry

    return {
        "clarification_message": message,
        "final_answer": message,
        "citations": [],
        "telemetry": telemetry,
    }


__all__ = [
    "classify_outcome",
    "escalation_node",
    "reflector_node",
    "route_after_reflect",
]
