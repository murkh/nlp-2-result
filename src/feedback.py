"""
Execution feedback vocabulary.

Shared by the agent loop, which produces observations, and the engines, which
render them into a generation prompt. Kept at this level so engines never import
from the agent package.

Error classes follow LitE-SQL: syntax, missing_column, missing_table, misc.
"""

import re
from typing import Any, Dict, List, Literal, Optional

CorrectionClass = Literal["syntax", "missing_column", "missing_table", "misc"]

EXECUTION_ERROR = "execution_error"
EMPTY_RESULT = "empty_result"
CRITIQUE = "critique"

_MISSING_COLUMN = re.compile(
    r"column\s+.*\s+does not exist"
    r"|no such column"
    r"|unknown column"
    r"|referenced column\s+.*\s+not found"
    r"|column\s+.*\s+not found"
    r"|binder error.*column",
    re.IGNORECASE | re.DOTALL,
)

_MISSING_TABLE = re.compile(
    r"relation\s+.*\s+does not exist"
    r"|table with name\s+.*\s+does not exist"
    r"|table\s+.*\s+does not exist"
    r"|no such table"
    r"|unknown table"
    r"|catalog error.*table",
    re.IGNORECASE | re.DOTALL,
)

_SYNTAX = re.compile(
    r"syntax error"
    r"|parser error"
    r"|parse error"
    r"|unexpected token"
    r"|incomplete input"
    r"|invalid syntax",
    re.IGNORECASE,
)

_ADVICE = {
    "missing_column": (
        "A column name was wrong. Use only column names listed in the schema, spelled exactly."
    ),
    "missing_table": (
        "A table name was wrong. Use only table names listed in the schema, spelled exactly."
    ),
    "syntax": "The statement did not parse. Re-check the dialect and write a single valid statement.",
    "misc": "The statement failed at execution time. Re-read the error and change the approach.",
}


def classify_error(message: str) -> CorrectionClass:
    """Map an engine error string onto a correction class."""
    if not message:
        return "misc"
    if _MISSING_COLUMN.search(message):
        return "missing_column"
    if _MISSING_TABLE.search(message):
        return "missing_table"
    if _SYNTAX.search(message):
        return "syntax"
    return "misc"


def advice_for(correction_class: str) -> str:
    """Corrective instruction to hand the next generation attempt."""
    return _ADVICE.get(correction_class, _ADVICE["misc"])


def observation(kind: str, **fields: Any) -> Dict[str, Any]:
    """One entry for the `observations` state list."""
    return {"kind": kind, **fields}


def _render_one(entry: Dict[str, Any]) -> str:
    kind = entry.get("kind")
    attempt = entry.get("attempt")

    if kind == EXECUTION_ERROR:
        correction = entry.get("correction_class", "misc")
        return (
            f"Attempt {attempt} failed ({correction}): {entry.get('error')}\n"
            f"{advice_for(correction)}"
        )
    if kind == EMPTY_RESULT:
        return (
            f"Attempt {attempt} returned zero rows. The filters likely do not match "
            f"the stored values. Statement was:\n{entry.get('code')}"
        )
    if kind == CRITIQUE:
        return f"Attempt {attempt} rejected on review: {entry.get('critique')}"
    return str(entry)


def render_observations(observations: Optional[List[Dict[str, Any]]]) -> str:
    """
    Render observations for a prompt, in list order.

    Order must be stable across runs or the generator prompt varies and the
    evaluation suites stop being reproducible. The structured branch is strictly
    sequential, which is what keeps this order deterministic.
    """
    if not observations:
        return ""
    return "\n".join(_render_one(entry) for entry in observations)


def observation_prompt_block(observations: Optional[List[Dict[str, Any]]]) -> str:
    """
    Feedback section appended to a generation prompt.

    Empty string when there are no observations, so a first attempt's prompt is
    byte-identical to the single-pass prompt.
    """
    rendered = render_observations(observations)
    if not rendered:
        return ""
    return (
        "\n\nObservations from this session. Use the real stored values, and do not "
        f"repeat a failed approach:\n{rendered}"
    )
