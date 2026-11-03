"""Pure scoring helpers for the tool-selection evals.

Live runs compare a model's tool call with the expected call in ``tasks.json``.

Matching rules:

- The scored call is the first tool call in the response. Later calls are kept
  in the result but do not change the score.
- The tool name must match exactly.
- Every key in the expected arguments must be present. Extra keys are ignored,
  so a model may repeat a default the task did not mention.
- Strings match after stripping surrounding whitespace. Comparison is case-sensitive.
- Booleans match exactly. They are not treated as integers.
- Integers match exactly. A float whose value is that integer (``63.0``) matches ``63``.
- Other numbers match when the absolute difference is at most ``1e-6`` or the
  relative difference is at most ``1e-4``.
- Lists match in order, with the same length, using these rules on each item.
- Nested objects use the same key rule: expected keys must match, extras are ignored.
- A wrong tool name fails the argument check as well. Argument accuracy is the
  share of repeats where the tool name and the arguments both match.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

ABS_TOL = 1e-6
REL_TOL = 1e-4

MATCHING_RULES = (
    "First tool call only. Tool name must match exactly. "
    "Expected argument keys are required; extra keys are ignored. "
    "Strings match after stripping whitespace, case-sensitive. "
    "Booleans are exact. Integers are exact (63.0 counts as 63). "
    "Other numbers match within 1e-6 absolute or 1e-4 relative. "
    "Lists match in order. A wrong tool name fails the argument check."
)


class ListedTool(Protocol):
    name: str
    description: str | None
    input_schema: dict[str, Any]


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def values_match(expected: Any, actual: Any) -> bool:
    """Return whether ``actual`` satisfies one expected value."""
    if _is_bool(expected) or _is_bool(actual):
        return _is_bool(expected) and _is_bool(actual) and expected is actual
    if isinstance(expected, str):
        return isinstance(actual, str) and expected.strip() == actual.strip()
    if expected is None or actual is None:
        return expected is actual
    if _is_int(expected) and _is_number(actual):
        if _is_int(actual):
            return expected == actual
        if isinstance(actual, float) and actual.is_integer():
            return expected == int(actual)
        return False
    if _is_number(expected) and _is_number(actual):
        exp = float(expected)
        act = float(actual)
        diff = abs(exp - act)
        scale = max(abs(exp), abs(act))
        if scale == 0:
            return True
        return diff <= ABS_TOL or diff / scale <= REL_TOL
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(values_match(item, other) for item, other in zip(expected, actual, strict=True))
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and values_match(value, actual[key]) for key, value in expected.items())
    return bool(expected == actual)


def arguments_match(expected: dict[str, Any], actual: Any) -> bool:
    """Return whether ``actual`` contains every expected argument."""
    if not isinstance(actual, dict):
        return False
    return values_match(expected, actual)


def to_openai_tools(tools: Sequence[ListedTool]) -> list[dict[str, Any]]:
    """Convert MCP ``tools/list`` entries to OpenAI chat-completions tools."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        schema = tool.input_schema
        if not isinstance(schema, dict):
            raise TypeError(f"tool {tool.name} input_schema must be an object")
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return converted


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
    return slug or "model"


@dataclass(frozen=True)
class RepeatScore:
    tool: str | None
    arguments: dict[str, Any] | None
    calls: list[dict[str, Any]]
    tool_match: bool
    argument_match: bool
    error: str | None


def score_repeat(expected_tool: str, expected_arguments: dict[str, Any], repeat: RepeatScore) -> RepeatScore:
    """Fill match flags from a captured call. ``repeat`` may leave them false."""
    tool_match = repeat.error is None and repeat.tool == expected_tool
    argument_match = tool_match and arguments_match(expected_arguments, repeat.arguments)
    return RepeatScore(
        tool=repeat.tool,
        arguments=repeat.arguments,
        calls=repeat.calls,
        tool_match=tool_match,
        argument_match=argument_match,
        error=repeat.error,
    )


def accuracy(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def render_summary(report: dict[str, Any]) -> str:
    """Markdown summary of one live run."""
    lines = [
        f"# Eval {report['model']}",
        "",
        f"- Date: {report['date']}",
        f"- Commit: `{report['commit']}`",
        f"- Tasks: {report['task_count']}",
        f"- Repeats per task: {report['repeats']}",
        f"- Tool-name accuracy: {report['tool_accuracy']:.1%}",
        f"- Argument accuracy: {report['argument_accuracy']:.1%}",
        "",
        "Argument accuracy counts a repeat only when the tool name matches and the expected arguments match.",
        "",
        MATCHING_RULES,
        "",
        "| Task | Tool | Arguments |",
        "| --- | --- | --- |",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| {task['id']} | {task['tool_hits']}/{task['repeats']} | {task['argument_hits']}/{task['repeats']} |"
        )
    lines.append("")
    return "\n".join(lines)
