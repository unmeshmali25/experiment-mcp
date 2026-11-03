"""Live tool-selection calls through OpenRouter.

The key is read from the environment and sent only as an Authorization header.
Errors report the HTTP status, not the request headers or the key.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import anyio
import httpx2

from experiment_mcp.evals.score import ListedTool, RepeatScore, to_openai_tools

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class RetryableError(Exception):
    """A live call failed in a way that is worth trying again."""


class LiveConfigError(Exception):
    """The live run cannot start."""


def resolve_model(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("EVAL_MODEL", "").strip()
    return env or DEFAULT_MODEL


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise LiveConfigError("OPENROUTER_API_KEY is not set. Add it to the environment or as a repository secret.")
    return key


async def with_retries(
    operation: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 3,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
) -> Any:
    """Run ``operation`` up to ``attempts`` times, backing off after ``RetryableError``."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except RetryableError:
            if attempt == attempts:
                raise
            await sleep(delay)
            delay *= 2
    raise RuntimeError("retries exhausted")


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise TypeError("tool arguments must be a JSON object")
        return parsed
    raise TypeError("tool arguments must be a JSON object or string")


def parse_chat_completion(body: dict[str, Any]) -> RepeatScore:
    """Read the first tool call from an OpenAI-compatible chat completion."""
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        return RepeatScore(None, None, [], False, False, f"response had no choices: {exc}")
    raw_calls = message.get("tool_calls") or []
    calls: list[dict[str, Any]] = []
    for call in raw_calls:
        function = call.get("function") or {}
        name = function.get("name")
        try:
            arguments = _parse_arguments(function.get("arguments"))
        except (json.JSONDecodeError, TypeError) as exc:
            calls.append({"name": name, "arguments": None, "error": str(exc)})
            continue
        calls.append({"name": name, "arguments": arguments})
    if not calls:
        return RepeatScore(None, None, [], False, False, "model returned no tool call")
    first = calls[0]
    if first.get("error"):
        return RepeatScore(
            str(first.get("name")) if first.get("name") else None, None, calls, False, False, str(first["error"])
        )
    selected_name = first.get("name")
    selected_arguments = first.get("arguments")
    if not isinstance(selected_name, str) or not isinstance(selected_arguments, dict):
        return RepeatScore(None, None, calls, False, False, "tool call was missing a name or arguments")
    return RepeatScore(selected_name, selected_arguments, calls, False, False, None)


async def complete_openrouter(
    *,
    api_key: str,
    model: str,
    task: str,
    tools: list[dict[str, Any]],
    timeout: float,
    base_url: str = OPENROUTER_URL,
) -> dict[str, Any]:
    """One chat completion. Raises ``RetryableError`` for timeouts and transient HTTP statuses."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 400,
        "tool_choice": "required",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Call exactly one experiment-mcp tool. Use only arguments the user asked for. Do not explain."
                ),
            },
            {"role": "user", "content": task},
        ],
        "tools": tools,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "experiment-mcp",
    }

    async def once() -> dict[str, Any]:
        try:
            async with httpx2.AsyncClient(timeout=timeout) as http:
                response = await http.post(base_url, headers=headers, json=payload)
        except (httpx2.TimeoutException, httpx2.TransportError) as exc:
            raise RetryableError("OpenRouter request timed out or failed to connect") from exc
        if response.status_code in RETRY_STATUSES:
            raise RetryableError(f"OpenRouter returned HTTP {response.status_code}")
        if response.status_code >= 400:
            raise LiveConfigError(f"OpenRouter returned HTTP {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            raise LiveConfigError("OpenRouter returned a response that was not a JSON object")
        return body

    try:
        return await with_retries(once)
    except RetryableError as exc:
        raise LiveConfigError(str(exc)) from exc


async def select_tool(
    *,
    api_key: str,
    model: str,
    task: str,
    listed_tools: Sequence[ListedTool],
    timeout: float,
) -> RepeatScore:
    try:
        body = await complete_openrouter(
            api_key=api_key,
            model=model,
            task=task,
            tools=to_openai_tools(listed_tools),
            timeout=timeout,
        )
    except LiveConfigError as exc:
        return RepeatScore(None, None, [], False, False, str(exc))
    return parse_chat_completion(body)
