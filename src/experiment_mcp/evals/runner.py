"""Compare recorded tool calls with the eval spec, then run those calls.

Offline mode does not call a model and does not need a network. It checks that
``evals/recorded.json`` selects the tool and arguments named in ``evals/tasks.json``,
then executes each call against an in-memory server.

``--live`` sends each task to an LLM and scores the tool it selects. It reads
``EXPERIMENT_MCP_EVAL_API_KEY``, ``EXPERIMENT_MCP_EVAL_PROVIDER`` (anthropic or
openai), and ``EXPERIMENT_MCP_EVAL_MODEL``. This mode is optional and is not
what CI runs.
"""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client
from mcp.types import TextContent

from experiment_mcp.server import build_server


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "evals" / "tasks.json").is_file():
            return parent
    return Path.cwd()


ROOT = _repo_root()
DEFAULT_TASKS = ROOT / "evals" / "tasks.json"
DEFAULT_RECORDED = ROOT / "evals" / "recorded.json"


@dataclass(frozen=True)
class Task:
    id: str
    task: str
    tool: str
    arguments: dict[str, Any]


def load_tasks(path: Path) -> list[Task]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[Task] = []
    for item in raw:
        expected = item["expected"]
        tasks.append(
            Task(
                id=item["id"],
                task=item["task"],
                tool=expected["tool"],
                arguments=expected["arguments"],
            )
        )
    return tasks


def load_recorded(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in raw}


def _arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def score_recorded(tasks: list[Task], recorded: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for task in tasks:
        row = recorded.get(task.id)
        if row is None:
            failures.append(f"{task.id}: no recorded call")
            continue
        if row.get("tool") != task.tool:
            failures.append(f"{task.id}: recorded tool {row.get('tool')!r} != {task.tool!r}")
            continue
        arguments = row.get("arguments")
        if not isinstance(arguments, dict) or not _arguments_match(task.arguments, arguments):
            failures.append(f"{task.id}: recorded arguments do not match the expected arguments")
    return failures


async def execute_recorded(tasks: list[Task], recorded: dict[str, dict[str, Any]], db_path: Path) -> list[str]:
    failures: list[str] = []
    server = build_server(db_path=db_path)
    async with Client(server, raise_exceptions=True) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        for task in tasks:
            if task.tool not in names:
                failures.append(f"{task.id}: server has no tool {task.tool}")
                continue
            arguments = recorded[task.id]["arguments"]
            result = await client.call_tool(task.tool, arguments)
            if result.is_error:
                block = result.content[0] if result.content else None
                text = block.text if isinstance(block, TextContent) else ""
                failures.append(f"{task.id}: {task.tool} returned an error: {text}")
    return failures


async def _live_choice(task: Task, tools: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    provider = os.environ.get("EXPERIMENT_MCP_EVAL_PROVIDER", "anthropic")
    api_key = os.environ.get("EXPERIMENT_MCP_EVAL_API_KEY", "")
    model = os.environ.get("EXPERIMENT_MCP_EVAL_MODEL", "")
    if not api_key or not model:
        raise RuntimeError("live mode needs EXPERIMENT_MCP_EVAL_API_KEY and EXPERIMENT_MCP_EVAL_MODEL")
    if provider == "anthropic":
        payload = {
            "model": model,
            "max_tokens": 400,
            "system": "Pick the single experiment-mcp tool that best answers the user. Do not invent numbers.",
            "tools": tools,
            "messages": [{"role": "user", "content": task.task}],
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        base = os.environ.get("EXPERIMENT_MCP_EVAL_BASE_URL", "https://api.anthropic.com")
        async with httpx2.AsyncClient(timeout=60) as http:
            response = await http.post(f"{base.rstrip('/')}/v1/messages", headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        for block in body.get("content", []):
            if block.get("type") == "tool_use":
                return str(block["name"]), dict(block.get("input") or {})
        raise RuntimeError(f"{task.id}: model returned no tool call")
    if provider == "openai":
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Pick the single experiment-mcp tool that best answers the user.",
                },
                {"role": "user", "content": task.task},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") or "",
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ],
            "tool_choice": "required",
        }
        headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
        base = os.environ.get("EXPERIMENT_MCP_EVAL_BASE_URL", "https://api.openai.com")
        async with httpx2.AsyncClient(timeout=60) as http:
            response = await http.post(f"{base.rstrip('/')}/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        calls = body["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            raise RuntimeError(f"{task.id}: model returned no tool call")
        function = calls[0]["function"]
        return str(function["name"]), json.loads(function["arguments"])
    raise RuntimeError(f"unsupported EXPERIMENT_MCP_EVAL_PROVIDER {provider!r}")


async def score_live(tasks: list[Task], db_path: Path) -> list[str]:
    failures: list[str] = []
    server = build_server(db_path=db_path)
    async with Client(server) as client:
        listed = await client.list_tools()
        schemas = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
            for tool in listed.tools
        ]
        for task in tasks:
            try:
                name, arguments = await _live_choice(task, schemas)
            except Exception as exc:
                failures.append(f"{task.id}: live call failed: {exc}")
                continue
            if name != task.tool or not _arguments_match(task.arguments, arguments):
                failures.append(f"{task.id}: model selected {name} {arguments}, expected {task.tool} {task.arguments}")
    return failures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m experiment_mcp.evals")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--recorded", type=Path, default=DEFAULT_RECORDED)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    tasks = load_tasks(args.tasks)
    recorded = load_recorded(args.recorded)
    failures = score_recorded(tasks, recorded)
    if not args.live:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            failures.extend(asyncio.run(execute_recorded(tasks, recorded, Path(tmp) / "experiments.db")))
    else:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            failures.extend(asyncio.run(score_live(tasks, Path(tmp) / "experiments.db")))
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    mode = "live" if args.live else "offline"
    print(f"{len(tasks)} eval tasks passed ({mode})")


if __name__ == "__main__":
    main()
