"""Compare recorded tool calls with the eval spec, then run those calls.

Offline mode does not call a model and does not need a network. It checks that
``evals/recorded.json`` selects the tool and arguments named in ``evals/tasks.json``,
then executes each call against an in-memory server. Normal CI runs this mode.

``--live`` sends each task to OpenRouter and scores the tool call. It reads
``OPENROUTER_API_KEY`` and the model from ``--model`` or ``EVAL_MODEL``.
The default model is ``openai/gpt-4o-mini``. Results are written under
``evals/results/``. This mode is not part of normal CI.
"""

import argparse
import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.types import TextContent

from experiment_mcp.evals.live import LiveConfigError, require_api_key, resolve_model, select_tool
from experiment_mcp.evals.score import (
    MATCHING_RULES,
    RepeatScore,
    accuracy,
    model_slug,
    render_summary,
    score_repeat,
)
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


def git_commit() -> str:
    env = os.environ.get("GITHUB_SHA", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _repeat_payload(repeat: RepeatScore) -> dict[str, Any]:
    return {
        "tool": repeat.tool,
        "arguments": repeat.arguments,
        "calls": repeat.calls,
        "tool_match": repeat.tool_match,
        "argument_match": repeat.argument_match,
        "error": repeat.error,
    }


async def run_live(
    tasks: list[Task],
    *,
    db_path: Path,
    model: str,
    repeats: int,
    timeout: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Score each task ``repeats`` times. Model calls are not executed."""
    if repeats < 1:
        raise LiveConfigError("--repeats must be at least 1")
    api_key = require_api_key()
    server = build_server(db_path=db_path)
    async with Client(server) as client:
        listed = await client.list_tools()
        tools = list(listed.tools)
    task_rows: list[dict[str, Any]] = []
    tool_flags: list[bool] = []
    argument_flags: list[bool] = []
    transport_errors = 0
    for task in tasks:
        scored: list[RepeatScore] = []
        for _ in range(repeats):
            captured = await select_tool(
                api_key=api_key,
                model=model,
                task=task.task,
                listed_tools=tools,
                timeout=timeout,
            )
            scored.append(score_repeat(task.tool, task.arguments, captured))
        hits_tool = sum(1 for item in scored if item.tool_match)
        hits_args = sum(1 for item in scored if item.argument_match)
        tool_flags.extend(item.tool_match for item in scored)
        argument_flags.extend(item.argument_match for item in scored)
        transport_errors += sum(1 for item in scored if item.error and item.error.startswith("OpenRouter"))
        task_rows.append(
            {
                "id": task.id,
                "prompt": task.task,
                "expected_tool": task.tool,
                "expected_arguments": task.arguments,
                "repeats": repeats,
                "tool_hits": hits_tool,
                "argument_hits": hits_args,
                "tool_accuracy": hits_tool / repeats,
                "argument_accuracy": hits_args / repeats,
                "calls": [_repeat_payload(item) for item in scored],
            }
        )
    if transport_errors == len(tasks) * repeats and tasks:
        raise LiveConfigError("Every OpenRouter call failed. Check the key, model id, and network.")
    report: dict[str, Any] = {
        "provider": "openrouter",
        "model": model,
        "date": datetime.now(UTC).date().isoformat(),
        "commit": git_commit(),
        "task_count": len(tasks),
        "repeats": repeats,
        "tool_accuracy": accuracy(tool_flags),
        "argument_accuracy": accuracy(argument_flags),
        "matching": MATCHING_RULES,
        "executed": False,
        "tasks": task_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_slug(model)}-{report['date']}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = render_summary(report)
    md_path.write_text(summary, encoding="utf-8")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(summary)
            handle.write("\n")
    print(summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m experiment_mcp.evals")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--recorded", type=Path, default=DEFAULT_RECORDED)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=None, help="OpenRouter model id. Overrides EVAL_MODEL.")
    parser.add_argument("--repeats", type=int, default=1, help="Live calls per task.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds per OpenRouter request.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals" / "results")
    args = parser.parse_args(argv)
    tasks = load_tasks(args.tasks)
    recorded = load_recorded(args.recorded)
    failures = score_recorded(tasks, recorded)
    if not args.live:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            failures.extend(asyncio.run(execute_recorded(tasks, recorded, Path(tmp) / "experiments.db")))
        if failures:
            for failure in failures:
                print(failure)
            raise SystemExit(1)
        print(f"{len(tasks)} eval tasks passed (offline)")
        return
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(
                run_live(
                    tasks,
                    db_path=Path(tmp) / "experiments.db",
                    model=resolve_model(args.model),
                    repeats=args.repeats,
                    timeout=args.timeout,
                    output_dir=args.output_dir,
                )
            )
    except LiveConfigError as exc:
        print(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
