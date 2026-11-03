from pathlib import Path

import pytest
from mcp import Client

from experiment_mcp.evals.live import (
    LiveConfigError,
    RetryableError,
    parse_chat_completion,
    require_api_key,
    resolve_model,
    with_retries,
)
from experiment_mcp.evals.score import (
    RepeatScore,
    arguments_match,
    model_slug,
    render_summary,
    score_repeat,
    to_openai_tools,
)
from experiment_mcp.server import build_server


class _Tool:
    def __init__(self, name: str, description: str | None, input_schema: dict[str, object]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema


def test_argument_rules() -> None:
    expected = {
        "metric": "proportion",
        "baseline": 0.1,
        "mde": 0.02,
        "n_per_arm": 63,
        "unit_ids": ["c00001", "c00002"],
        "splits": {"control": 0.5, "treatment": 0.5},
        "use_cuped": True,
    }
    actual = {
        "metric": " proportion ",
        "baseline": 0.1000000001,
        "mde": 0.02,
        "n_per_arm": 63.0,
        "unit_ids": ["c00001", "c00002"],
        "splits": {"treatment": 0.5, "control": 0.5, "note": "ignored"},
        "use_cuped": True,
        "alpha": 0.05,
    }
    assert arguments_match(expected, actual)
    assert not arguments_match(expected, {**actual, "metric": "mean"})
    assert not arguments_match(expected, {**actual, "n_per_arm": 64})
    assert not arguments_match(expected, {**actual, "unit_ids": ["c00002", "c00001"]})
    assert not arguments_match(expected, {**actual, "use_cuped": False})
    assert not arguments_match({"offer_id": "WEEKEND_15"}, {"outcome": "redeemed"})
    assert arguments_match({}, {"status": "running"})


def test_openai_tool_conversion() -> None:
    tools = to_openai_tools(
        [
            _Tool(
                "power_calc",
                "Estimate sample size.",
                {"type": "object", "properties": {"metric": {"type": "string"}}, "required": ["metric"]},
            )
        ]
    )
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "power_calc",
                "description": "Estimate sample size.",
                "parameters": {
                    "type": "object",
                    "properties": {"metric": {"type": "string"}},
                    "required": ["metric"],
                },
            },
        }
    ]


def test_parse_scores_the_first_tool_call_only() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "power_calc",
                                "arguments": '{"metric": "proportion", "baseline": 0.1}',
                            }
                        },
                        {"function": {"name": "list_experiments", "arguments": "{}"}},
                    ]
                }
            }
        ]
    }
    parsed = parse_chat_completion(body)
    scored = score_repeat("power_calc", {"metric": "proportion", "baseline": 0.1}, parsed)
    assert scored.tool_match is True
    assert scored.argument_match is True
    assert len(scored.calls) == 2
    missed = score_repeat("list_experiments", {}, parsed)
    assert missed.tool_match is False
    assert missed.argument_match is False


def test_parse_missing_tool_call() -> None:
    parsed = parse_chat_completion({"choices": [{"message": {"content": "hello"}}]})
    assert parsed.error == "model returned no tool call"
    assert parsed.tool_match is False


def test_summary_and_slug() -> None:
    assert model_slug("openai/gpt-4o-mini") == "openai-gpt-4o-mini"
    text = render_summary(
        {
            "model": "openai/gpt-4o-mini",
            "date": "2026-11-03",
            "commit": "abc",
            "task_count": 1,
            "repeats": 2,
            "tool_accuracy": 0.5,
            "argument_accuracy": 0.5,
            "tasks": [{"id": "power-redemption", "tool_hits": 1, "argument_hits": 1, "repeats": 2}],
        }
    )
    assert "50.0%" in text
    assert "power-redemption" in text
    assert "abc" in text


def test_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_MODEL", raising=False)
    assert resolve_model(None) == "openai/gpt-4o-mini"
    assert resolve_model("google/gemini-2.5-flash") == "google/gemini-2.5-flash"
    monkeypatch.setenv("EVAL_MODEL", "openai/gpt-4o-mini")
    assert resolve_model(None) == "openai/gpt-4o-mini"
    assert resolve_model("other/model") == "other/model"


def test_missing_key_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LiveConfigError, match="OPENROUTER_API_KEY"):
        require_api_key()


@pytest.mark.anyio
async def test_retries_backoff_then_succeed() -> None:
    calls = {"n": 0}
    delays: list[float] = []

    async def operation() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("OpenRouter returned HTTP 429")
        return "ok"

    async def sleep(delay: float) -> None:
        delays.append(delay)

    assert await with_retries(operation, attempts=3, sleep=sleep) == "ok"
    assert delays == [1.0, 2.0]


@pytest.mark.anyio
async def test_non_retryable_is_not_retried() -> None:
    calls = {"n": 0}

    async def operation() -> str:
        calls["n"] += 1
        raise LiveConfigError("OpenRouter returned HTTP 401")

    with pytest.raises(LiveConfigError, match="401"):
        await with_retries(operation, attempts=3, sleep=_no_sleep)
    assert calls["n"] == 1


async def _no_sleep(_delay: float) -> None:
    raise AssertionError("sleep should not run")


@pytest.mark.anyio
async def test_listed_server_tools_convert(tmp_path: Path) -> None:
    server = build_server(db_path=tmp_path / "experiments.db")
    async with Client(server) as client:
        listed = await client.list_tools()
    converted = to_openai_tools(list(listed.tools))
    power = next(tool for tool in converted if tool["function"]["name"] == "power_calc")
    assert power["type"] == "function"
    assert "metric" in power["function"]["parameters"]["properties"]
    assert "description" in power["function"]
    assert {tool["function"]["name"] for tool in converted} >= {
        "analyze_lift",
        "create_experiment",
        "close_experiment",
    }


def test_repeat_score_is_a_dataclass() -> None:
    blank = RepeatScore(None, None, [], False, False, "model returned no tool call")
    scored = score_repeat("power_calc", {"metric": "mean"}, blank)
    assert scored.argument_match is False
