from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from mcp import Client
from mcp.types import (
    CreateMessageResult,
    ElicitResult,
    PromptReference,
    ResourceTemplateReference,
    TextContent,
)

from experiment_mcp.server import build_server


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[Client]:
    server = build_server(db_path=tmp_path / "experiments.db")
    async with Client(server, raise_exceptions=True) as connected:
        yield connected


@pytest.mark.anyio
async def test_lists_tools_resources_and_prompts(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert {
        "power_calc",
        "assign_cells",
        "analyze_lift",
        "check_srm",
        "list_experiments",
        "create_experiment",
        "close_experiment",
        "draft_readout",
    } <= set(tools)
    assert tools["power_calc"].annotations is not None
    assert tools["power_calc"].annotations.read_only_hint is True
    assert tools["close_experiment"].annotations is not None
    assert tools["close_experiment"].annotations.destructive_hint is True
    assert "confirm" in tools["close_experiment"].input_schema["properties"]
    assert "decision" not in tools["close_experiment"].input_schema["properties"]
    assert tools["power_calc"].output_schema is not None
    resources = {resource.uri for resource in (await client.list_resources()).resources}
    assert "dataset://schema" in resources
    assert "dataset://summary" in resources
    assert "experiment://index" in resources
    templates = {template.uri_template for template in (await client.list_resource_templates()).resource_templates}
    assert "experiment://{experiment_id}" in templates
    assert "experiment://{experiment_id}/results" in templates
    prompts = {prompt.name for prompt in (await client.list_prompts()).prompts}
    assert prompts == {"design_experiment", "write_readout"}
    assert client.server_capabilities is not None
    assert client.server_capabilities.completions is not None


@pytest.mark.anyio
async def test_power_tool_validates_input_and_returns_structured_output(client: Client) -> None:
    result = await client.call_tool(
        "power_calc",
        {"metric": "proportion", "baseline": 0.1, "mde": 0.02},
    )
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["n_per_arm"] == 3841
    bad = await client.call_tool("power_calc", {"metric": "proportion", "baseline": 0.1, "mde": 0.02, "alpha": 2})
    assert bad.is_error is True


@pytest.mark.anyio
async def test_offer_lift_and_resources(client: Client) -> None:
    schema = await client.read_resource("dataset://schema")
    assert schema.contents[0].text is not None
    assert "synthetic" in schema.contents[0].text
    summary = await client.read_resource("dataset://summary")
    assert "WEEKEND_15" in summary.contents[0].text
    created = await client.call_tool(
        "create_experiment",
        {
            "name": "Weekend",
            "hypothesis": "Redemption rises.",
            "primary_metric": "redeemed",
            "offer_id": "WEEKEND_15",
            "salt": "fixed-salt",
        },
    )
    experiment_id = created.structured_content["id"]
    record = await client.read_resource(f"experiment://{experiment_id}")
    assert "fixed-salt" in record.contents[0].text
    results = await client.read_resource(f"experiment://{experiment_id}/results")
    assert "unadjusted" in results.contents[0].text
    assert "cuped" in results.contents[0].text
    lift = await client.call_tool(
        "analyze_lift",
        {"offer_id": "WEEKEND_15", "outcome": "redeemed"},
    )
    assert lift.structured_content["control"]["n"] > 2
    assert lift.structured_content["treatment"]["n"] > 2


@pytest.mark.anyio
async def test_missing_experiment_resource_is_not_found(client: Client) -> None:
    with pytest.raises(Exception, match="does-not-exist|No experiment"):
        await client.read_resource("experiment://does-not-exist")


@pytest.mark.anyio
async def test_bootstrap_reports_increasing_progress(client: Client) -> None:
    reports: list[tuple[float, float | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        reports.append((progress, total))
        assert message is not None

    result = await client.call_tool(
        "analyze_lift",
        {
            "control": [1, 2, 3, 4, 5, 6],
            "treatment": [2, 3, 4, 5, 6, 7],
            "outcome": "spend",
            "ci_method": "bootstrap",
            "bootstrap_iterations": 100,
            "seed": 1,
        },
        progress_callback=on_progress,
    )
    assert result.is_error is False
    assert result.structured_content["ci_method"] == "bootstrap"
    assert [progress for progress, _total in reports] == sorted(progress for progress, _total in reports)
    assert reports[0][0] < reports[-1][0]
    assert reports[-1] == (100.0, 100.0)


@pytest.mark.anyio
async def test_close_falls_back_without_elicitation_and_confirms_with_argument(
    tmp_path: Path,
) -> None:
    server = build_server(db_path=tmp_path / "experiments.db")
    async with Client(server, raise_exceptions=True) as client:
        created = await client.call_tool(
            "create_experiment",
            {"name": "Hold", "hypothesis": "Something changes.", "primary_metric": "post_spend"},
        )
        experiment_id = created.structured_content["id"]
        blocked = await client.call_tool("close_experiment", {"experiment_id": experiment_id})
        assert blocked.structured_content["closed"] is False
        assert blocked.structured_content["reason"] == "confirmation_required"
        closed = await client.call_tool(
            "close_experiment",
            {"experiment_id": experiment_id, "confirm": True},
        )
        assert closed.structured_content["closed"] is True
        again = await client.call_tool(
            "close_experiment",
            {"experiment_id": experiment_id, "confirm": True},
        )
        assert again.structured_content["reason"] == "already_closed"


@pytest.mark.anyio
async def test_close_uses_elicitation_callback(tmp_path: Path) -> None:
    server = build_server(db_path=tmp_path / "experiments.db")

    async def accept(_context: object, params: object) -> ElicitResult:
        assert "Close experiment" in params.message  # type: ignore[attr-defined]
        return ElicitResult(action="accept", content={"confirm": True})

    async with Client(server, elicitation_callback=accept, raise_exceptions=True) as client:
        created = await client.call_tool(
            "create_experiment",
            {"name": "Ask", "hypothesis": "Ask first.", "primary_metric": "redeemed"},
        )
        result = await client.call_tool(
            "close_experiment",
            {"experiment_id": created.structured_content["id"]},
        )
        assert result.structured_content["closed"] is True


@pytest.mark.anyio
async def test_declined_elicitation_does_not_close(tmp_path: Path) -> None:
    server = build_server(db_path=tmp_path / "experiments.db")

    async def decline(_context: object, _params: object) -> ElicitResult:
        return ElicitResult(action="decline")

    async with Client(server, elicitation_callback=decline, raise_exceptions=True) as client:
        created = await client.call_tool(
            "create_experiment",
            {"name": "Keep", "hypothesis": "Stay open.", "primary_metric": "redeemed"},
        )
        result = await client.call_tool(
            "close_experiment",
            {"experiment_id": created.structured_content["id"]},
        )
        assert result.structured_content["reason"] == "declined"
        listed = await client.call_tool("list_experiments", {"status": "running"})
        assert listed.structured_content["experiments"][0]["id"] == created.structured_content["id"]


@pytest.mark.anyio
async def test_draft_readout_samples_or_falls_back(tmp_path: Path) -> None:
    server = build_server(db_path=tmp_path / "experiments.db")
    async with Client(server, raise_exceptions=True) as client:
        created = await client.call_tool(
            "create_experiment",
            {
                "name": "Readout",
                "hypothesis": "Spend rises.",
                "primary_metric": "post_spend",
                "offer_id": "WEEKEND_15",
            },
        )
        fallback = await client.call_tool(
            "draft_readout",
            {"experiment_id": created.structured_content["id"]},
        )
        assert fallback.structured_content["source"] == "fallback"
        assert "synthetic" in fallback.structured_content["text"]

    async def sample(_context: object, _params: object) -> CreateMessageResult:
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text="Synthetic readout: redemption was not the primary metric."),
            model="mock",
            stop_reason="end_turn",
        )

    async with Client(server, sampling_callback=sample, raise_exceptions=True) as client:
        drafted = await client.call_tool(
            "draft_readout",
            {"experiment_id": created.structured_content["id"]},
        )
        assert drafted.structured_content["source"] == "sampling"
        assert drafted.structured_content["text"].startswith("Synthetic readout")


@pytest.mark.anyio
async def test_completions_cover_prompts_and_resources(tmp_path: Path) -> None:
    server = build_server(db_path=tmp_path / "experiments.db")
    async with Client(server, raise_exceptions=True) as client:
        metrics = await client.complete(
            PromptReference(name="design_experiment"),
            {"name": "metric", "value": "post"},
        )
        assert metrics.completion.values == ["post_spend"]
        audiences = await client.complete(
            PromptReference(name="write_readout"),
            {"name": "audience", "value": ""},
        )
        assert "analyst" in audiences.completion.values
        created = await client.call_tool(
            "create_experiment",
            {"name": "Named", "hypothesis": "Stored.", "primary_metric": "redeemed"},
        )
        experiment_id = created.structured_content["id"]
        completed = await client.complete(
            ResourceTemplateReference(uri="experiment://{experiment_id}"),
            {"name": "experiment_id", "value": experiment_id[:4]},
        )
        assert experiment_id in completed.completion.values
        prompt = await client.get_prompt(
            "design_experiment",
            {"metric": "redeemed", "baseline": "0.08", "mde": "0.02", "population": "active"},
        )
        assert "power_calc" in prompt.messages[0].content.text  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_resource_list_notification(tmp_path: Path) -> None:
    server = build_server(db_path=tmp_path / "experiments.db")
    async with Client(server, raise_exceptions=True) as client:
        async with client.listen(resources_list_changed=True) as subscription:
            await client.call_tool(
                "create_experiment",
                {"name": "Notify", "hypothesis": "Lists change.", "primary_metric": "redeemed"},
            )
            event = await subscription.__anext__()
            assert event is not None
