# Why MCP instead of a plain API?

The statistics in `src/experiment_mcp/stats/` are plain functions. Nothing in that package knows about MCP. The server in `src/experiment_mcp/server.py` is a client contract: how a model runtime discovers those functions, passes arguments, and gets structured results back.

A REST endpoint can expose the same function. MCP is a different contract, aimed at clients that already speak it.

## One server, many clients

Claude Desktop, Claude Code, Cursor, and MCP Inspector all speak MCP. This repo has two entry points for the same `MCPServer`:

- stdio: `experiment-mcp serve`
- Streamable HTTP: `experiment-mcp serve --transport http`

Adding another MCP client is a config block (command, or URL plus a bearer token). Adding a tool is one decorated function. The client calls `tools/list` and `tools/call`. There is no per-client SDK adapter in this repository.

A hand-maintained function-calling integration does not work that way. Each client SDK wants its own tool list, its own JSON shape, and its own dispatch loop. N clients and M operations become N times M pieces of glue. MCP is N plus M: N client configs, M tools on one server.

## Schemas come from the function

`power_calc` is a Python function with type hints, `Field` constraints, a docstring, a `PowerResult` return type, and `ToolAnnotations`. The SDK turns that into the tool's input schema, output schema, description, and annotations. Clients do not carry a second copy.

A REST API plus function calling usually keeps three artifacts: the HTTP handler, an OpenAI-style `tools` array, and an Anthropic-style `tools` array. Those copies drift. This server does not publish a separate tool catalog.

## The channel is two-way

A request/response API can return a body. These calls also use server-to-client traffic that is part of the protocol:

- `close_experiment` returns `Elicit(...)` from `resolve_close` before it writes. The client shows a form (`confirm: bool`). Decline and cancel leave the row running. The message is `Close experiment {id} ({name})? The record is kept and the status becomes closed.`
- `draft_readout` returns `Sample(...)` when the client declares sampling. The client's own model writes the prose. This process does not hold a model API key. If the client has no sampling capability, the tool returns a local template and `source` is `fallback`.
- Bootstrap `analyze_lift` calls `ctx.report_progress` between batches (`Bootstrap replicate {done} of {total}`) and checks for cancellation between batches.
- `create_experiment` and `close_experiment` call `ctx.notify_resources_changed` and `ctx.notify_resource_updated` for `experiment://index`, `experiment://{id}`, and `experiment://{id}/results`.

None of that is a custom websocket invented beside a REST API. It is the MCP session the client already opened. The HTTP app does not set `stateless_http`, so a pre-2026 client keeps a session on this process.

## Context, not only actions

Tools perform work. Two other primitives are for the model to read:

- Resources: `dataset://schema`, `dataset://summary`, `experiment://index`, `experiment://{experiment_id}`, `experiment://{experiment_id}/results`.
- Prompts: `design_experiment` and `write_readout`, with completions for metric, population, audience, and stored experiment ids.

A REST API can serve the same JSON. The client then needs its own convention for "this URL is context" versus "this URL is an action." MCP already separates `resources/read` and `prompts/get` from `tools/call`.

## Hints and scopes

`power_calc` is registered with `readOnlyHint` and `openWorldHint` false. `close_experiment` sets `destructiveHint`. Those annotations are hints for the client UI. A client can ignore them. They are not access control.

On Streamable HTTP the process is an OAuth resource server. It checks the bearer JWT, and `_ensure_scope` requires `experiments:read` or `experiments:write` per tool. A read token can call `power_calc`. It cannot call `create_experiment`. stdio does not see a token. The check is skipped there. The boundary is whoever can start the process.

## When a plain API is the better fit

- No model in the path. `power_calculation` is a function. Call it, or put one HTTP route in front of it. MCP adds a protocol for clients that list tools and call them.
- The integration is fixed and service-to-service. One known caller, one known payload, no discovery, no elicitation, no sampling. The extra session and framing are cost without a client that uses them.
- The security review you need is the ordinary one. MCP is younger than HTTP APIs and OAuth. This server still validates JWTs and scopes. The protocol does not make that review shorter, and a bug in a tool is still a bug in the function the tool calls.
- The tool is a wrapper. `analyze_lift` on an `offer_id` reads local CSV. A future tool could call an internal experimentation HTTP API. MCP would still be the model-facing contract. The system of record would still be that API.

## Comparison

| | This MCP server | A plain HTTP API plus function calling |
| --- | --- | --- |
| Add a client that already speaks MCP | Config: stdio command or HTTP URL | A new adapter for that client's tool schema and dispatch loop |
| Tool contract | Generated from `power_calc` and `PowerResult` at `tools/list` | Hand-written JSON, usually once per vendor SDK |
| Mid-call confirmation | `close_experiment` elicitation | A flag on the request, or a second endpoint the adapter knows about |
| Readout prose | `draft_readout` samples the client model, or a local template | The server calls a model API, or the adapter does |
| Long job | `notifications/progress` from `analyze_lift` | Polling, SSE, or whatever the adapter invented |
| Context the model can read | Resources and prompts | More endpoints, plus client code that decides to fetch them |
| Safety | Annotations are hints. HTTP scopes are enforced | Whatever the API checks. Hints are not a standard field |

## Side by side

The MCP tool is the implementation. The REST sketch is not code in this repo. It shows the extra catalog a function-calling client has to keep.

MCP, abbreviated from `server.py`. The parameters, defaults, and `PowerResult` fields are the schema:

```python
@mcp.tool(
    title="Sample size or MDE",
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
async def power_calc(
    metric: Literal["proportion", "mean"],
    alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05,
    power: Annotated[float, Field(gt=0, lt=1)] = 0.8,
    baseline: float | None = None,
    mde: float | None = None,
    n_per_arm: int | None = None,
    sigma: float | None = None,
) -> PowerResult:
    """Estimate equal-allocation sample size or MDE for a two-sided test."""
    _ensure_scope(READ_SCOPE, error=ToolError)
    return power_calculation(
        metric=metric, alpha=alpha, power=power, baseline=baseline,
        sigma=sigma, mde=mde, n_per_arm=n_per_arm,
    )
```

Any MCP client then does the equivalent of `tools/list` and `tools/call`. It does not embed the schema.

The same calculation behind a route, with the catalogs a typical function-calling stack maintains by hand:

```python
@app.post("/v1/power")
def power(body: PowerBody) -> PowerResult:
    return power_calculation(**body.model_dump())

# Copy 1, OpenAI Chat Completions.
OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "power_calc",
        "description": "Estimate equal-allocation sample size or MDE for a two-sided test.",
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": ["proportion", "mean"]},
                "alpha": {"type": "number", "default": 0.05},
                "power": {"type": "number", "default": 0.8},
                "baseline": {"type": "number"},
                "mde": {"type": "number"},
                "n_per_arm": {"type": "integer"},
                "sigma": {"type": "number"},
            },
            "required": ["metric"],
        },
    },
}]

# Copy 2, Anthropic Messages. Same fields, different envelope.
ANTHROPIC_TOOLS = [{
    "name": "power_calc",
    "description": "Estimate equal-allocation sample size or MDE for a two-sided test.",
    "input_schema": OPENAI_TOOLS[0]["function"]["parameters"],
}]

def dispatch(name: str, arguments: dict) -> dict:
    if name != "power_calc":
        raise ValueError(name)
    response = httpx.post(f"{BASE}/v1/power", json=arguments, headers=AUTH)
    response.raise_for_status()
    return response.json()
```

That sketch has no `readOnlyHint`, no scope check, no progress channel, and no way for the server to ask the user a question except by inventing another field. Each of those is a further convention between one client and this API. The MCP server already has them, for every client that speaks the protocol.
