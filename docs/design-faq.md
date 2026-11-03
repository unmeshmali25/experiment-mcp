# Design FAQ

How this server runs, and the choices behind the tools. Paths and names match the code in `src/experiment_mcp/`.

## What happens from client launch to a tool call?

A client such as opencode reads its config and starts `uv run experiment-mcp serve` as a child process. The default transport is stdio: JSON-RPC on stdin and stdout, with no socket. The process is `experiment-mcp serve` in `src/experiment_mcp/cli.py`, which calls `MCPServer.run(transport="stdio")`.

The client and server then exchange capabilities. A client on the pre-2026 protocol sends `initialize`. A client on the 2026-07-28 revision sends `server/discover`, with its capabilities on the request. This server, built with `MCPServer` in `src/experiment_mcp/server.py`, answers either one. The server always advertises tools, resources, and prompts. The client advertises sampling or elicitation only when it implements them. opencode, in the recorded run under `docs/demo-results/`, declares neither, so `draft_readout` and `close_experiment` take their fallbacks.

Next the client calls `tools/list`. Each tool comes back with its name, description, input schema, output schema, and annotations. Those are generated from the Python function, not from a hand-written catalog. The client hands that list to the model. Some clients, including opencode, insert a tool-search step and do not put every schema in the first prompt. That step is the client's. It is not a method this server implements.

The model emits a tool call. The client sends `tools/call`. The SDK validates the arguments against the schema, the Python function runs, and the structured result goes back to the client. The model writes the reply from that result.

## Where does the server run?

In stdio mode it is a local process. The client starts it and stops it. The SQLite file defaults to `.experiment_mcp/experiments.db` in the working directory (`EXPERIMENT_MCP_DB` or `--db` overrides that). The offer tables are the CSVs in `src/experiment_mcp/data/synthetic/`, loaded from the installed package unless `--data-dir` or `EXPERIMENT_MCP_DATA_DIR` points somewhere else.

Hosting is the other transport: `experiment-mcp serve --transport http`. That is Streamable HTTP from the same `MCPServer`. The HTTP process expects a bearer JWT unless `EXPERIMENT_MCP_ALLOW_INSECURE_NO_AUTH=1` is set. The `Dockerfile` runs that HTTP command and stores the database at `/data/experiments.db`. This repository does not record a deployment of that image.

## Who decides the arguments, and who does the math?

The model chooses the arguments. For a bootstrap readout it might send `ci_method` `bootstrap`, `bootstrap_iterations` `200`, and `seed` `1`. The client forwards that object. The server checks it against the tool schema and then runs `analyze_lift` in `src/experiment_mcp/server.py`, which calls `experiment_mcp.stats`. The model does not compute the means or the interval. It only formats what the tool returned.

A bootstrap replicate resamples each arm with replacement, at the original arm size, and recomputes the difference of means. On an `offer_id` that arm is one row per exposed customer. `iter_bootstrap_diffs` in `src/experiment_mcp/stats/bootstrap.py` repeats that `N` times with `numpy`'s Generator seeded by `seed`, so the same seed and data give the same replicates. The interval is the `alpha/2` and `1 - alpha/2` percentiles of those replicates. The default `alpha` is 0.05, so those are the 2.5th and 97.5th percentiles. The reported lift is still the full-sample difference. The interval is what changes relative to the normal approximation, which uses each arm's sample variance instead of the resampled percentiles. The tool reports progress after each batch (`Bootstrap replicate {done} of {total}`) because that loop is the long one.

## Why is the computation a server tool rather than giving the LLM the CSV?

The model is not a stable calculator. The same table can produce different arithmetic, or figures that were not in the file. `analyze_lift` on `offer_id` `WEEKEND_15` returns counts, means, a lift, and an interval. It does not return the customer rows. The same inputs and seed produce the same structured result, and `tests/test_stats.py` locks the formulas.

Sending the CSVs would also put about 4,000 customers into the prompt on every question. That costs tokens, hits context limits, and copies customer-level rows into the model transcript. The tool keeps those rows in the server process and returns aggregates.

## How does the server signal risky tools, and how is risk enforced?

`analyze_lift` reads the CSVs (or arrays the caller passed). It does not write SQLite. `create_experiment` inserts a row. `close_experiment` updates that row to `closed`.

Annotations on the tool are hints for the client:

- Read tools, including `analyze_lift`, set `readOnlyHint` true and `openWorldHint` false. They do not set `destructiveHint` or `idempotentHint`.
- `create_experiment` sets `readOnlyHint` false, `destructiveHint` false, `idempotentHint` false, and `openWorldHint` false.
- `close_experiment` sets `readOnlyHint` false, `destructiveHint` true, `idempotentHint` true, and `openWorldHint` false.

A client may use those hints to ask a person before it runs a call. The server does not rely on the client honoring them.

Enforcement in this process is separate. `close_experiment` asks for confirmation through elicitation before it updates the row. If the client cannot ask, the row stays open until the call is repeated with `confirm` true. That flag is an argument, so a caller that sets it is asserting the person already agreed. `create_experiment` does not elicit. On Streamable HTTP, `_ensure_scope` in `server.py` requires `experiments:read` for the read tools and `experiments:write` for `create_experiment` and `close_experiment`. A write token may also call read tools. stdio has no token, so the scope check does not run there.

## How would a write-heavy extension be made safe, and should it be a separate server?

This server does not launch offers or move traffic. `assign_cells` only returns an assignment. A tool that applied a traffic slice would be a different kind of write, and it is not implemented here. The safeguards below are the shape that write would need. They are not present in the current tools.

- Preview and apply as two steps, with a preview id the apply call must echo.
- Hard caps in the function: maximum units, maximum spend, allowed offer ids.
- A person in the loop. Elicitation is enough for a small change, the way `close_experiment` confirms a status update. A large launch needs an approval recorded outside the model transcript.
- Narrow scopes, for example `experiments:read`, `experiments:preview`, and `experiments:launch`, on short-lived tokens bound to a user. Today the server only defines `experiments:read` and `experiments:write`.
- An idempotency key so a retried launch does not apply twice.
- An audit log of who launched what.
- A way to roll the change back, and a kill switch tied to guardrail metrics.
- Staging as the default target, with production as an explicit argument.

More tools on this same `MCPServer` are a normal extension. A split pays off when the blast radius, the credentials, or the owning team should not be shared, or when one connection should not carry every tool. The middle ground is one codebase that registers a subset of tools for a given scope or profile. This server does not do that. Every connected client receives the full tool list, and HTTP enforces scopes only when the tool runs.

## What happens as the tool count grows?

Eight tools is a short list. At something like 60, three things show up. The model picks a neighboring tool. Every turn pays for schemas the question did not need. Overlapping names make the wrong call look reasonable.

Mitigations that fit this design:

- Fewer tools, each aimed at an outcome, instead of one tool per internal function.
- Descriptions that say when to call the tool and when not to. The docstring is what `tools/list` sends.
- Client-side tool search or lazy loading, as opencode does. The server still lists everything when asked.
- A second server, or a profile that exposes a subset, when one connection's catalog gets too large.
- Tool-selection checks. `evals/tasks.json` pairs a task with the tool and arguments it should produce. `python -m experiment_mcp.evals` compares `evals/recorded.json` to that spec and runs the calls. It measures the current eight tools. It does not, by itself, say what happens at 60. A larger catalog would add cases there before adding the tools.
