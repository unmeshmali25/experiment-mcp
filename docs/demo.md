# Demo

A short pass over the server with Claude Desktop or Claude Code on stdio, then one HTTP call. Install and the client config are in the README. The dataset is the committed synthetic offer `WEEKEND_15`.

Figures below are from that file (generator seed `20261102`). They are simulated outcomes, not a measured experiment.

## 1. See what the server exposes

Prompt:

```text
List the experiment tools, resources, and prompts you have. Do not call any tool yet.
```

The client should name these tools: `power_calc`, `assign_cells`, `analyze_lift`, `check_srm`, `list_experiments`, `create_experiment`, `close_experiment`, `draft_readout`.

Resources: `dataset://schema`, `dataset://summary`, `experiment://index`, and the templates `experiment://{experiment_id}` and `experiment://{experiment_id}/results`.

Prompts: `design_experiment` and `write_readout`.

In MCP Inspector, the same lists are the Tools, Resources, and Prompts tabs after connecting to `uv run experiment-mcp serve`.

## 2. Sample size

Prompt:

```text
How many customers per arm do I need to detect a 2 percentage point redemption lift from a 10% baseline at 80% power and alpha 0.05?
```

Expected tool: `power_calc` with `metric` `proportion`, `baseline` `0.1`, `mde` `0.02`, `alpha` `0.05`, `power` `0.8`.

Expected structured result: `solved_for` is `n_per_arm`, `n_per_arm` is `3841`. The formula text describes an equal-allocation two-sided normal approximation.

## 3. Store an experiment

Prompt:

```text
Create a running experiment named "Weekend 15". Hypothesis: the weekend offer raises redemption. Primary metric redeemed. Offer WEEKEND_15.
```

Expected tool: `create_experiment`.

Expected result: `status` `running`, `primary_metric` `redeemed`, `offer_id` `WEEKEND_15`, a generated `salt`, and an `id` starting with `exp_`. Keep that id for the later steps. `experiment://index` and `experiment://{id}` now resolve.

## 4. Bootstrap lift

Prompt, with the id from step 3 if you want the readout tied to the row. The lift itself reads the offer file, not the SQLite row:

```text
Estimate the redemption lift for offer WEEKEND_15 with a bootstrap confidence interval, 200 replicates, seed 1.
```

Expected tool: `analyze_lift` with `offer_id` `WEEKEND_15`, `outcome` `redeemed`, `ci_method` `bootstrap`, `bootstrap_iterations` `200`, `seed` `1`.

Progress notifications, in order: `Bootstrap replicate 20 of 200`, then 40, 60, 80, 100, 120, 140, 160, 180, and 200. A client that does not render progress still returns the final result. The server log lines match those counts.

On this file the structured result has:

- `ci_method`: `bootstrap`
- `bootstrap_iterations`: `200`
- control n `2016`, mean about `0.0883`
- treatment n `1984`, mean about `0.1316`
- `absolute_lift` about `0.04326`
- `absolute_ci_low` about `0.02374`, `absolute_ci_high` about `0.06382`
- `relative_lift` about `0.490`
- `relative_ci_low` about `0.245`, `relative_ci_high` about `0.816`

The point estimate matches the normal-approximation lift on the same rows. The interval is the bootstrap percentile interval for seed 1.

## 5. Draft a readout

Prompt:

```text
Draft a readout for experiment <id>.
```

Expected tool: `draft_readout` with that `experiment_id`.

If the client declares sampling, `source` is `sampling` and `text` is whatever that client's model returns. The server does not call a model API. The fact sheet in `facts` includes `dataset_origin: synthetic` and the unadjusted and CUPED lift JSON for `redeemed`, the experiment's primary metric.

If the client does not declare sampling, `source` is `fallback`. `text` starts with `Readout draft (analyst). Source: template, because the client did not provide sampling.` and ends with `This draft does not add estimates beyond the facts above.`

## 6. Close the experiment

Prompt:

```text
Close experiment <id>.
```

Expected tool: `close_experiment` with `confirm` false or omitted.

If the client supports form elicitation, it shows: `Close experiment <id> (Weekend 15)? The record is kept and the status becomes closed.` The form field is `confirm` (boolean). Submit `true`.

Expected result: `closed` true, `status` `closed`, `reason` `closed`, detail `The experiment is closed. The record was kept.`

Submit `false`, or decline or cancel the form, and the row stays `running` (`reason` `not_confirmed`, `declined`, or `cancelled`).

If the client cannot elicit, the tool does not close the row. `reason` is `confirmation_required` and the detail says to call again with `confirm` true only after a person has agreed. A second call with `confirm` true then returns `reason` `closed`.

## 7. Read-only HTTP token

stdio does not check tokens. This step uses Streamable HTTP.

Terminal A:

```bash
uv run experiment-mcp dev-auth
```

Leave it on 127.0.0.1:9000. Copy the printed `EXPERIMENT_MCP_DEV_AUTH_SECRET`.

Terminal B:

```bash
export EXPERIMENT_MCP_ISSUER_URL=http://127.0.0.1:9000
export EXPERIMENT_MCP_RESOURCE_URL=http://127.0.0.1:8000/mcp
export EXPERIMENT_MCP_JWKS_URL=http://127.0.0.1:9000/.well-known/jwks.json
uv run experiment-mcp serve --transport http --host 127.0.0.1 --port 8000
```

Mint a read-only token:

```bash
curl -s -X POST http://127.0.0.1:9000/token \
  -H "Authorization: Bearer $EXPERIMENT_MCP_DEV_AUTH_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"resource":"http://127.0.0.1:8000/mcp","scope":"experiments:read","sub":"reader"}'
```

The JSON `scope` is `experiments:read`. Point an MCP HTTP client at `http://127.0.0.1:8000/mcp` with `Authorization: Bearer <access_token>`.

`power_calc` or `list_experiments` succeeds.

`create_experiment` with a name, a hypothesis, and `primary_metric` `redeemed` fails as a tool error. The text includes `Missing required scope 'experiments:write'`.

A token minted with `"scope":"experiments:write"` includes `experiments:read` as well. That token can create an experiment. A missing or invalid bearer token is HTTP 401 on `/mcp`, before a tool runs. `GET /.well-known/oauth-protected-resource/mcp` lists both scopes.
