# Available experiment tools and resources

**Session ID:** ses_e6120662effeJDfEzOLOk5hRYA
**Created:** 11/2/2026, 7:46:56 PM
**Updated:** 11/2/2026, 7:50:28 PM

---

## User

List the experiment tools, resources, and prompts you have. Do not call any tool yet.

---

## Assistant

_Thinking:_

I should answer honestly from what's visible in the catalog rather than making calls - I can note that experiment-mcp shows only 5 of 8 tools listed, and I haven't actually checked resources or prompts since no prompt listing tool exists in the catalog.



I haven't called anything. Here's what I can see from the tool catalog.

**experiment-mcp tools (5 of 8 visible)**
- `assign_cells`: assigns units to arms with a salted SHA-256 bucket. Takes `unit_ids`, `salt`, and `splits` (weights summing to 1).
- `create_experiment`: stores a running experiment design. It does not assign traffic by itself. Takes `name`, `hypothesis`, `primary_metric` (`redeemed` or `post_spend`), and optional `offer_id`, `salt`, `traffic_split`, and `notes`.
- `close_experiment`: marks an experiment closed. It asks the user to confirm unless `confirm` is set to true.
- `draft_readout`: asks the client model to summarize an experiment, with a template fallback.
- `list_experiments`: lists stored experiments, optionally filtered by `running` or `closed`.

The server instructions also mention `power_calc`, `check_srm`, and `analyze_lift`. Those are probably the three hidden tools, but I haven't confirmed them. Finding them would require a `search` call, which I didn't make.

**Resources**
I don't have a resource listing for experiment-mcp. The `opencode` namespace has `list_mcp_resources` and `read_mcp_resource`, which could list them, but I haven't run them.

**Prompts**
I have no tool for listing MCP prompts, so I can't tell you whether this server exposes any.

Let me know if you want me to run `search` to reveal the hidden tools or call `list_mcp_resources` next.

---

## User

How many customers per arm do I need to detect a 2 percentage point redemption lift from a 10% baseline at 80% power and alpha 0.05?

---

## Assistant

_Thinking:_

I don't have access to a power calculation tool here, so I'll compute this by hand using the standard two-proportion sample size formula, plugging in p1=0.10, p2=0.12, their average and complement, and the critical z-values for 95% confidence and 80% power.

Working through the arithmetic: the numerator terms sum to about 1.239, squared gives roughly 1.536, divided by the squared difference in proportions (0.0004) yields approximately 3841 participants needed per arm. I'll present this hand calculation clearly and note that I'm happy to verify it another way if needed.



About **3,841 customers per arm** (about 7,682 total for a two-arm test).

This is the standard two-proportion sample size formula, done by hand. Inputs: baseline 10%, treatment 12%, two-sided alpha 0.05 (z = 1.96), and 80% power (z = 0.84). I haven't run the server's `power_calc` tool to confirm the number. Since it's a hidden tool, I'd need a `search` call to find it. I can do that if you want the tool's output as a cross-check.

Assumptions that change the number:
- **Two-sided test.** A one-sided test would need fewer customers.
- **No CUPED adjustment.** A pre-period covariate could reduce the required sample.
- **Every customer is exposed.** If only a fraction of assigned customers redeem-eligible, the required assignment size goes up.

---

## User

Create a running experiment named "Weekend 15". Hypothesis: the weekend offer raises redemption. Primary metric redeemed. Offer WEEKEND_15.

---

## Assistant

**Tool: execute**

**Input:**
```json
{
  "code": "return await tools[\"experiment-mcp\"].create_experiment({ name: \"Weekend 15\", hypothesis: \"the weekend offer raises redemption\", primary_metric: \"redeemed\", offer_id: \"WEEKEND_15\" });"
}
```

{
  "id": "exp_fdd364052e8e",
  "name": "Weekend 15",
  "hypothesis": "the weekend offer raises redemption",
  "primary_metric": "redeemed",
  "offer_id": "WEEKEND_15",
  "salt": "c81c89ed4a709319",
  "traffic_split": {
    "control": 0.5,
    "treatment": 0.5
  },
  "status": "running",
  "created_at": "2026-11-03T00:48:33.364923+00:00",
  "closed_at": null,
  "notes": ""
}

---

## Assistant

I created the experiment "Weekend 15" and it's running.

- **ID:** `exp_fdd364052e8e`
- **Hypothesis:** the weekend offer raises redemption
- **Primary metric:** redeemed
- **Offer:** WEEKEND_15
- **Traffic split:** 50% control, 50% treatment (the default, since I didn't specify one)
- **Salt:** `c81c89ed4a709319`

This only records the design. No customers are assigned yet. If you want a different split, tell me and I can recreate it with that split.

---

## User

Using experiment-mcp, estimate the redemption lift for offer WEEKEND_15 with a bootstrap confidence interval, 200 replicates, seed 1.

---

## Assistant

_Thinking:_

Since analyze_lift isn't directly visible, I should search for it to find its location and signature.



**Tool: execute**

**Input:**
```json
{
  "code": "return search({ query: \"analyze_lift\", namespace: \"experiment-mcp\" });"
}
```

{
  "items": [
    {
      "path": "tools[\"experiment-mcp\"].analyze_lift",
      "description": "Estimate absolute and relative lift with a confidence interval.\n\n        Pass offer_id to read the bundled synthetic exposures, or pass control\n        and treatment arrays. Bootstrap intervals report progress and stop\n        between batches if the client cancels. CUPED uses a pre-period\n        covariate when one is available. Reported arm means stay on the\n        original scale; the lift uses adjusted outcomes when CUPED is on.\n        ",
      "signature": "tools[\"experiment-mcp\"].analyze_lift({\n  /** redeemed or post_spend for an offer. Any label for raw arrays. @default \"redeemed\" */\n  outcome?: string,\n  /** Offer id in the bundled dataset. @default null */\n  offer_id?: string | null,\n  /** Control outcomes. @default null */\n  control?: Array<number> | null,\n  /** Treatment outcomes. @default null */\n  treatment?: Array<number> | null,\n  /** Control CUPED covariate. @default null */\n  control_covariate?: Array<number> | null,\n  /** Treatment CUPED covariate. @default null */\n  treatment_covariate?: Array<number> | null,\n  /** Adjust both arms with a pooled CUPED theta. @default false */\n  use_cuped?: boolean,\n  /** @default \"normal\" */\n  ci_method?: \"normal\" | \"bootstrap\",\n  /** @default 1000 @integer @minimum 100 @maximum 10000 */\n  bootstrap_iterations?: number,\n  /** @default 0 @integer */\n  seed?: number,\n  /** @default 0.05 @exclusiveMinimum 0 @exclusiveMaximum 1 */\n  alpha?: number,\n  /** @default \"control\" */\n  control_arm?: string,\n  /** @default \"treatment\" */\n  treatment_arm?: string,\n}): Promise<{\n  outcome: string,\n  ci_method: \"normal\" | \"bootstrap\",\n  alpha: number,\n  use_cuped: boolean,\n  /** @default null */\n  cuped_theta?: number | null,\n  control: {\n    /** @integer */\n    n: number,\n    mean: number,\n    variance: number,\n  },\n  treatment: {\n    /** @integer */\n    n: number,\n    mean: number,\n    variance: number,\n  },\n  absolute_lift: number,\n  absolute_ci_low: number,\n  absolute_ci_high: number,\n  /** @default null */\n  relative_lift?: number | null,\n  /** @default null */\n  relative_ci_low?: number | null,\n  /** @default null */\n  relative_ci_high?: number | null,\n  /** @default null */\n  bootstrap_iterations?: number | null,\n  /** @default false */\n  cancelled?: boolean,\n}>"
    }
  ],
  "remaining": 0,
  "next": null
}

---

## Assistant

**Tool: execute**

**Input:**
```json
{
  "code": "return await tools[\"experiment-mcp\"].analyze_lift({ outcome: \"redeemed\", offer_id: \"WEEKEND_15\", ci_method: \"bootstrap\", bootstrap_iterations: 200, seed: 1 });"
}
```

{
  "outcome": "redeemed",
  "ci_method": "bootstrap",
  "alpha": 0.05,
  "use_cuped": false,
  "cuped_theta": null,
  "control": {
    "n": 2016,
    "mean": 0.0882936507936508,
    "variance": 0.08053783134428294
  },
  "treatment": {
    "n": 1984,
    "mean": 0.13155241935483872,
    "variance": 0.1143039932165341
  },
  "absolute_lift": 0.04325876856118792,
  "absolute_ci_low": 0.023735919098822326,
  "absolute_ci_high": 0.06381668426779313,
  "relative_lift": 0.4899420079739036,
  "relative_ci_low": 0.2450511786600496,
  "relative_ci_high": 0.8162941462736377,
  "bootstrap_iterations": 200,
  "cancelled": false
}

---

## Assistant

Using the bundled **synthetic** WEEKEND_15 data, the redemption lift is positive and its 95% bootstrap interval excludes zero (200 replicates, seed 1):

| | Control | Treatment |
|---|---|---|
| n | 2,016 | 1,984 |
| Redemption rate | 8.8% | 13.2% |

- **Absolute lift:** +4.3 percentage points (95% CI: +2.4 to +6.4 pp)
- **Relative lift:** +49% (95% CI: +25% to +82%)

Since this is synthetic data, these numbers don't show real customer behavior or a measured incrementality effect. The bootstrap used only 200 replicates, so the interval endpoints will shift a bit with more replicates or a different seed. CUPED was off.

---

## User

Using experiment-mcp, draft a readout for experiment exp_fdd364052e8e

---

## Assistant

**Tool: execute**

**Input:**
```json
{
  "code": "return await tools[\"experiment-mcp\"].draft_readout({ experiment_id: \"exp_fdd364052e8e\" });"
}
```

{
  "experiment_id": "exp_fdd364052e8e",
  "source": "fallback",
  "text": "Readout draft (analyst). Source: template, because the client did not provide sampling.\n\nComputed facts:\nexperiment_id: exp_fdd364052e8e\nname: Weekend 15\nstatus: running\nhypothesis: the weekend offer raises redemption\nprimary_metric: redeemed\noffer_id: WEEKEND_15\nsalt: c81c89ed4a709319\ntraffic_split: {\"control\": 0.5, \"treatment\": 0.5}\ndataset_origin: synthetic\nThe dataset bundled with this server is synthetic. Do not call these figures real-world results.\nunadjusted: {\"outcome\":\"redeemed\",\"ci_method\":\"normal\",\"alpha\":0.05,\"use_cuped\":false,\"cuped_theta\":null,\"control\":{\"n\":2016,\"mean\":0.0882936507936508,\"variance\":0.08053783134428294},\"treatment\":{\"n\":1984,\"mean\":0.13155241935483872,\"variance\":0.1143039932165341},\"absolute_lift\":0.04325876856118792,\"absolute_ci_low\":0.023899500649374576,\"absolute_ci_high\":0.06261803647300127,\"relative_lift\":0.4899420079739036,\"relative_ci_low\":0.22144648721957544,\"relative_ci_high\":0.7584375287282317,\"bootstrap_iterations\":null,\"cancelled\":false}\ncuped: {\"outcome\":\"redeemed\",\"ci_method\":\"normal\",\"alpha\":0.05,\"use_cuped\":true,\"cuped_theta\":-0.0002891484433280224,\"control\":{\"n\":2016,\"mean\":0.0882936507936508,\"variance\":0.08053783134428294},\"treatment\":{\"n\":1984,\"mean\":0.13155241935483872,\"variance\":0.1143039932165341},\"absolute_lift\":0.043174915468079514,\"absolute_ci_low\":0.023816266675240297,\"absolute_ci_high\":0.06253356426091873,\"relative_lift\":0.48876206739041056,\"relative_ci_low\":0.22051006869919854,\"relative_ci_high\":0.7570140660816226,\"bootstrap_iterations\":null,\"cancelled\":false}\n\nThis draft does not add estimates beyond the facts above.",
  "facts": "experiment_id: exp_fdd364052e8e\nname: Weekend 15\nstatus: running\nhypothesis: the weekend offer raises redemption\nprimary_metric: redeemed\noffer_id: WEEKEND_15\nsalt: c81c89ed4a709319\ntraffic_split: {\"control\": 0.5, \"treatment\": 0.5}\ndataset_origin: synthetic\nThe dataset bundled with this server is synthetic. Do not call these figures real-world results.\nunadjusted: {\"outcome\":\"redeemed\",\"ci_method\":\"normal\",\"alpha\":0.05,\"use_cuped\":false,\"cuped_theta\":null,\"control\":{\"n\":2016,\"mean\":0.0882936507936508,\"variance\":0.08053783134428294},\"treatment\":{\"n\":1984,\"mean\":0.13155241935483872,\"variance\":0.1143039932165341},\"absolute_lift\":0.04325876856118792,\"absolute_ci_low\":0.023899500649374576,\"absolute_ci_high\":0.06261803647300127,\"relative_lift\":0.4899420079739036,\"relative_ci_low\":0.22144648721957544,\"relative_ci_high\":0.7584375287282317,\"bootstrap_iterations\":null,\"cancelled\":false}\ncuped: {\"outcome\":\"redeemed\",\"ci_method\":\"normal\",\"alpha\":0.05,\"use_cuped\":true,\"cuped_theta\":-0.0002891484433280224,\"control\":{\"n\":2016,\"mean\":0.0882936507936508,\"variance\":0.08053783134428294},\"treatment\":{\"n\":1984,\"mean\":0.13155241935483872,\"variance\":0.1143039932165341},\"absolute_lift\":0.043174915468079514,\"absolute_ci_low\":0.023816266675240297,\"absolute_ci_high\":0.06253356426091873,\"relative_lift\":0.48876206739041056,\"relative_ci_low\":0.22051006869919854,\"relative_ci_high\":0.7570140660816226,\"bootstrap_iterations\":null,\"cancelled\":false}"
}

---

## Assistant

_Thinking:_

The draft readout came back as a template fallback, so I should flag it's synthetic and note the confidence intervals use the normal method rather than my bootstrap figures. I also notice the CUPED theta is minimal despite cuped being marked true, suggesting a covariate was used even though it's not clear one was actually supplied—worth a brief mention.



The readout tool returned a template draft, not a model-written summary, because the client didn't support sampling. Here it is:

**Readout: Weekend 15** (`exp_fdd364052e8e`, running)

- **Hypothesis:** the weekend offer raises redemption
- **Offer:** WEEKEND_15
- **Primary metric:** redeemed
- **Traffic split:** 50% control, 50% treatment
- **Data source:** synthetic. These figures are not real customer results.

**Results (normal-approximation CI, alpha 0.05)**

| | Control | Treatment |
|---|---|---|
| n | 2,016 | 1,984 |
| Redemption rate | 8.8% | 13.2% |

- **Absolute lift:** +4.3 pp (95% CI: +2.4 to +6.3 pp)
- **Relative lift:** +49% (95% CI: +22% to +76%)

**CUPED-adjusted:** absolute lift +4.3 pp (95% CI: +2.4 to +6.3 pp). The adjustment barely changes the estimate (theta ≈ -0.0003).

The CIs here come from the normal method, so they differ slightly from the bootstrap intervals I reported earlier. The draft doesn't add interpretation beyond these facts. It also doesn't address whether the result is significant beyond the CI or whether the experiment should be closed.
