"""MCP layer: tools, resources, prompts, completions, sampling, and elicitation.

Statistics live in ``experiment_mcp.stats``. This module only adapts them to
the Model Context Protocol.
"""

import json
import logging
import secrets
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import numpy as np
from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.routes import create_protected_resource_routes
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import (
    AcceptedElicitation,
    CancelledElicitation,
    Context,
    DeclinedElicitation,
    Elicit,
    ElicitationResult,
    Resolve,
    Sample,
)
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError, ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPDeprecationWarning
from mcp.types import (
    Completion,
    CompletionArgument,
    CompletionContext,
    CreateMessageResult,
    PromptReference,
    ResourceTemplateReference,
    SamplingMessage,
    TextContent,
    ToolAnnotations,
)
from pydantic import AnyHttpUrl, BaseModel, Field
from starlette.applications import Starlette

from experiment_mcp import __version__
from experiment_mcp.auth import AuthConfig, JwtTokenVerifier
from experiment_mcp.dataset import OfferDataset, packaged_data_dir
from experiment_mcp.models import (
    AssignmentResult,
    CloseResult,
    ExperimentList,
    ExperimentRecord,
    LiftResult,
    PowerResult,
    ReadoutDraft,
    SrmResult,
)
from experiment_mcp.scopes import READ_SCOPE, SUPPORTED_SCOPES, WRITE_SCOPE
from experiment_mcp.stats.assignment import assign_units
from experiment_mcp.stats.bootstrap import iter_bootstrap_diffs, percentile_interval
from experiment_mcp.stats.lift import analyze_lift as compute_lift
from experiment_mcp.stats.lift import apply_cuped
from experiment_mcp.stats.power import power_calculation
from experiment_mcp.stats.srm import sample_ratio_mismatch
from experiment_mcp.store import ExperimentStore

logger = logging.getLogger(__name__)

INSTRUCTIONS = """You help design and read retail-offer experiments.
The bundled dataset is synthetic. Never describe its results as real customer behavior
or as a measured incrementality claim.
Use power_calc for sample size or MDE, assign_cells for a deterministic split,
check_srm after assignment, and analyze_lift for absolute and relative lift.
analyze_lift can adjust for a pre-period covariate with CUPED and can report a percentile bootstrap interval.
create_experiment records a design. close_experiment asks the user to confirm before it changes status.
draft_readout asks the client model to summarize computed facts, and falls back to a template
when sampling is unavailable.
"""

_READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_CREATE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_CLOSE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)

_DATASET_SCHEMA = json.dumps(
    {
        "origin": "synthetic",
        "warning": "The bundled tables are generated. They are not a public customer extract.",
        "tables": {
            "customers.csv": {
                "customer_id": "string, synthetic id",
                "segment": "new | active | lapsed",
                "region": "NE | SE | MW | W",
                "pre_period_spend": "numeric covariate for CUPED, simulated pre-period spend",
            },
            "exposures.csv": {
                "customer_id": "string",
                "offer_id": "string",
                "arm": "control | treatment, assigned by the generator's salted hash",
                "redeemed": "0 or 1",
                "post_spend": "numeric spend in the offer window, simulated",
            },
        },
        "optional_public_data": (
            "dunnhumby The Complete Journey can be mapped onto these columns from a local copy. "
            "This server does not download it."
        ),
    },
    indent=2,
)

_METRICS = ("redeemed", "post_spend")
_AUDIENCES = ("executive", "analyst", "engineer")
_POPULATIONS = ("all_customers", "new", "active", "lapsed")
_EXPERIMENT_TEMPLATE = "experiment://{experiment_id}"
_RESULTS_TEMPLATE = "experiment://{experiment_id}/results"


class CloseForm(BaseModel):
    confirm: bool = Field(description="True closes the experiment. False leaves it running.")


class CloseDecision(BaseModel):
    confirm: bool
    via: Literal["argument", "fallback", "missing", "already_closed"]


@dataclass
class AppState:
    store: ExperimentStore
    dataset: OfferDataset


class _Runtime:
    def __init__(self) -> None:
        self.state: AppState | None = None


def _ensure_scope(required: str, *, error: type[Exception]) -> None:
    token = get_access_token()
    if token is None:
        return
    scopes = set(token.scopes)
    allowed = required in scopes or (required == READ_SCOPE and WRITE_SCOPE in scopes)
    if not allowed:
        raise error(
            f"Missing required scope '{required}'. "
            "Read tools accept experiments:read or experiments:write. "
            "Write tools require experiments:write."
        )


def _state(ctx: Context) -> AppState:
    state = ctx.request_context.lifespan_context
    if not isinstance(state, AppState):
        raise ToolError("server lifespan is not initialized")
    return state


def _supports_form_elicitation(ctx: Context) -> bool:
    capabilities = ctx.client_capabilities
    if capabilities is None or capabilities.elicitation is None:
        return False
    elicitation = capabilities.elicitation
    return elicitation.form is not None or elicitation.url is None


def _supports_sampling(ctx: Context) -> bool:
    capabilities = ctx.client_capabilities
    return capabilities is not None and capabilities.sampling is not None


def _client_cancelled(ctx: Context) -> bool:
    """True when the peer asked this request to stop.

    Async cancellation also arrives as a cancelled await. This check covers the
    signal-only path, where the SDK sets an event and lets the handler finish.
    """
    request = ctx.request_context
    for candidate in (ctx, request, request.session):
        event = getattr(candidate, "cancel_requested", None)
        is_set = getattr(event, "is_set", None)
        if callable(is_set) and bool(is_set()):
            return True
    return False


async def _emit_log(ctx: Context, message: str) -> None:
    logger.info(message)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MCPDeprecationWarning)
            await ctx.info(message)
    except Exception:
        logger.debug("MCP log notification was not delivered", exc_info=True)


async def _publish(ctx: Context, *uris: str) -> None:
    try:
        await ctx.notify_resources_changed()
        for uri in uris:
            await ctx.notify_resource_updated(uri)
    except Exception:
        logger.debug("subscription publish failed", exc_info=True)
    try:
        await ctx.session.send_resource_list_changed()
        for uri in uris:
            await ctx.session.send_resource_updated(uri)
    except Exception:
        logger.debug("legacy resource notification skipped", exc_info=True)


def _validate_split(splits: dict[str, float]) -> dict[str, float]:
    if len(splits) < 2:
        raise ToolError("traffic_split needs at least two arms")
    if any(weight <= 0 for weight in splits.values()):
        raise ToolError("traffic weights must be positive")
    total = sum(splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ToolError(f"traffic weights must sum to 1 (got {total})")
    return splits


def _prefix(values: list[str] | tuple[str, ...], typed: str) -> list[str]:
    return [value for value in values if value.startswith(typed)][:100]


def _facts(state: AppState, record: ExperimentRecord) -> str:
    lines = [
        f"experiment_id: {record.id}",
        f"name: {record.name}",
        f"status: {record.status}",
        f"hypothesis: {record.hypothesis}",
        f"primary_metric: {record.primary_metric}",
        f"offer_id: {record.offer_id or 'none'}",
        f"salt: {record.salt}",
        f"traffic_split: {json.dumps(record.traffic_split)}",
        f"dataset_origin: {state.dataset.origin}",
        "The dataset bundled with this server is synthetic. Do not call these figures real-world results.",
    ]
    if record.offer_id is None:
        lines.append("No offer is linked, so no lift was computed.")
        return "\n".join(lines)
    try:
        control, treatment, covariate_c, covariate_t = state.dataset.outcome_arrays(
            record.offer_id, record.primary_metric
        )
    except ValueError as exc:
        lines.append(f"Lift was not computed: {exc}")
        return "\n".join(lines)
    plain = compute_lift(outcome=record.primary_metric, control=control, treatment=treatment)
    adjusted = compute_lift(
        outcome=record.primary_metric,
        control=control,
        treatment=treatment,
        control_covariate=covariate_c,
        treatment_covariate=covariate_t,
    )
    lines.append("unadjusted: " + plain.model_dump_json())
    lines.append("cuped: " + adjusted.model_dump_json())
    return "\n".join(lines)


def _fallback_readout(facts: str, audience: str) -> str:
    return (
        f"Readout draft ({audience}). Source: template, because the client did not provide sampling.\n\n"
        "Computed facts:\n"
        f"{facts}\n\n"
        "This draft does not add estimates beyond the facts above."
    )


def build_server(
    *,
    db_path: Path,
    data_dir: Path | None = None,
    auth: AuthConfig | None = None,
) -> MCPServer:
    """Build a server bound to one SQLite file and one dataset directory."""
    runtime = _Runtime()
    dataset_dir = data_dir or packaged_data_dir()

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[AppState]:
        state = AppState(store=ExperimentStore(db_path), dataset=OfferDataset.load(dataset_dir))
        if state.dataset.origin != "synthetic":
            logger.warning("Dataset origin is %r, not 'synthetic'", state.dataset.origin)
        runtime.state = state
        try:
            yield state
        finally:
            runtime.state = None

    token_verifier = JwtTokenVerifier(auth) if auth is not None else None
    auth_settings = None
    if auth is not None:
        auth_settings = AuthSettings(
            issuer_url=AnyHttpUrl(auth.issuer),
            resource_server_url=AnyHttpUrl(auth.resource),
            required_scopes=[],
            validate_token_resource=True,
        )

    mcp = MCPServer(
        "experiment-mcp",
        instructions=INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
        token_verifier=token_verifier,
        auth=auth_settings,
    )

    @mcp.tool(title="Sample size or MDE", annotations=_READ)
    async def power_calc(
        metric: Literal["proportion", "mean"],
        alpha: Annotated[float, Field(gt=0, lt=1, description="Two-sided type I error.")] = 0.05,
        power: Annotated[float, Field(gt=0, lt=1, description="Desired power, 1 - beta.")] = 0.8,
        baseline: Annotated[
            float | None,
            Field(gt=0, lt=1, description="Control rate. Required for proportion designs."),
        ] = None,
        mde: Annotated[
            float | None,
            Field(description="Absolute difference to detect. Omit when solving for MDE."),
        ] = None,
        n_per_arm: Annotated[
            int | None,
            Field(ge=2, description="Per-arm sample size. Omit when solving for sample size."),
        ] = None,
        sigma: Annotated[
            float | None,
            Field(gt=0, description="Outcome standard deviation. Required for mean designs."),
        ] = None,
    ) -> PowerResult:
        """Estimate equal-allocation sample size or MDE for a two-sided test.

        Provide exactly one of mde or n_per_arm. The normal approximation does
        not include peeking, multiple metrics, or a finite-population correction.
        """
        _ensure_scope(READ_SCOPE, error=ToolError)
        try:
            return power_calculation(
                metric=metric,
                alpha=alpha,
                power=power,
                baseline=baseline,
                sigma=sigma,
                mde=mde,
                n_per_arm=n_per_arm,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool(title="Assign experiment cells", annotations=_READ)
    async def assign_cells(
        unit_ids: Annotated[list[str], Field(description="Stable unit ids, usually customer ids.")],
        salt: Annotated[str, Field(min_length=1, description="Experiment salt. Changing it reassigns units.")],
        splits: Annotated[
            dict[str, float],
            Field(description="Arm name to traffic weight. Weights must sum to 1."),
        ],
    ) -> AssignmentResult:
        """Assign units to arms with a salted SHA-256 bucket.

        Arms are ordered by name. The same salt and unit id always map to the
        same arm. This does not write an experiment record.
        """
        _ensure_scope(READ_SCOPE, error=ToolError)
        try:
            return assign_units(unit_ids, salt, splits)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool(title="Analyze lift", annotations=_READ)
    async def analyze_lift(
        ctx: Context,
        outcome: Annotated[
            str,
            Field(description="redeemed or post_spend for an offer. Any label for raw arrays."),
        ] = "redeemed",
        offer_id: Annotated[str | None, Field(description="Offer id in the bundled dataset.")] = None,
        control: Annotated[list[float] | None, Field(description="Control outcomes.")] = None,
        treatment: Annotated[list[float] | None, Field(description="Treatment outcomes.")] = None,
        control_covariate: Annotated[list[float] | None, Field(description="Control CUPED covariate.")] = None,
        treatment_covariate: Annotated[list[float] | None, Field(description="Treatment CUPED covariate.")] = None,
        use_cuped: Annotated[bool, Field(description="Adjust both arms with a pooled CUPED theta.")] = False,
        ci_method: Literal["normal", "bootstrap"] = "normal",
        bootstrap_iterations: Annotated[int, Field(ge=100, le=10_000)] = 1000,
        seed: int = 0,
        alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05,
        control_arm: str = "control",
        treatment_arm: str = "treatment",
    ) -> LiftResult:
        """Estimate absolute and relative lift with a confidence interval.

        Pass offer_id to read the bundled synthetic exposures, or pass control
        and treatment arrays. Bootstrap intervals report progress and stop
        between batches if the client cancels. CUPED uses a pre-period
        covariate when one is available. Reported arm means stay on the
        original scale; the lift uses adjusted outcomes when CUPED is on.
        """
        _ensure_scope(READ_SCOPE, error=ToolError)
        using_offer = offer_id is not None
        using_arrays = control is not None or treatment is not None
        if using_offer == using_arrays:
            raise ToolError("Provide either offer_id or both control and treatment arrays.")
        try:
            if using_offer:
                assert offer_id is not None
                if outcome not in {"redeemed", "post_spend"}:
                    raise ValueError("offer analysis outcome must be redeemed or post_spend")
                control_arr, treatment_arr, cov_c, cov_t = _state(ctx).dataset.outcome_arrays(
                    offer_id,
                    outcome,
                    control_arm=control_arm,
                    treatment_arm=treatment_arm,
                )
                if use_cuped:
                    control_covariate = cov_c.tolist()
                    treatment_covariate = cov_t.tolist()
                control_values = control_arr.tolist()
                treatment_values = treatment_arr.tolist()
            else:
                if control is None or treatment is None:
                    raise ValueError("array analysis needs both control and treatment")
                control_values = control
                treatment_values = treatment
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        covariate_c = control_covariate if use_cuped else None
        covariate_t = treatment_covariate if use_cuped else None
        if use_cuped and (covariate_c is None or covariate_t is None):
            raise ToolError("use_cuped requires control_covariate and treatment_covariate.")

        bootstrap_low = bootstrap_high = None
        relative_low = relative_high = None
        iterations: int | None = None
        if ci_method == "bootstrap":
            control_arr = np.asarray(control_values, dtype=np.float64)
            treatment_arr = np.asarray(treatment_values, dtype=np.float64)
            if use_cuped:
                assert covariate_c is not None and covariate_t is not None
                control_arr, treatment_arr, _theta = apply_cuped(
                    control_arr,
                    np.asarray(covariate_c, dtype=np.float64),
                    treatment_arr,
                    np.asarray(covariate_t, dtype=np.float64),
                )
            absolute_chunks: list[np.ndarray] = []
            relative_chunks: list[np.ndarray] = []
            batch = max(1, bootstrap_iterations // 10)
            await _emit_log(ctx, f"bootstrap started iterations={bootstrap_iterations} seed={seed}")
            try:
                stream = iter_bootstrap_diffs(
                    control_arr,
                    treatment_arr,
                    iterations=bootstrap_iterations,
                    seed=seed,
                    batch_size=batch,
                )
                for done, total, diffs, relative in stream:
                    if _client_cancelled(ctx):
                        await _emit_log(ctx, f"bootstrap cancellation requested at {done}/{total}")
                        raise anyio.get_cancelled_exc_class()()
                    absolute_chunks.append(diffs)
                    relative_chunks.append(relative)
                    await ctx.report_progress(
                        float(done),
                        float(total),
                        f"Bootstrap replicate {done} of {total}",
                    )
                    await _emit_log(ctx, f"bootstrap progress {done}/{total}")
                    await anyio.sleep(0)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            merged = np.concatenate(absolute_chunks)
            bootstrap_low, bootstrap_high = percentile_interval(merged, alpha)
            relative_merged = np.concatenate(relative_chunks)
            if bool(np.isfinite(relative_merged).all()):
                relative_low, relative_high = percentile_interval(relative_merged, alpha)
            iterations = int(merged.size)
            await _emit_log(ctx, f"bootstrap finished replicates={iterations}")

        try:
            return compute_lift(
                outcome=outcome,
                control=control_values,
                treatment=treatment_values,
                alpha=alpha,
                control_covariate=covariate_c,
                treatment_covariate=covariate_t,
                ci_method=ci_method,
                bootstrap_iterations=iterations,
                bootstrap_low=bootstrap_low,
                bootstrap_high=bootstrap_high,
                relative_bootstrap_low=relative_low,
                relative_bootstrap_high=relative_high,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool(title="Check sample ratio", annotations=_READ)
    async def check_srm(
        counts: Annotated[dict[str, int], Field(description="Observed assignments per arm.")],
        expected_weights: Annotated[
            dict[str, float],
            Field(description="Intended traffic weights. Must sum to 1 and match the count keys."),
        ],
        alpha: Annotated[
            float,
            Field(gt=0, lt=1, description="Mismatch threshold. 0.001 is the usual SRM bar."),
        ] = 0.001,
    ) -> SrmResult:
        """Chi-square test of observed arm counts against intended weights.

        A mismatch means the counts are unlikely given the weights. It does not
        identify the cause.
        """
        _ensure_scope(READ_SCOPE, error=ToolError)
        try:
            return sample_ratio_mismatch(counts, expected_weights, alpha=alpha)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool(title="List experiments", annotations=_READ)
    async def list_experiments(
        ctx: Context,
        status: Literal["running", "closed"] | None = None,
    ) -> ExperimentList:
        """List experiment records stored in this server's SQLite file."""
        _ensure_scope(READ_SCOPE, error=ToolError)
        return ExperimentList(experiments=_state(ctx).store.list_experiments(status))

    @mcp.tool(title="Create experiment", annotations=_CREATE)
    async def create_experiment(
        ctx: Context,
        name: Annotated[str, Field(min_length=1)],
        hypothesis: Annotated[str, Field(min_length=1)],
        primary_metric: Literal["redeemed", "post_spend"],
        offer_id: str | None = None,
        salt: Annotated[str | None, Field(min_length=1)] = None,
        traffic_split: dict[str, float] | None = None,
        notes: str = "",
    ) -> ExperimentRecord:
        """Store a running experiment. This does not assign traffic by itself."""
        _ensure_scope(WRITE_SCOPE, error=ToolError)
        state = _state(ctx)
        if offer_id is not None and offer_id not in state.dataset.offer_ids:
            known = ", ".join(state.dataset.offer_ids) or "(none)"
            raise ToolError(f"unknown offer_id {offer_id!r}. Known offers: {known}")
        split = _validate_split(traffic_split or {"control": 0.5, "treatment": 0.5})
        record = state.store.create(
            name=name,
            hypothesis=hypothesis,
            primary_metric=primary_metric,
            offer_id=offer_id,
            salt=salt or secrets.token_hex(8),
            traffic_split=split,
            notes=notes,
        )
        await _publish(
            ctx,
            "experiment://index",
            f"experiment://{record.id}",
            f"experiment://{record.id}/results",
        )
        return record

    async def resolve_close(
        experiment_id: str,
        confirm: bool,
        ctx: Context,
    ) -> Elicit[CloseForm] | CloseDecision:
        record = _state(ctx).store.get(experiment_id)
        if record is None:
            return CloseDecision(confirm=False, via="missing")
        if record.status == "closed":
            return CloseDecision(confirm=False, via="already_closed")
        if confirm:
            return CloseDecision(confirm=True, via="argument")
        if not _supports_form_elicitation(ctx):
            return CloseDecision(confirm=False, via="fallback")
        return Elicit(
            f"Close experiment {experiment_id} ({record.name})? The record is kept and the status becomes closed.",
            CloseForm,
        )

    @mcp.tool(title="Close experiment", annotations=_CLOSE)
    async def close_experiment(
        ctx: Context,
        experiment_id: Annotated[str, Field(min_length=1)],
        decision: Annotated[ElicitationResult[Any], Resolve(resolve_close)],
        confirm: Annotated[
            bool,
            Field(
                description=(
                    "Leave false to ask the user through elicitation. "
                    "Set true only after a person has agreed, which is the fallback "
                    "when the client cannot show a form."
                )
            ),
        ] = False,
    ) -> CloseResult:
        """Mark an experiment closed after the user confirms.

        The confirmation form is elicited when the client supports it. If the
        client does not, the experiment stays open until this tool is called
        again with confirm=true. Declining or cancelling the form does not close it.
        """
        _ensure_scope(WRITE_SCOPE, error=ToolError)
        state = _state(ctx)
        data = decision.data if isinstance(decision, AcceptedElicitation) else None
        if isinstance(data, CloseDecision) and data.via == "missing":
            return CloseResult(
                experiment_id=experiment_id,
                closed=False,
                reason="not_found",
                detail=f"No experiment with id {experiment_id!r}.",
            )
        if isinstance(data, CloseDecision) and data.via == "already_closed":
            return CloseResult(
                experiment_id=experiment_id,
                closed=False,
                status="closed",
                reason="already_closed",
                detail="The experiment is already closed.",
            )
        confirmed = False
        if isinstance(data, CloseDecision) and data.via == "argument" and data.confirm:
            confirmed = True
        elif isinstance(data, CloseForm) and data.confirm:
            confirmed = True
        elif isinstance(decision, DeclinedElicitation):
            return CloseResult(
                experiment_id=experiment_id,
                closed=False,
                status="running",
                reason="declined",
                detail="The user declined. The experiment is still running.",
            )
        elif isinstance(decision, CancelledElicitation):
            return CloseResult(
                experiment_id=experiment_id,
                closed=False,
                status="running",
                reason="cancelled",
                detail="The confirmation was cancelled. The experiment is still running.",
            )
        elif isinstance(data, CloseDecision) and data.via == "fallback":
            return CloseResult(
                experiment_id=experiment_id,
                closed=False,
                status="running",
                reason="confirmation_required",
                detail=(
                    "This client cannot ask the user to confirm. "
                    "Ask them yourself, then call close_experiment again with confirm=true."
                ),
            )
        elif isinstance(data, CloseForm) and not data.confirm:
            return CloseResult(
                experiment_id=experiment_id,
                closed=False,
                status="running",
                reason="not_confirmed",
                detail="The form was submitted with confirm=false. The experiment is still running.",
            )
        if not confirmed:
            return CloseResult(
                experiment_id=experiment_id,
                closed=False,
                status="running",
                reason="not_confirmed",
                detail="The experiment was not closed.",
            )
        updated = state.store.close(experiment_id)
        if updated is None:
            return CloseResult(
                experiment_id=experiment_id,
                closed=False,
                reason="not_found",
                detail=f"No experiment with id {experiment_id!r}.",
            )
        await _publish(
            ctx,
            "experiment://index",
            f"experiment://{experiment_id}",
            f"experiment://{experiment_id}/results",
        )
        return CloseResult(
            experiment_id=experiment_id,
            closed=True,
            status=updated.status,
            reason="closed",
            detail="The experiment is closed. The record was kept.",
        )

    async def resolve_draft(experiment_id: str, ctx: Context) -> Sample | str:
        state = _state(ctx)
        record = state.store.get(experiment_id)
        if record is None:
            raise ToolError(f"No experiment with id {experiment_id!r}.")
        facts = _facts(state, record)
        if not _supports_sampling(ctx):
            return _fallback_readout(facts, "analyst")
        prompt = (
            "Write a short readout for an analyst. Use only these facts. "
            "Do not invent numbers, customers, or a recommendation that the facts do not support.\n\n" + facts
        )
        return Sample(
            [SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
            max_tokens=500,
            system_prompt=(
                "You draft experiment readouts. Use only the figures in the user message. "
                "Say when a figure was not computed. The bundled dataset is synthetic unless the facts say otherwise."
            ),
        )

    @mcp.tool(title="Draft a readout", annotations=_READ)
    async def draft_readout(
        ctx: Context,
        experiment_id: Annotated[str, Field(min_length=1)],
        draft: Annotated[CreateMessageResult | str, Resolve(resolve_draft)],
    ) -> ReadoutDraft:
        """Ask the client model to summarize an experiment, with a template fallback.

        Sampling is deprecated in the 2026-07-28 spec and many clients do not
        offer it. Those clients get a deterministic draft built from the same facts.
        """
        _ensure_scope(READ_SCOPE, error=ToolError)
        record = _state(ctx).store.get(experiment_id)
        facts = "" if record is None else _facts(_state(ctx), record)
        if isinstance(draft, str):
            return ReadoutDraft(
                experiment_id=experiment_id,
                source="fallback",
                text=draft,
                facts=facts,
            )
        content = draft.content
        text = content.text if isinstance(content, TextContent) else str(content)
        return ReadoutDraft(
            experiment_id=experiment_id,
            source="sampling",
            text=text,
            facts=facts,
        )

    @mcp.resource(
        "dataset://schema",
        mime_type="application/json",
        description="Synthetic dataset column schema.",
    )
    async def dataset_schema() -> str:
        """Column definitions for the bundled synthetic tables."""
        _ensure_scope(READ_SCOPE, error=ResourceError)
        return _DATASET_SCHEMA

    @mcp.resource(
        "dataset://summary",
        mime_type="application/json",
        description="Row counts for the bundled synthetic dataset.",
    )
    async def dataset_summary() -> str:
        """Counts and the generator metadata. The origin field says synthetic."""
        _ensure_scope(READ_SCOPE, error=ResourceError)
        if runtime.state is None:
            raise ResourceError("server lifespan is not initialized")
        return json.dumps(runtime.state.dataset.summary(), indent=2)

    @mcp.resource(
        "experiment://index",
        mime_type="application/json",
        description="All experiment records in the local store.",
    )
    async def experiment_index() -> str:
        """List stored experiments. Changes when an experiment is created or closed."""
        _ensure_scope(READ_SCOPE, error=ResourceError)
        if runtime.state is None:
            raise ResourceError("server lifespan is not initialized")
        records = [record.model_dump() for record in runtime.state.store.list_experiments()]
        return json.dumps({"experiments": records}, indent=2)

    @mcp.resource(
        _EXPERIMENT_TEMPLATE,
        mime_type="application/json",
        description="One experiment record.",
    )
    async def read_experiment(experiment_id: str, ctx: Context) -> str:
        """Return one stored experiment."""
        _ensure_scope(READ_SCOPE, error=ResourceError)
        record = _state(ctx).store.get(experiment_id)
        if record is None:
            raise ResourceNotFoundError(f"No experiment {experiment_id}")
        return record.model_dump_json(indent=2)

    @mcp.resource(
        _RESULTS_TEMPLATE,
        mime_type="application/json",
        description="Normal-approximation lift for an experiment's linked offer.",
    )
    async def read_results(experiment_id: str, ctx: Context) -> str:
        """Compute unadjusted and CUPED lift for the linked offer, if there is one."""
        _ensure_scope(READ_SCOPE, error=ResourceError)
        state = _state(ctx)
        record = state.store.get(experiment_id)
        if record is None:
            raise ResourceNotFoundError(f"No experiment {experiment_id}")
        payload: dict[str, Any] = {
            "experiment_id": record.id,
            "dataset_origin": state.dataset.origin,
            "warning": "Synthetic data. Do not cite this as a real-world incrementality result.",
        }
        if record.offer_id is None:
            payload["available"] = False
            payload["reason"] = "This experiment has no offer_id, so there are no rows to analyze."
            return json.dumps(payload, indent=2)
        try:
            control, treatment, cov_c, cov_t = state.dataset.outcome_arrays(record.offer_id, record.primary_metric)
        except ValueError as exc:
            raise ResourceError(str(exc)) from exc
        plain = compute_lift(outcome=record.primary_metric, control=control, treatment=treatment)
        adjusted = compute_lift(
            outcome=record.primary_metric,
            control=control,
            treatment=treatment,
            control_covariate=cov_c,
            treatment_covariate=cov_t,
        )
        payload["available"] = True
        payload["unadjusted"] = plain.model_dump()
        payload["cuped"] = adjusted.model_dump()
        return json.dumps(payload, indent=2)

    @mcp.prompt(title="Design an experiment")
    async def design_experiment(
        metric: str,
        baseline: str,
        mde: str,
        population: str,
        ctx: Context,
    ) -> str:
        """Plan a retail-offer test before any readout is written."""
        _ensure_scope(READ_SCOPE, error=ToolError)
        del ctx
        return (
            "Design a retail-offer experiment. The dataset in this server is synthetic.\n"
            f"Metric: {metric}\n"
            f"Baseline: {baseline}\n"
            f"Minimum detectable effect (absolute): {mde}\n"
            f"Population: {population}\n\n"
            "Call power_calc with metric proportion when the metric is a rate such as redeemed, "
            "or metric mean when it is post_spend. Use the baseline and MDE above.\n"
            "Then call assign_cells with a fresh salt and a traffic split that sums to 1.\n"
            "Call check_srm on the resulting counts.\n"
            "If the user wants the design stored, call create_experiment. "
            "Do not invent a historical lift."
        )

    @mcp.prompt(title="Write a readout")
    async def write_readout(experiment_id: str, audience: str, ctx: Context) -> str:
        """Outline a readout that stays inside the computed facts."""
        _ensure_scope(READ_SCOPE, error=ToolError)
        del ctx
        return (
            f"Write a {audience} readout for experiment {experiment_id}.\n"
            f"Read experiment://{experiment_id} and experiment://{experiment_id}/results before you draft.\n"
            "You may call draft_readout. If its source is fallback, keep the template's numbers "
            "and do not add any.\n"
            "State that the bundled data is synthetic when that is what the resource says.\n"
            "Do not claim a business decision the interval does not support."
        )

    @mcp.completion()
    async def complete(
        ref: PromptReference | ResourceTemplateReference,
        argument: CompletionArgument,
        context: CompletionContext | None,
    ) -> Completion | None:
        _ensure_scope(READ_SCOPE, error=ToolError)
        del context
        state = runtime.state
        typed = argument.value
        if isinstance(ref, PromptReference) and ref.name == "design_experiment":
            if argument.name == "metric":
                return Completion(values=_prefix(_METRICS, typed))
            if argument.name == "population":
                return Completion(values=_prefix(_POPULATIONS, typed))
            return Completion(values=[])
        if isinstance(ref, PromptReference) and ref.name == "write_readout" and argument.name == "audience":
            return Completion(values=_prefix(_AUDIENCES, typed))
        wants_ids = (
            isinstance(ref, PromptReference) and ref.name == "write_readout" and argument.name == "experiment_id"
        ) or (
            isinstance(ref, ResourceTemplateReference)
            and ref.uri in {_EXPERIMENT_TEMPLATE, _RESULTS_TEMPLATE}
            and argument.name == "experiment_id"
        )
        if wants_ids:
            if state is None:
                return Completion(values=[])
            return Completion(values=_prefix(state.store.ids(), typed))
        return Completion(values=[])

    return mcp


def advertise_scopes(app: Starlette, *, resource_url: str, issuer_url: str) -> None:
    """Publish both scopes on the protected-resource metadata document.

    The SDK copies ``AuthSettings.required_scopes`` into that document and also
    requires every listed scope on every request. This server requires only a
    valid token at the HTTP layer and checks read versus write inside each tool,
    so the metadata route is replaced with one that advertises both scopes.
    """
    prefix = "/.well-known/oauth-protected-resource"
    routes = [route for route in app.router.routes if not str(getattr(route, "path", "")).startswith(prefix)]
    routes.extend(
        create_protected_resource_routes(
            resource_url=AnyHttpUrl(resource_url),
            authorization_servers=[AnyHttpUrl(issuer_url)],
            scopes_supported=list(SUPPORTED_SCOPES),
            resource_name="experiment-mcp",
        )
    )
    app.router.routes[:] = routes


def http_app(
    server: MCPServer,
    *,
    host: str,
    allowed_hosts: list[str],
    allowed_origins: list[str],
) -> Starlette:
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        host=host,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        ),
    )
    settings = server.settings.auth
    if settings is not None and settings.resource_server_url is not None:
        advertise_scopes(
            app,
            resource_url=str(settings.resource_server_url),
            issuer_url=str(settings.issuer_url),
        )
    return app
