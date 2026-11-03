"""Pydantic models shared by the stats package and the MCP layer.

These types are the structured tool results. They do not import the MCP SDK.
"""

from typing import Literal

from pydantic import BaseModel, Field

MetricKind = Literal["proportion", "mean"]
OutcomeName = Literal["redeemed", "post_spend"]
CiMethod = Literal["normal", "bootstrap"]
PrimaryMetric = Literal["redeemed", "post_spend"]
ExperimentStatus = Literal["running", "closed"]


class PowerResult(BaseModel):
    """Sample size or minimum detectable effect from a two-sample normal approximation."""

    metric: MetricKind
    alpha: float
    power: float
    alternative: Literal["two-sided"] = "two-sided"
    allocation: Literal["equal"] = "equal"
    baseline: float | None = None
    sigma: float | None = None
    mde: float | None = Field(
        default=None,
        description="Absolute difference (treatment minus control) the design can detect.",
    )
    n_per_arm: int | None = None
    solved_for: Literal["n_per_arm", "mde"]
    formula: str


class AssignmentRow(BaseModel):
    unit_id: str
    arm: str
    bucket: float


class AssignmentResult(BaseModel):
    salt: str
    arm_order: list[str]
    counts: dict[str, int]
    assignments: list[AssignmentRow]


class GroupSummary(BaseModel):
    n: int
    mean: float
    variance: float


class LiftResult(BaseModel):
    """Absolute and relative lift with a confidence interval.

    Means are the unadjusted outcome means. When CUPED is on, the interval and
    the lift use the covariate-adjusted outcomes; the reported means stay on
    the original scale so a readout can show both.
    """

    outcome: str
    ci_method: CiMethod
    alpha: float
    use_cuped: bool
    cuped_theta: float | None = None
    control: GroupSummary
    treatment: GroupSummary
    absolute_lift: float
    absolute_ci_low: float
    absolute_ci_high: float
    relative_lift: float | None = None
    relative_ci_low: float | None = None
    relative_ci_high: float | None = None
    bootstrap_iterations: int | None = None
    cancelled: bool = False


class SrmResult(BaseModel):
    """Chi-square sample-ratio check.

    ``mismatch`` is true when the p-value is below ``alpha``. A small p-value
    is evidence the observed counts do not match the intended weights. It is
    not, by itself, proof of a bug.
    """

    counts: dict[str, int]
    expected_counts: dict[str, float]
    expected_weights: dict[str, float]
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    alpha: float
    mismatch: bool


class ExperimentRecord(BaseModel):
    id: str
    name: str
    hypothesis: str
    primary_metric: PrimaryMetric
    offer_id: str | None
    salt: str
    traffic_split: dict[str, float]
    status: ExperimentStatus
    created_at: str
    closed_at: str | None
    notes: str


class ExperimentSummary(BaseModel):
    id: str
    name: str
    status: ExperimentStatus
    primary_metric: PrimaryMetric
    offer_id: str | None
    created_at: str


class ExperimentList(BaseModel):
    experiments: list[ExperimentSummary]


class CloseResult(BaseModel):
    experiment_id: str
    closed: bool
    status: ExperimentStatus | None = None
    reason: Literal[
        "closed",
        "already_closed",
        "not_found",
        "confirmation_required",
        "declined",
        "cancelled",
        "not_confirmed",
    ]
    detail: str


class ReadoutDraft(BaseModel):
    experiment_id: str
    source: Literal["sampling", "fallback"]
    text: str
    facts: str
