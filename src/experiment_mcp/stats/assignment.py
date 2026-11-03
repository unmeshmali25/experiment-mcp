"""Deterministic hash assignment.

The bucket is ``SHA-256(salt + 0x1f + unit_id)``, taking the first 8 bytes as a
big-endian integer and dividing by 2**64. Arms are ordered by name. The last
arm receives every bucket at or above the previous cumulative weight, so the
weights do not have to sum to 1 with perfect binary precision.
"""

import hashlib

from experiment_mcp.models import AssignmentResult, AssignmentRow

_SEPARATOR = "\x1f"


def unit_bucket(unit_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}{_SEPARATOR}{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _thresholds(splits: dict[str, float]) -> list[tuple[str, float]]:
    if not splits:
        raise ValueError("splits must name at least one arm")
    if any(weight <= 0 for weight in splits.values()):
        raise ValueError("every arm weight must be positive")
    total = sum(splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"arm weights must sum to 1 (got {total})")
    ordered = sorted(splits)
    cumulative = 0.0
    thresholds: list[tuple[str, float]] = []
    for index, arm in enumerate(ordered):
        cumulative += splits[arm]
        if index == len(ordered) - 1:
            cumulative = 1.0
        thresholds.append((arm, cumulative))
    return thresholds


def arm_for_bucket(bucket: float, thresholds: list[tuple[str, float]]) -> str:
    for arm, ceiling in thresholds:
        if bucket < ceiling:
            return arm
    return thresholds[-1][0]


def assign_units(unit_ids: list[str], salt: str, splits: dict[str, float]) -> AssignmentResult:
    if not salt:
        raise ValueError("salt must be a non-empty string")
    if len(unit_ids) > 50_000:
        raise ValueError("unit_ids is capped at 50000 per call")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit_ids must be unique")
    thresholds = _thresholds(splits)
    rows: list[AssignmentRow] = []
    counts = {arm: 0 for arm, _ in thresholds}
    for unit_id in unit_ids:
        if not unit_id:
            raise ValueError("unit ids must be non-empty")
        bucket = unit_bucket(unit_id, salt)
        arm = arm_for_bucket(bucket, thresholds)
        counts[arm] += 1
        rows.append(AssignmentRow(unit_id=unit_id, arm=arm, bucket=bucket))
    return AssignmentResult(
        salt=salt,
        arm_order=[arm for arm, _ in thresholds],
        counts=counts,
        assignments=rows,
    )
