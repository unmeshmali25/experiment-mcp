"""Load the committed synthetic retail-offer tables."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def packaged_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "synthetic"


@dataclass(frozen=True)
class Customer:
    customer_id: str
    segment: str
    region: str
    pre_period_spend: float


@dataclass(frozen=True)
class Exposure:
    customer_id: str
    offer_id: str
    arm: str
    redeemed: int
    post_spend: float


@dataclass(frozen=True)
class OfferDataset:
    """In-memory copy of the synthetic tables.

    ``origin`` is read from ``metadata.json`` and is expected to be ``synthetic``.
    """

    origin: str
    metadata: dict[str, object]
    customers: dict[str, Customer]
    exposures: tuple[Exposure, ...]

    @classmethod
    def load(cls, directory: Path) -> "OfferDataset":
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"No metadata.json in {directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        origin = str(metadata.get("origin", ""))
        customers: dict[str, Customer] = {}
        with (directory / "customers.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                customers[row["customer_id"]] = Customer(
                    customer_id=row["customer_id"],
                    segment=row["segment"],
                    region=row["region"],
                    pre_period_spend=float(row["pre_period_spend"]),
                )
        exposures: list[Exposure] = []
        with (directory / "exposures.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                exposures.append(
                    Exposure(
                        customer_id=row["customer_id"],
                        offer_id=row["offer_id"],
                        arm=row["arm"],
                        redeemed=int(row["redeemed"]),
                        post_spend=float(row["post_spend"]),
                    )
                )
        return cls(
            origin=origin,
            metadata=metadata,
            customers=customers,
            exposures=tuple(exposures),
        )

    @property
    def offer_ids(self) -> list[str]:
        return sorted({row.offer_id for row in self.exposures})

    def summary(self) -> dict[str, object]:
        segments: dict[str, int] = {}
        regions: dict[str, int] = {}
        for customer in self.customers.values():
            segments[customer.segment] = segments.get(customer.segment, 0) + 1
            regions[customer.region] = regions.get(customer.region, 0) + 1
        arm_counts: dict[str, dict[str, int]] = {}
        for row in self.exposures:
            arm_counts.setdefault(row.offer_id, {})
            arm_counts[row.offer_id][row.arm] = arm_counts[row.offer_id].get(row.arm, 0) + 1
        return {
            "origin": self.origin,
            "warning": (
                "Synthetic data. These rows are not real customers and the outcomes are not measured incrementality."
            ),
            "n_customers": len(self.customers),
            "n_exposures": len(self.exposures),
            "offers": self.offer_ids,
            "arm_counts": arm_counts,
            "segments": segments,
            "regions": regions,
            "metadata": self.metadata,
        }

    def outcome_arrays(
        self,
        offer_id: str,
        outcome: str,
        *,
        control_arm: str = "control",
        treatment_arm: str = "treatment",
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        if outcome not in {"redeemed", "post_spend"}:
            raise ValueError("outcome must be redeemed or post_spend")
        control_y: list[float] = []
        treatment_y: list[float] = []
        control_x: list[float] = []
        treatment_x: list[float] = []
        seen = False
        for row in self.exposures:
            if row.offer_id != offer_id:
                continue
            seen = True
            if row.arm not in {control_arm, treatment_arm}:
                continue
            customer = self.customers.get(row.customer_id)
            if customer is None:
                raise ValueError(f"exposure for unknown customer {row.customer_id}")
            value = float(row.redeemed if outcome == "redeemed" else row.post_spend)
            if row.arm == control_arm:
                control_y.append(value)
                control_x.append(customer.pre_period_spend)
            else:
                treatment_y.append(value)
                treatment_x.append(customer.pre_period_spend)
        if not seen:
            known = ", ".join(self.offer_ids) or "(none)"
            raise ValueError(f"unknown offer_id {offer_id!r}. Known offers: {known}")
        if len(control_y) < 2 or len(treatment_y) < 2:
            raise ValueError(
                f"offer {offer_id} does not have at least 2 rows in each of {control_arm!r} and {treatment_arm!r}"
            )
        return (
            np.asarray(control_y, dtype=np.float64),
            np.asarray(treatment_y, dtype=np.float64),
            np.asarray(control_x, dtype=np.float64),
            np.asarray(treatment_x, dtype=np.float64),
        )
