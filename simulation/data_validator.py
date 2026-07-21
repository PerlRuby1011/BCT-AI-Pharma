"""Validation checks for synthetic transaction data.

Confirms schema integrity and that the injected anomaly rate/distribution
matches expectations before the data is fed to the AI modules or blockchain
simulation layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "transaction_id",
    "timestamp",
    "product_id",
    "drug_class",
    "from_org",
    "from_org_type",
    "to_org",
    "to_org_type",
    "transaction_type",
    "temperature_c",
    "quantity",
    "location",
    "is_anomaly",
    "anomaly_type",
]


@dataclass
class ValidationReport:
    """Result of a data validation pass.

    Attributes:
        is_valid: Whether all checks passed.
        errors: List of hard-failure messages (cause ``is_valid=False``).
        warnings: List of soft-failure messages that do not fail validation.
        stats: Summary statistics computed during validation.
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def validate_schema(df: pd.DataFrame) -> List[str]:
    """Check that all required columns are present.

    Args:
        df: Transactions DataFrame.

    Returns:
        List of error messages for any missing columns (empty if valid).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return [f"Missing required column: {col}" for col in missing]


def validate_uniqueness(df: pd.DataFrame) -> List[str]:
    """Check that transaction IDs are unique and non-null.

    Args:
        df: Transactions DataFrame.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    if df["transaction_id"].isnull().any():
        errors.append("Found null transaction_id values")
    duplicate_count = int(df["transaction_id"].duplicated().sum())
    if duplicate_count > 0:
        errors.append(f"Found {duplicate_count} duplicate transaction_id values")
    return errors


def validate_anomaly_rate(
    df: pd.DataFrame, expected_rate: float, tolerance: float = 0.01
) -> List[str]:
    """Check that the observed anomaly rate is within tolerance of expectation.

    Args:
        df: Transactions DataFrame with an ``is_anomaly`` column.
        expected_rate: Expected fraction of anomalous transactions (e.g. 0.05).
        tolerance: Absolute tolerance allowed around ``expected_rate``.

    Returns:
        List of error messages (empty if valid).
    """
    observed_rate = float(df["is_anomaly"].mean())
    if abs(observed_rate - expected_rate) > tolerance:
        return [
            f"Observed anomaly rate {observed_rate:.4f} deviates from expected "
            f"{expected_rate:.4f} by more than tolerance {tolerance:.4f}"
        ]
    return []


def validate_temperature_ranges(df: pd.DataFrame) -> List[str]:
    """Sanity-check that temperature readings fall within a physically plausible range.

    Args:
        df: Transactions DataFrame with a ``temperature_c`` column.

    Returns:
        List of warning-level messages (empty if valid).
    """
    warnings = []
    out_of_range = df[(df["temperature_c"] < -40) | (df["temperature_c"] > 60)]
    if len(out_of_range) > 0:
        warnings.append(
            f"{len(out_of_range)} readings fall outside plausible [-40, 60] C range"
        )
    return warnings


def validate_transactions(
    df: pd.DataFrame,
    expected_anomaly_rate: float = 0.05,
    anomaly_rate_tolerance: float = 0.01,
) -> ValidationReport:
    """Run the full validation suite against a transactions DataFrame.

    Args:
        df: Transactions DataFrame to validate.
        expected_anomaly_rate: Expected overall anomaly fraction.
        anomaly_rate_tolerance: Absolute tolerance for the anomaly rate check.

    Returns:
        A :class:`ValidationReport` summarizing the outcome.
    """
    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(validate_schema(df))
    if errors:
        return ValidationReport(is_valid=False, errors=errors, warnings=warnings)

    errors.extend(validate_uniqueness(df))
    errors.extend(
        validate_anomaly_rate(df, expected_anomaly_rate, anomaly_rate_tolerance)
    )
    warnings.extend(validate_temperature_ranges(df))

    stats = {
        "n_transactions": len(df),
        "n_anomalies": int(df["is_anomaly"].sum()),
        "anomaly_rate": float(df["is_anomaly"].mean()),
        "anomaly_type_counts": df.loc[df["is_anomaly"], "anomaly_type"]
        .value_counts()
        .to_dict(),
        "drug_class_distribution": df["drug_class"].value_counts(normalize=True).to_dict(),
    }

    return ValidationReport(
        is_valid=len(errors) == 0, errors=errors, warnings=warnings, stats=stats
    )
