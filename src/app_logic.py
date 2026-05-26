"""Shared app behavior for chat routing and dataset profiling."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd


MAX_DATASET_BYTES = 30 * 1024 * 1024
SUPPORTED_DATASET_EXTENSIONS = ("csv", "xlsx", "xlsm")
REPORT_NAMES = ("eda_report.md", "ml_report.md")
MAX_PROFILE_COLUMNS = 40
MAX_SUMMARY_COLUMNS = 12
MAX_CATEGORICAL_COLUMNS = 8


@dataclass(frozen=True)
class DatasetProfile:
    """Compact dataset profile that is safe to place in an LLM prompt."""

    file_name: str
    size_mb: float
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    duplicate_rows: int
    numeric_summary: dict[str, dict[str, Any]]
    categorical_summary: dict[str, dict[str, Any]]
    target_column: str
    target_summary: dict[str, Any]
    sample_rows: list[dict[str, Any]]


def validate_dataset_upload(file_name: str, size_bytes: int) -> None:
    """Validate file size and extension before reading user data."""

    extension = Path(file_name).suffix.lower().lstrip(".")
    supported = ", ".join(f".{ext}" for ext in SUPPORTED_DATASET_EXTENSIONS)

    if extension not in SUPPORTED_DATASET_EXTENSIONS:
        raise ValueError(f"Unsupported file type. Please upload one of: {supported}.")

    if size_bytes > MAX_DATASET_BYTES:
        max_mb = MAX_DATASET_BYTES // (1024 * 1024)
        raise ValueError(f"Dataset is too large. Maximum supported size is {max_mb} MB.")


def load_dataset_from_bytes(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Load a supported uploaded dataset into a DataFrame."""

    validate_dataset_upload(file_name, len(file_bytes))
    extension = Path(file_name).suffix.lower().lstrip(".")
    buffer = BytesIO(file_bytes)

    if extension == "csv":
        return pd.read_csv(buffer)

    return pd.read_excel(buffer)


def build_dataset_profile(
    df: pd.DataFrame,
    *,
    file_name: str,
    size_bytes: int,
    target_column: str,
) -> DatasetProfile:
    """Create a compact, analysis-ready profile from a DataFrame."""

    if target_column not in df.columns:
        raise ValueError("Please choose a valid target variable from the uploaded dataset.")

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(exclude="number")
    target = df[target_column]

    numeric_summary = _frame_to_nested_dict(
        numeric_df.describe().transpose().head(MAX_SUMMARY_COLUMNS)
        if not numeric_df.empty
        else pd.DataFrame()
    )
    categorical_summary = _categorical_summary(categorical_df)
    target_summary = _target_summary(target)

    return DatasetProfile(
        file_name=file_name,
        size_mb=round(size_bytes / (1024 * 1024), 2),
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        columns=[str(column) for column in df.columns[:MAX_PROFILE_COLUMNS]],
        dtypes={
            str(column): str(dtype)
            for column, dtype in df.dtypes.head(MAX_PROFILE_COLUMNS).items()
        },
        missing_counts={
            str(column): int(count)
            for column, count in df.isna()
            .sum()
            .sort_values(ascending=False)
            .head(MAX_SUMMARY_COLUMNS)
            .items()
        },
        duplicate_rows=int(df.duplicated().sum()),
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        target_column=str(target_column),
        target_summary=target_summary,
        sample_rows=_safe_records(df.head(2)),
    )


def dataset_profile_to_prompt(user_prompt: str, profile: DatasetProfile) -> str:
    """Combine the user's request with dataset context for the analysis crew."""

    return f"""
{dataset_profile_to_context(user_prompt, profile)}

This is a supervised learning workflow because the user selected a target variable.
The agents must infer the correct supervised task type from the target summary and
dataset context instead of relying on Python-side rules. Produce practical, real-world
EDA and ML recommendations grounded in this dataset profile. Highlight risks like
leakage, imbalance, missing data, outliers, and deployment concerns. If the available
profile is not enough for a specific conclusion, say what additional analysis should be run.
""".strip()


def dataset_profile_to_context(user_prompt: str, profile: DatasetProfile) -> str:
    """Create neutral dataset context for chat/routing without forcing analysis."""

    return f"""
User request:
{user_prompt}

DATASET CONTEXT:
- File: {profile.file_name}
- Size: {profile.size_mb} MB
- Shape: {profile.row_count} rows x {profile.column_count} columns
- Target variable: {profile.target_column}
- Columns: {profile.columns}
- Dtypes: {profile.dtypes}
- Missing values, top columns: {profile.missing_counts}
- Duplicate rows: {profile.duplicate_rows}
- Numeric summary, first {MAX_SUMMARY_COLUMNS} numeric columns: {profile.numeric_summary}
- Categorical summary, first {MAX_CATEGORICAL_COLUMNS} categorical columns: {profile.categorical_summary}
- Target summary: {profile.target_summary}
- Sample rows, first 2: {profile.sample_rows}
""".strip()


def dataset_profile_to_route_context(user_prompt: str, profile: DatasetProfile) -> str:
    """Create a tiny context for deciding whether full analysis is needed."""

    return f"""
User request:
{user_prompt}

Dataset is loaded:
- File: {profile.file_name}
- Shape: {profile.row_count} rows x {profile.column_count} columns
- Target variable: {profile.target_column}
""".strip()


def report_paths(output_dir: Path) -> list[Path]:
    """Return standard report file paths."""

    return [output_dir / name for name in REPORT_NAMES]


def clear_generated_reports(output_dir: Path) -> None:
    """Remove stale report files so old analysis is not shown as current."""

    for path in report_paths(output_dir):
        if path.exists():
            path.unlink()


def _frame_to_nested_dict(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}

    return {
        str(index): {str(column): _safe_value(value) for column, value in row.items()}
        for index, row in frame.iterrows()
    }


def _categorical_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for column in df.columns[:MAX_CATEGORICAL_COLUMNS]:
        series = df[column]
        summary[str(column)] = {
            "unique_values": int(series.nunique(dropna=True)),
            "missing_values": int(series.isna().sum()),
            "top_values": {
                str(key): int(value)
                for key, value in series.value_counts(dropna=False).head(3).items()
            },
        }

    return summary


def _target_summary(series: pd.Series) -> dict[str, Any]:
    value_counts = series.value_counts(dropna=False).head(6)
    summary: dict[str, Any] = {
        "dtype": str(series.dtype),
        "missing_values": int(series.isna().sum()),
        "unique_values": int(series.nunique(dropna=True)),
        "top_values": {str(key): int(value) for key, value in value_counts.items()},
    }

    if pd.api.types.is_numeric_dtype(series):
        summary["numeric_stats"] = {
            key: _safe_value(value)
            for key, value in series.describe().to_dict().items()
        }

    return summary


def _safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(column): _safe_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value
