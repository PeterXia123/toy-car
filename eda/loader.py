from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eda.models import ProjectConfig, VariableInfo


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_project_config(path: str) -> ProjectConfig:
    return ProjectConfig.from_dict(load_yaml(path))


def load_variables_config(path: str) -> dict[str, VariableInfo]:
    raw = load_yaml(path)
    variables = {}
    for name, info in raw.get("variables", {}).items():
        variables[name] = VariableInfo(
            name=name,
            var_type=info.get("type", ""),
            downstream=info.get("downstream", []),
            description=info.get("description", ""),
            constraints=info.get("constraints", {}),
            valid_values=info.get("valid_values"),
            expected_dtype=info.get("dtype", ""),
            format_hint=info.get("format", ""),
        )
    return variables


def load_checks_config(path: str) -> dict:
    return load_yaml(path)


def load_data(
    project: ProjectConfig,
    variables_config: dict[str, VariableInfo] | None = None,
) -> pd.DataFrame:
    file_path = os.path.join(project.data_path, project.data_file)
    if not os.path.exists(file_path):
        alt = Path(project.data_file)
        if alt.exists():
            file_path = str(alt)
        else:
            raise FileNotFoundError(f"Data file not found: {file_path}")

    if project.data_format == "parquet":
        df = pd.read_parquet(file_path)
    elif project.data_format == "csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported format: {project.data_format}")

    df = apply_column_mapping(df, project.column_mapping)
    if variables_config:
        validate_input_dtypes(df, project.column_mapping, variables_config)
    df = _normalize_obs_month(df)
    df = _normalize_indicators(df)
    df = apply_filters(df, project.filters)
    df = _optimize_dtypes(df)
    return df


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    reverse_map = {}
    for standard_name, actual_name in mapping.items():
        if actual_name in df.columns and actual_name != standard_name:
            reverse_map[actual_name] = standard_name
    if reverse_map:
        df = df.rename(columns=reverse_map)
    return df


_DTYPE_OK = {
    "integer": lambda dt: pd.api.types.is_integer_dtype(dt),
    "float": lambda dt: pd.api.types.is_numeric_dtype(dt),
    "datetime": lambda dt: pd.api.types.is_datetime64_any_dtype(dt),
    "string": lambda dt: pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt),
    "string_or_integer": lambda dt: True,
}

_DTYPE_COMPAT = {
    "integer": lambda dt: pd.api.types.is_float_dtype(dt),
    "datetime": lambda dt: pd.api.types.is_object_dtype(dt),
}

_FIX_HINTS = {
    "integer": "df['{col}'] = df['{col}'].fillna(0).astype(int)",
    "float": "df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')",
    "datetime": "df['{col}'] = pd.to_datetime(df['{col}'])",
    "string": "df['{col}'] = df['{col}'].astype(str)",
}


def validate_input_dtypes(
    df: pd.DataFrame,
    column_mapping: dict[str, str],
    variables_config: dict[str, VariableInfo],
) -> None:
    errors = []
    warns = []
    n_ok = 0

    for var_name, var_info in variables_config.items():
        if var_name not in df.columns or not var_info.expected_dtype:
            continue

        display = column_mapping.get(var_name, var_name)
        actual = df[var_name].dtype
        expected = var_info.expected_dtype

        if var_info.var_type == "Date":
            if pd.api.types.is_datetime64_any_dtype(actual):
                n_ok += 1
                continue
            if pd.api.types.is_float_dtype(actual):
                warns.append((display, expected, str(actual), var_info.format_hint,
                              "NaN present -> float promotion, will auto-convert"))
                continue
            if pd.api.types.is_integer_dtype(actual):
                n_ok += 1
                continue
            if pd.api.types.is_object_dtype(actual):
                warns.append((display, expected, str(actual), var_info.format_hint,
                              "String dates detected, will attempt auto-parse"))
                continue
            errors.append((display, expected, str(actual), var_info.format_hint))
            continue

        ok_fn = _DTYPE_OK.get(expected)
        if ok_fn and ok_fn(actual):
            n_ok += 1
            continue

        compat_fn = _DTYPE_COMPAT.get(expected)
        if compat_fn and compat_fn(actual):
            reason = "NaN present -> float promotion, will auto-convert"
            if expected == "datetime" and pd.api.types.is_object_dtype(actual):
                reason = "String dates detected, will attempt auto-parse"
            warns.append((display, expected, str(actual), var_info.format_hint, reason))
            continue

        errors.append((display, expected, str(actual), var_info.format_hint))

    total = n_ok + len(warns) + len(errors)
    if not warns and not errors:
        print(f"  Data format validation: {total} columns checked, all OK")
        return

    print(f"\n{'='*60}")
    print(f"DATA FORMAT VALIDATION")
    print(f"{'='*60}")
    print(f"Checked {total} columns: {n_ok} OK, {len(warns)} warning(s), {len(errors)} error(s)\n")

    for display, expected, actual, fmt, reason in warns:
        label = f"{expected} ({fmt})" if fmt else expected
        print(f"  [WARN] {display}: expected {label}, got {actual}")
        print(f"         {reason}\n")

    for display, expected, actual, fmt in errors:
        label = f"{expected} ({fmt})" if fmt else expected
        fix = _FIX_HINTS.get(expected, "").format(col=display)
        print(f"  [FAIL] {display}: expected {label}, got {actual}")
        if fix:
            print(f"         Fix: {fix}")
        print()

    print(f"{'='*60}\n")

    if errors:
        cols = ", ".join(e[0] for e in errors)
        raise ValueError(
            f"Data validation failed: {len(errors)} column(s) have incompatible dtypes "
            f"({cols}). Please fix and re-run."
        )


def _normalize_obs_month(df: pd.DataFrame) -> pd.DataFrame:
    """Convert rpt_mth from int/float YYYYMM to datetime if needed."""
    if "rpt_mth" not in df.columns:
        return df
    if pd.api.types.is_integer_dtype(df["rpt_mth"]):
        df["rpt_mth"] = pd.to_datetime(df["rpt_mth"].astype(str), format="%Y%m")
    elif pd.api.types.is_float_dtype(df["rpt_mth"]):
        df["rpt_mth"] = pd.to_datetime(
            df["rpt_mth"].fillna(0).round().astype("int64").astype(str),
            format="%Y%m",
            errors="coerce",
        )
    elif pd.api.types.is_object_dtype(df["rpt_mth"]):
        df["rpt_mth"] = pd.to_datetime(df["rpt_mth"])
    return df


def _normalize_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in binary indicator columns with 0 and downcast to int8."""
    indicator_cols = [
        "fl_close", "fl_wo", "fl_evt", "fl_excl",
        "new_evt", "new_wo", "lag_fl_evt", "lag_fl_wo",
        "fl_restr",
    ]
    for col in indicator_cols:
        if col in df.columns and df[col].dtype == np.float64:
            df[col] = df[col].fillna(0).astype(np.int8)

    if "rcv_amt" in df.columns:
        df["rcv_amt"] = df["rcv_amt"].fillna(0)

    return df


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert low-cardinality objects to category."""
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype(np.int8)
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype(np.int16)
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype(np.int32)

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    exclude = filters.get("exclude_months", [])
    if exclude and "rpt_mth" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["rpt_mth"]):
            exclude = pd.to_datetime(exclude)
        df = df[~df["rpt_mth"].isin(exclude)]

    lvl_filter = filters.get("perf_lvl1_filter")
    if lvl_filter is not None and "grp1" in df.columns:
        df = df[df["grp1"] == lvl_filter]

    return df


def has_column(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns
