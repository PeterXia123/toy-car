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
        )
    return variables


def load_checks_config(path: str) -> dict:
    return load_yaml(path)


def load_data(project: ProjectConfig) -> pd.DataFrame:
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
    df = _normalize_obs_month(df)
    df = _normalize_indicators(df)
    df = apply_filters(df, project.filters)
    if "acct_id" in df.columns:
        df["acct_id"] = df["acct_id"].astype("category")
    return df


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    reverse_map = {}
    for standard_name, actual_name in mapping.items():
        if actual_name in df.columns and actual_name != standard_name:
            reverse_map[actual_name] = standard_name
    if reverse_map:
        df = df.rename(columns=reverse_map)
    return df


def _normalize_obs_month(df: pd.DataFrame) -> pd.DataFrame:
    """Convert obs_month from int YYYYMM to datetime if needed."""
    if "obs_month" not in df.columns:
        return df
    if pd.api.types.is_integer_dtype(df["obs_month"]):
        df["obs_month"] = pd.to_datetime(df["obs_month"].astype(str), format="%Y%m")
    elif pd.api.types.is_object_dtype(df["obs_month"]):
        df["obs_month"] = pd.to_datetime(df["obs_month"])
    return df


def _normalize_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in binary indicator columns with 0 so comparisons work."""
    indicator_cols = [
        "ind_closed", "ind_CO", "ind_dft", "ind_excl",
        "new_to_dft", "new_to_CO", "lag_ind_dft", "lag_ind_CO",
        "ind_restructure",
    ]
    for col in indicator_cols:
        if col in df.columns and df[col].dtype == np.float64:
            df[col] = df[col].fillna(0).astype(np.int64)
    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    exclude = filters.get("exclude_months", [])
    if exclude and "obs_month" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["obs_month"]):
            exclude = pd.to_datetime(exclude)
        df = df[~df["obs_month"].isin(exclude)]

    lvl_filter = filters.get("perf_lvl1_filter")
    if lvl_filter is not None and "perf_lvl1" in df.columns:
        df = df[df["perf_lvl1"] == lvl_filter]

    return df


def has_column(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns
