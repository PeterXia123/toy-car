from __future__ import annotations

import numpy as np
import pandas as pd

from eda.models import Finding, VariableInfo


_REQUIRED_COLUMNS = {
    "obs_month": (["Data"], "Observation month is required for all time-series checks."),
    "acct_id": (["Data"], "Account ID is required for account-level tracking and proportion calculation."),
    "dpd": (["PD", "DF"], "Days past due is required for default definition and PD estimation."),
    "ind_dft": (["PD", "DF"], "Default indicator is required for default rate and PD checks."),
    "ind_closed": (["ERL", "LGD"], "Closure indicator is required for terminal event and LGD checks."),
    "ind_CO": (["LGD"], "Charge-off indicator is required for LGD calculation."),
    "balance": (["LGD", "EAD"], "Balance is required for LGD and EAD checks."),
    "score_orig": (["Score_Alignment", "PD"], "Origination score is required for score-default alignment checks."),
    "score_bhv": (["Score_Alignment", "SICR"], "Behavior score is required for SICR and score alignment checks."),
    "perf_lvl1": (["Data"], "Performance level is required for default rate denominator and account classification."),
    "dt_opened": (["Data", "DF"], "Open date is required for mob calculation and vintage analysis."),
    "mob": (["DF", "ERL"], "Months on book is required for DF cohort and term structure estimation."),
    "new_to_dft": (["PD"], "New-to-default indicator is required for PD transition rate calculation."),
    "recovery": (["LGD"], "Recovery amount is required for LGD calculation."),
    "next_dft_bal": (["LGD"], "Balance at default is required for LGD numerator calculation."),
    "mths_to_dft": (["PD"], "Months to default is required for PD term structure estimation."),
    "ind_restructure": (["Score_Alignment"], "Restructure indicator is required for score alignment exclusion logic."),
}


def run(
    df: pd.DataFrame,
    checks_cfg: dict,
    variables_cfg: dict[str, VariableInfo],
    product: str,
) -> list[Finding]:
    findings: list[Finding] = []
    dq = checks_cfg.get("data_quality", {})

    findings += _check_required_columns(df, product)

    if dq.get("missing_values", {}).get("enabled", False):
        findings += check_missing_values(df, dq["missing_values"], variables_cfg, product)

    if dq.get("negative_values", {}).get("enabled", False):
        findings += check_negative_values(df, dq["negative_values"], variables_cfg, product)

    if dq.get("record_count_drop", {}).get("enabled", False):
        findings += check_record_count_drops(df, dq["record_count_drop"], product)

    if dq.get("extreme_values", {}).get("enabled", False):
        findings += check_extreme_values(df, dq["extreme_values"], variables_cfg, product)

    if dq.get("dtype_check", {}).get("enabled", False):
        findings += check_dtypes(df, dq["dtype_check"], variables_cfg, product)

    return findings


def _check_required_columns(df: pd.DataFrame, product: str) -> list[Finding]:
    findings = []
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if not missing:
        return findings
    for col in missing:
        downstream, reason = _REQUIRED_COLUMNS[col]
        findings.append(Finding(
            product=product,
            parameter=downstream,
            impact="High",
            question=f"Required column `{col}` is missing from data. {reason} Related checks will be skipped.",
            check_id="DQ0",
            variable=col,
            stats={"missing_column": col},
        ))
    return findings


def check_missing_values(
    df: pd.DataFrame,
    cfg: dict,
    variables_cfg: dict[str, VariableInfo],
    product: str,
) -> list[Finding]:
    threshold = cfg.get("threshold", 0.05)
    impact = cfg.get("impact", "Medium")
    parameter = cfg.get("parameter", "Data")
    findings: list[Finding] = []

    missing_matrix = {}
    has_month = "obs_month" in df.columns

    # Pre-compute monthly missing counts once for all variables (much faster than per-var lambda)
    monthly_total = df.groupby("obs_month").size() if has_month else None

    for var_name, var_info in variables_cfg.items():
        if var_name not in df.columns:
            continue

        overall_rate = float(df[var_name].isna().mean())
        missing_matrix[var_name] = overall_rate

        if overall_rate > threshold:
            per_month = {}
            if has_month and monthly_total is not None:
                monthly_na = df[var_name].isna().groupby(df["obs_month"]).sum()
                monthly_rate = (monthly_na / monthly_total).fillna(0)
                per_month = {str(k): round(float(v), 4) for k, v in monthly_rate.items()}

            findings.append(Finding(
                product=product,
                parameter=parameter,
                impact=impact,
                question=(
                    f"Variable `{var_name}` has {overall_rate:.1%} missing values overall, "
                    f"exceeding the {threshold:.0%} threshold. "
                    f"This may impact {', '.join(var_info.downstream)}."
                ),
                check_id="DQ1",
                variable=var_name,
                stats={"overall_missing_rate": round(overall_rate, 4), "per_month": per_month},
            ))

    if has_month and missing_matrix and monthly_total is not None:
        all_monthly = {}
        for var_name in missing_matrix:
            if var_name in df.columns:
                monthly_na = df[var_name].isna().groupby(df["obs_month"]).sum()
                monthly_rate = (monthly_na / monthly_total).fillna(0)
                all_monthly[var_name] = {str(k): round(float(v), 4) for k, v in monthly_rate.items()}

        findings.append(Finding(
            product=product,
            parameter=parameter,
            impact="Low",
            question="Missing value heatmap data generated for all variables across observation months.",
            check_id="DQ1_HEATMAP",
            variable="ALL",
            reference_only=True,
            stats={"missing_heatmap": all_monthly},
        ))

    return findings


def check_negative_values(
    df: pd.DataFrame,
    cfg: dict,
    variables_cfg: dict[str, VariableInfo],
    product: str,
) -> list[Finding]:
    impact = cfg.get("impact", "Medium")
    parameter = cfg.get("parameter", "Data")
    findings: list[Finding] = []

    for var_name, var_info in variables_cfg.items():
        if var_name not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[var_name]):
            continue

        min_val = var_info.constraints.get("min")
        if min_val is None or min_val < 0:
            continue

        neg_mask = df[var_name] < min_val
        neg_count = neg_mask.sum()
        if neg_count == 0:
            continue

        neg_rate = neg_count / len(df)
        ex_cols = [c for c in ["acct_id", "obs_month", var_name] if c in df.columns]
        examples = df.loc[neg_mask, ex_cols].head(20)

        findings.append(Finding(
            product=product,
            parameter=parameter,
            impact=impact,
            question=(
                f"Variable `{var_name}` has {neg_count:,} records ({neg_rate:.2%}) "
                f"with values below {min_val}. This variable has a minimum constraint of {min_val}. "
                f"This may impact {', '.join(var_info.downstream)}."
            ),
            check_id="DQ2",
            variable=var_name,
            examples=examples,
            stats={"negative_count": int(neg_count), "negative_rate": round(neg_rate, 4)},
        ))

    return findings


def check_record_count_drops(
    df: pd.DataFrame,
    cfg: dict,
    product: str,
) -> list[Finding]:
    if "obs_month" not in df.columns:
        return []

    threshold = cfg.get("threshold", 0.10)
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", "Data")
    findings: list[Finding] = []

    monthly_counts = df.groupby("obs_month").size().sort_index()

    for i in range(1, len(monthly_counts)):
        prev_count = monthly_counts.iloc[i - 1]
        curr_count = monthly_counts.iloc[i]
        if prev_count == 0:
            continue
        change = (curr_count - prev_count) / prev_count

        if change < -threshold:
            month = monthly_counts.index[i]
            prev_month = monthly_counts.index[i - 1]
            findings.append(Finding(
                product=product,
                parameter=parameter,
                impact=impact,
                question=(
                    f"Record count dropped by {abs(change):.1%} from {prev_month} "
                    f"({prev_count:,} records) to {month} ({curr_count:,} records), "
                    f"exceeding the {threshold:.0%} threshold. "
                    f"This may indicate data truncation or incomplete extraction."
                ),
                check_id="DQ3",
                variable="obs_month",
                stats={
                    "month": str(month),
                    "prev_count": int(prev_count),
                    "curr_count": int(curr_count),
                    "change_pct": round(change, 4),
                },
            ))

    if len(monthly_counts) > 0:
        findings.append(Finding(
            product=product,
            parameter=parameter,
            impact="Low",
            question="Account count trend data generated.",
            check_id="DQ3_TREND",
            variable="obs_month",
            reference_only=True,
            stats={"account_counts": {str(k): int(v) for k, v in monthly_counts.items()}},
        ))

    return findings


def check_extreme_values(
    df: pd.DataFrame,
    cfg: dict,
    variables_cfg: dict[str, VariableInfo],
    product: str,
) -> list[Finding]:
    method = cfg.get("method", "iqr")
    iqr_mult = cfg.get("iqr_multiplier", 3.0)
    zscore_thresh = cfg.get("zscore_threshold", 3.0)
    impact = cfg.get("impact", "Medium")
    parameter = cfg.get("parameter", "Data")
    findings: list[Finding] = []

    skip_types = {"Status", "Derived", "ID"}
    for var_name, var_info in variables_cfg.items():
        if var_name not in df.columns:
            continue
        if var_info.var_type in skip_types:
            continue
        if not pd.api.types.is_numeric_dtype(df[var_name]):
            continue
        if var_info.valid_values is not None:
            continue

        col_data = df[var_name]
        series = col_data.dropna()
        if len(series) < 10:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - iqr_mult * iqr
            upper = q3 + iqr_mult * iqr
            outlier_mask = (df[var_name] < lower) | (df[var_name] > upper)
        else:
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            outlier_mask = ((df[var_name] - mean) / std).abs() > zscore_thresh

        outlier_count = outlier_mask.sum()
        if outlier_count == 0:
            continue

        outlier_rate = outlier_count / len(df)
        if outlier_rate < 0.001:
            continue

        findings.append(Finding(
            product=product,
            parameter=parameter,
            impact=impact,
            question=(
                f"Variable `{var_name}` has {outlier_count:,} extreme values ({outlier_rate:.2%}) "
                f"detected using {method.upper()} method. "
                f"Range: [{series.min():.2f}, {series.max():.2f}], "
                f"IQR bounds: [{lower:.2f}, {upper:.2f}]."
                if method == "iqr"
                else f"Variable `{var_name}` has {outlier_count:,} extreme values ({outlier_rate:.2%}) "
                f"detected using Z-score method (threshold={zscore_thresh})."
            ),
            check_id="DQ4",
            variable=var_name,
            examples=df.loc[outlier_mask, [c for c in ["acct_id", "obs_month", var_name] if c in df.columns]].head(20),
            stats={"outlier_count": int(outlier_count), "outlier_rate": round(outlier_rate, 4)},
        ))

    return findings


_EXPECTED_DTYPES = {
    "Date": "datetime",
    "Term": "numeric",
    "Value": "numeric",
    "Rate": "numeric",
    "Score": "numeric",
    "Status": "numeric",
    "Derived": "numeric",
    "ID": None,
}


def check_dtypes(
    df: pd.DataFrame,
    cfg: dict,
    variables_cfg: dict[str, VariableInfo],
    product: str,
) -> list[Finding]:
    impact = cfg.get("impact", "Low")
    parameter = cfg.get("parameter", "Data")
    findings: list[Finding] = []

    for var_name, var_info in variables_cfg.items():
        if var_name not in df.columns:
            continue

        expected = _EXPECTED_DTYPES.get(var_info.var_type)
        if expected is None:
            continue

        actual = df[var_name].dtype
        mismatch = False

        if expected == "datetime" and not pd.api.types.is_datetime64_any_dtype(actual):
            mismatch = True
        elif expected == "numeric" and not pd.api.types.is_numeric_dtype(actual):
            mismatch = True

        if mismatch:
            findings.append(Finding(
                product=product,
                parameter=parameter,
                impact=impact,
                question=(
                    f"Variable `{var_name}` expected type '{expected}' "
                    f"(variable type: {var_info.var_type}) but actual dtype is '{actual}'. "
                    f"This may cause calculation errors downstream."
                ),
                check_id="DQ5",
                variable=var_name,
                stats={"expected": expected, "actual": str(actual)},
            ))

    return findings
