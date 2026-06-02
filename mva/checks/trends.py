from __future__ import annotations

import numpy as np
import pandas as pd

from mva.models import Finding, VariableInfo


def run(
    df: pd.DataFrame,
    checks_cfg: dict,
    variables_cfg: dict[str, VariableInfo],
    product: str,
) -> list[Finding]:
    findings: list[Finding] = []
    tr = checks_cfg.get("trends", {})

    ts_cfg = tr.get("time_series", {})
    if ts_cfg.get("enabled", False) and "obs_month" in df.columns:
        findings += _run_time_series(df, ts_cfg, product)

    return findings


def run_account_tracking(
    df: pd.DataFrame,
    checks_cfg: dict,
    product: str,
) -> list[Finding]:
    at = checks_cfg.get("account_tracking", {})
    findings: list[Finding] = []

    if not {"acct_id", "obs_month"}.issubset(df.columns):
        return findings

    last_records = df.groupby("acct_id", observed=True).last()
    global_max = df["obs_month"].max()

    if at.get("right_censoring", {}).get("enabled", False):
        findings += _check_right_censoring(last_records, global_max, at["right_censoring"], product)

    if at.get("right_censoring", {}).get("enabled", False):
        findings += _check_censored_trend(df, last_records, global_max, product)

    if at.get("single_record", {}).get("enabled", False):
        findings += _check_single_record(df, at["single_record"], product)

    if at.get("record_gaps", {}).get("enabled", False):
        findings += _check_record_gaps(df, at["record_gaps"], product)

    if at.get("incomplete_month", {}).get("enabled", False):
        findings += _check_incomplete_months(df, at["incomplete_month"], product)

    if at.get("disappear_profile", {}).get("enabled", False):
        findings += _check_disappear_profile(last_records, global_max, at["disappear_profile"], product)

    if at.get("left_censoring", {}).get("enabled", False):
        findings += _check_left_censoring(df, at["left_censoring"], product)

    if at.get("post_default_tracking", {}).get("enabled", False):
        findings += _check_post_default_tracking(df, at["post_default_tracking"], product)

    findings += _check_disappear_classification(last_records, global_max, at, product)

    return findings


# ================================================================
# Time Series Trend Analysis
# ================================================================

def _run_time_series(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []
    variables = cfg.get("variables", [])
    anomaly_thresh = cfg.get("anomaly_threshold", 2.5)
    impact = cfg.get("impact", "Medium")
    parameter = cfg.get("parameter", "Data")

    for var_name in variables:
        if var_name not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[var_name]):
            continue

        monthly = df.groupby("obs_month")[var_name].agg(["mean", "std", "count"]).sort_index()

        if len(monthly) < 4:
            continue

        mean_series = monthly["mean"]
        overall_mean = mean_series.mean()
        overall_std = mean_series.std()

        anomalies = {}
        if overall_std > 0:
            z_scores = (mean_series - overall_mean) / overall_std
            for month, z in z_scores.items():
                if abs(z) > anomaly_thresh:
                    anomalies[str(month)] = round(float(z), 2)

        if anomalies:
            findings.append(Finding(
                product=product,
                parameter=parameter,
                impact=impact,
                question=(
                    f"Variable `{var_name}` shows anomalous trend at months: "
                    f"{', '.join(anomalies.keys())}. "
                    f"Z-scores exceed {anomaly_thresh} threshold."
                ),
                check_id="TS1",
                variable=var_name,
                stats={
                    "anomalies": anomalies,
                    "monthly_mean": {str(k): round(float(v), 4) for k, v in mean_series.items()},
                },
            ))

    return findings


# ================================================================
# Account Tracking (AT1-AT8)
# ================================================================

def _build_censored_mask(last_records: pd.DataFrame, global_max) -> pd.Series:
    mask = last_records["obs_month"] < global_max
    for col in ["ind_closed", "ind_CO", "ind_dft"]:
        if col in last_records.columns:
            mask = mask & (last_records[col] != 1)
    return mask


def _check_right_censoring(
    last_records: pd.DataFrame, global_max, cfg: dict, product: str,
) -> list[Finding]:
    """AT1: Accounts that disappear without closure, chargeoff, or default."""
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", ["ERL", "DF", "LGD"])

    censored_mask = _build_censored_mask(last_records, global_max)
    censored_count = int(censored_mask.sum())
    total_accounts = len(last_records)
    censored_rate = censored_count / total_accounts if total_accounts > 0 else 0

    if censored_count == 0:
        return []

    return [Finding(
        product=product,
        parameter=parameter,
        impact=impact,
        question=(
            f"{censored_count:,} accounts ({censored_rate:.1%}) are right-censored — "
            f"they disappear before {global_max} without closure, chargeoff, or default. "
            f"These accounts cannot contribute complete event histories for ERL or DF estimation."
        ),
        check_id="AT1",
        variable="acct_id",
        stats={
            "censored_count": censored_count,
            "total_accounts": total_accounts,
            "censored_rate": round(censored_rate, 4),
        },
    )]


def _check_censored_trend(
    df: pd.DataFrame, last_records: pd.DataFrame, global_max, product: str,
) -> list[Finding]:
    """AT2: Monthly attrition rate — accounts present this month but gone next month without reason."""
    months = sorted(df["obs_month"].unique())
    if len(months) < 2:
        return []

    indicator_cols = [c for c in ["ind_closed", "ind_CO", "ind_dft"] if c in df.columns]
    trend = {}
    for i in range(len(months) - 1):
        m_curr, m_next = months[i], months[i + 1]
        accts_curr = set(df.loc[df["obs_month"] == m_curr, "acct_id"])
        accts_next = set(df.loc[df["obs_month"] == m_next, "acct_id"])
        disappeared = accts_curr - accts_next
        if not disappeared or not accts_curr:
            trend[str(m_curr)] = {"disappeared": 0, "active": len(accts_curr), "rate": 0.0}
            continue
        # Filter out accounts that disappeared for a known reason
        dis_records = df[(df["acct_id"].isin(disappeared)) & (df["obs_month"] == m_curr)]
        unexplained = dis_records.copy()
        for col in indicator_cols:
            unexplained = unexplained[unexplained[col] != 1]
        n_unexplained = len(unexplained)
        trend[str(m_curr)] = {
            "disappeared": n_unexplained,
            "active": len(accts_curr),
            "rate": round(n_unexplained / len(accts_curr), 4),
        }

    total_unexplained = sum(v["disappeared"] for v in trend.values())
    avg_rate = sum(v["rate"] for v in trend.values()) / len(trend) if trend else 0

    return [Finding(
        product=product,
        parameter="Data",
        impact="Medium",
        question=(
            f"Monthly unexplained attrition: {total_unexplained} account disappearances across "
            f"{len(trend)} months (avg rate: {avg_rate:.2%}/month). "
            f"Accounts present in month t but absent in month t+1 without closure/CO/default."
        ),
        check_id="AT2",
        variable="acct_id",
        reference_only=True,
        stats={"attrition_trend": trend},
    )]


def _check_single_record(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT3: Accounts with only 1 record in the entire dataset."""
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", ["ERL", "PD"])

    counts = df.groupby("acct_id", observed=True).size()
    single = int((counts == 1).sum())
    total = len(counts)
    rate = single / total if total > 0 else 0

    if single == 0:
        return []

    return [Finding(
        product=product,
        parameter=parameter,
        impact=impact,
        question=(
            f"{single:,} accounts ({rate:.1%}) have only 1 observation record. "
            f"Single-record accounts cannot contribute to transition matrices, "
            f"cure rate estimation, or ERL calculation."
        ),
        check_id="AT3",
        variable="acct_id",
        stats={"single_record_count": single, "rate": round(rate, 4)},
    )]


def _check_record_gaps(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT4: Accounts with gaps in obs_month sequence."""
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", ["Data", "ERL"])
    max_gap = cfg.get("max_allowed_gap_months", 1)

    sorted_df = df[["acct_id", "obs_month"]]

    if pd.api.types.is_datetime64_any_dtype(sorted_df["obs_month"]):
        month_diff = sorted_df.groupby("acct_id", observed=True)["obs_month"].diff().dt.days / 30.44
    else:
        obs_num = pd.to_datetime(sorted_df["obs_month"]).astype(np.int64)
        month_diff = sorted_df.groupby("acct_id", observed=True)[obs_num.name].diff() / (30.44 * 24 * 3600 * 1e9)

    gap_mask = month_diff > (max_gap + 0.5)
    gap_count = int(gap_mask.sum())

    if gap_count == 0:
        return []

    gap_accounts = int(sorted_df.loc[gap_mask, "acct_id"].nunique())
    total_accounts = int(sorted_df["acct_id"].nunique())

    return [Finding(
        product=product,
        parameter=parameter,
        impact=impact,
        question=(
            f"{gap_accounts:,} accounts ({gap_accounts/total_accounts:.1%}) have gaps "
            f"in their obs_month sequence exceeding {max_gap} month(s). "
            f"Record gaps break cohort continuity and may cause incorrect "
            f"transition matrix estimation."
        ),
        check_id="AT4",
        variable="acct_id",
        stats={"gap_accounts": gap_accounts, "total_gaps": gap_count,
               "affected_accounts": gap_accounts, "total_accounts": total_accounts,
               "account_rate": round(gap_accounts / total_accounts, 4) if total_accounts > 0 else 0},
    )]


def _check_incomplete_months(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT5: Observation months with account count below 50% of the global median."""
    findings: list[Finding] = []
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", "Data")
    min_ratio = cfg.get("min_account_ratio", 0.5)

    monthly_counts = df.groupby("obs_month")["acct_id"].nunique().sort_index()

    if len(monthly_counts) < 3:
        return []

    median_count = monthly_counts.median()
    if median_count == 0:
        return []

    for month, curr in monthly_counts.items():
        ratio = curr / median_count
        if ratio < min_ratio:
            findings.append(Finding(
                product=product,
                parameter=parameter,
                impact=impact,
                question=(
                    f"obs_month {month} has {curr:,} unique accounts, "
                    f"below {min_ratio:.0%} of the median ({median_count:.0f}). "
                    f"This suggests incomplete data extraction for that month."
                ),
                check_id="AT5",
                variable="obs_month",
                stats={
                    "month": str(month),
                    "current_count": int(curr),
                    "median_count": int(median_count),
                    "ratio": round(ratio, 4),
                },
            ))

    return findings


def _check_disappear_profile(
    last_records: pd.DataFrame, global_max, cfg: dict, product: str,
) -> list[Finding]:
    """AT6: Compare profile of censored vs non-censored accounts."""
    impact = cfg.get("impact", "Medium")
    parameter = cfg.get("parameter", "PD")

    censored_mask = _build_censored_mask(last_records, global_max)
    censored = last_records[censored_mask]
    non_censored = last_records[~censored_mask]

    if len(censored) < 10 or len(non_censored) < 10:
        return []

    profile_vars = ["dpd", "balance", "score_orig", "score_bhv"]
    comparison = {}
    for var in profile_vars:
        if var in last_records.columns and pd.api.types.is_numeric_dtype(last_records[var]):
            c_mean = censored[var].mean()
            nc_mean = non_censored[var].mean()
            if pd.notna(c_mean) and pd.notna(nc_mean):
                comparison[var] = {
                    "censored_mean": round(float(c_mean), 4),
                    "non_censored_mean": round(float(nc_mean), 4),
                    "diff_pct": round(float((c_mean - nc_mean) / nc_mean * 100), 2) if nc_mean != 0 else None,
                }

    if not comparison:
        return []

    diffs = [f"{v}: censored={d['censored_mean']:.2f} vs normal={d['non_censored_mean']:.2f}"
             for v, d in comparison.items() if d.get("diff_pct") is not None]

    return [Finding(
        product=product,
        parameter=parameter,
        impact=impact,
        question=(
            f"Censored accounts show different characteristics from non-censored: "
            f"{'; '.join(diffs[:3])}. "
            f"If censored accounts are systematically riskier, "
            f"their exclusion will bias PD estimates downward."
        ),
        check_id="AT6",
        variable="acct_id",
        stats={"profile_comparison": comparison},
    )]


def _check_left_censoring(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT7: Accounts whose dt_opened is before the earliest obs_month in the dataset (left censoring)."""
    impact = cfg.get("impact", "Medium")
    parameter = cfg.get("parameter", "Data")

    if "dt_opened" not in df.columns or "obs_month" not in df.columns:
        return []
    if not pd.api.types.is_datetime64_any_dtype(df["dt_opened"]):
        return []

    window_start = df["obs_month"].min()
    valid = df["dt_opened"].notna()
    if not valid.any():
        return []

    left_censored = df.loc[valid, "dt_opened"] < window_start
    lc_accts = df.loc[valid][left_censored].groupby("acct_id", observed=True).ngroups if "acct_id" in df.columns else int(left_censored.sum())
    total_accts = df["acct_id"].nunique() if "acct_id" in df.columns else len(df)
    rate = lc_accts / total_accts if total_accts > 0 else 0

    if lc_accts == 0:
        return []

    return [Finding(
        product=product,
        parameter=parameter,
        impact=impact,
        question=(
            f"{lc_accts:,} accounts ({rate:.1%}) have dt_opened before data window start ({window_start}). "
            f"These left-censored accounts lack early-life history, which may bias survival analysis and ERL estimates."
        ),
        check_id="AT7",
        variable="dt_opened",
        stats={"left_censored_accounts": lc_accts, "total_accounts": total_accts, "rate": round(rate, 4)},
    )]


def _check_post_default_tracking(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT8: Whether defaulted accounts have enough post-default observation periods."""
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", "LGD")
    min_months = cfg.get("min_post_default_months", 12)

    if "ind_dft" not in df.columns:
        return []

    defaulted = df[df["ind_dft"] == 1]
    if len(defaulted) == 0:
        return []

    first_default = defaulted.groupby("acct_id", observed=True)["obs_month"].min()
    last_record = df.groupby("acct_id", observed=True)["obs_month"].max()

    common = first_default.index.intersection(last_record.index)
    if len(common) == 0:
        return []

    first_default = first_default[common]
    last_record = last_record[common]

    if pd.api.types.is_datetime64_any_dtype(df["obs_month"]):
        post_months = (last_record - first_default).dt.days / 30.44
    else:
        fd_dt = pd.to_datetime(first_default)
        lr_dt = pd.to_datetime(last_record)
        post_months = (lr_dt - fd_dt).dt.days / 30.44

    under_tracked = int((post_months < min_months).sum())
    total_defaulted = len(common)
    rate = under_tracked / total_defaulted if total_defaulted > 0 else 0

    if under_tracked == 0:
        return []

    return [Finding(
        product=product,
        parameter=parameter,
        impact=impact,
        question=(
            f"{under_tracked:,} defaulted accounts ({rate:.1%}) have fewer than "
            f"{min_months} months of post-default tracking. "
            f"Insufficient workout period may lead to incomplete recovery "
            f"observation and LGD overestimation."
        ),
        check_id="AT8",
        variable="ind_dft",
        stats={
            "under_tracked": under_tracked,
            "total_defaulted": total_defaulted,
            "rate": round(rate, 4),
            "median_post_months": round(float(post_months.median()), 1),
        },
    )]


def _check_disappear_classification(
    last_records: pd.DataFrame, global_max, at_cfg: dict, product: str,
) -> list[Finding]:
    """AT9: Classify disappeared accounts by final status."""
    disappeared = last_records[last_records["obs_month"] < global_max]
    if len(disappeared) == 0:
        return []

    total = len(disappeared)
    cats = {}

    dft_mask = pd.Series(False, index=disappeared.index)
    if "ind_dft" in disappeared.columns:
        dft_mask = disappeared["ind_dft"] == 1
    cats["Default"] = int(dft_mask.sum())

    closed_mask = pd.Series(False, index=disappeared.index)
    if "ind_closed" in disappeared.columns:
        closed_mask = (disappeared["ind_closed"] == 1) & ~dft_mask
    cats["Closed"] = int(closed_mask.sum())

    excl_mask = pd.Series(False, index=disappeared.index)
    if "perf_lvl1" in disappeared.columns:
        excl_mask = (disappeared["perf_lvl1"] == 3) & ~dft_mask & ~closed_mask
    cats["Exclusion"] = int(excl_mask.sum())

    classified = dft_mask | closed_mask | excl_mask
    cats["Censored"] = int((~classified).sum())

    dist = {k: {"count": v, "pct": round(v / total, 4)} for k, v in cats.items()}

    return [Finding(
        product=product,
        parameter=["ERL", "DF", "PD"],
        impact="Medium",
        question=(
            f"{total:,} accounts disappear before the last observation month. "
            f"Breakdown: Default {cats['Default']:,} ({cats['Default']/total:.1%}), "
            f"Closed {cats['Closed']:,} ({cats['Closed']/total:.1%}), "
            f"Exclusion {cats['Exclusion']:,} ({cats['Exclusion']/total:.1%}), "
            f"Censored {cats['Censored']:,} ({cats['Censored']/total:.1%}). "
            f"High censored proportion may bias survival analysis."
        ),
        check_id="AT9",
        variable="acct_id",
        reference_only=True,
        stats={"distribution": dist, "total_disappeared": total},
    )]
