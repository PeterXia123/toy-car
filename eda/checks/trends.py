# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from eda.models import Finding, VariableInfo


def _case_sample(df: pd.DataFrame, mask, cols: list[str]) -> pd.DataFrame | None:
    if "eid" not in df.columns:
        return None
    affected = df.loc[mask, "eid"]
    if len(affected) == 0:
        return None
    sample_id = affected.iloc[0]
    base = ["eid", "rpt_mth"]
    keep = [c for c in base if c in df.columns]
    keep += [c for c in cols if c in df.columns and c not in keep]
    result = df.loc[df["eid"] == sample_id, keep]
    return result.sort_values("rpt_mth") if "rpt_mth" in df.columns else result


def run(
    df: pd.DataFrame,
    checks_cfg: dict,
    variables_cfg: dict[str, VariableInfo],
    product: str,
) -> list[Finding]:
    findings: list[Finding] = []
    tr = checks_cfg.get("trends", {})

    ts_cfg = tr.get("time_series", {})
    if ts_cfg.get("enabled", False) and "rpt_mth" in df.columns:
        findings += _run_time_series(df, ts_cfg, product)

    return findings


def run_account_tracking(
    df: pd.DataFrame,
    checks_cfg: dict,
    product: str,
) -> list[Finding]:
    at = checks_cfg.get("account_tracking", {})
    findings: list[Finding] = []

    if not {"eid", "rpt_mth"}.issubset(df.columns):
        return findings

    last_records = df.groupby("eid", observed=True).last()
    global_max = df["rpt_mth"].max()

    if at.get("right_censoring", {}).get("enabled", False):
        findings += _check_right_censoring(last_records, global_max, at["right_censoring"], product)

    if at.get("right_censoring", {}).get("enabled", False):
        findings += _check_censored_trend(df, last_records, global_max, product)
        findings += _check_attrition_spikes(df, at["right_censoring"], product)

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

        monthly = df.groupby("rpt_mth")[var_name].agg(["mean", "std", "count"]).sort_index()

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
    # Exclude accounts still present in the last month — only flag those
    # that disappeared before the last observation month without reason
    cutoff = global_max - pd.DateOffset(months=1)
    mask = last_records["rpt_mth"] < cutoff
    for col in ["fl_close", "fl_wo", "fl_evt"]:
        if col in last_records.columns:
            mask = mask & (last_records[col] != 1)
    return mask


def _check_right_censoring(
    last_records: pd.DataFrame, global_max, cfg: dict, product: str,
) -> list[Finding]:
    """AT1: Detect attrition spikes that suggest data migration events."""
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", ["ERL", "DF", "LGD"])
    spike_threshold = cfg.get("spike_threshold", 0.04)

    censored_mask = _build_censored_mask(last_records, global_max)
    censored_count = int(censored_mask.sum())
    total_accounts = len(last_records)
    censored_rate = censored_count / total_accounts if total_accounts > 0 else 0

    findings: list[Finding] = []

    if censored_count == 0:
        return findings

    findings.append(Finding(
        product=product,
        parameter=parameter,
        impact="Low",
        question=(
            f"{censored_count:,} accounts ({censored_rate:.1%}) are right-censored — "
            f"they disappear before the second-to-last observation month without closure, chargeoff, or default. "
            f"These accounts cannot contribute complete event histories for ERL or DF estimation."
        ),
        check_id="AT1",
        variable="eid",
        stats={
            "censored_count": censored_count,
            "total_accounts": total_accounts,
            "censored_rate": round(censored_rate, 4),
        },
    ))

    return findings


def _check_attrition_spikes(
    df: pd.DataFrame, cfg: dict, product: str,
) -> list[Finding]:
    """AT10: Flag months where unexplained attrition exceeds spike_threshold — likely data migration."""
    spike_threshold = cfg.get("spike_threshold", 0.04)
    parameter = cfg.get("parameter", ["ERL", "DF", "LGD"])

    months = sorted(df["rpt_mth"].unique())
    if len(months) < 2:
        return []

    indicator_cols = [c for c in ["fl_close", "fl_wo", "fl_evt"] if c in df.columns]
    acct_months = df[["eid", "rpt_mth"]].drop_duplicates()
    present = acct_months.groupby("rpt_mth", observed=True)["eid"].count()

    spikes = []
    for i in range(len(months) - 1):
        m_curr, m_next = months[i], months[i + 1]
        n_active = int(present.get(m_curr, 0))
        if n_active == 0:
            continue
        curr_accts = acct_months.loc[acct_months["rpt_mth"] == m_curr, "eid"]
        next_accts = acct_months.loc[acct_months["rpt_mth"] == m_next, "eid"]
        merged = curr_accts.to_frame().merge(next_accts.to_frame(), on="eid", how="left", indicator=True)
        gone_ids = merged.loc[merged["_merge"] == "left_only", "eid"]
        if len(gone_ids) == 0:
            continue
        dis_records = df.loc[(df["eid"].isin(gone_ids.values)) & (df["rpt_mth"] == m_curr)]
        explained_mask = pd.Series(False, index=dis_records.index)
        for col in indicator_cols:
            explained_mask = explained_mask | (dis_records[col] == 1)
        n_unexplained = int((~explained_mask).sum())
        rate = n_unexplained / n_active
        if rate >= spike_threshold:
            spikes.append({
                "month": str(m_curr),
                "disappeared": n_unexplained,
                "active": n_active,
                "rate": round(rate, 4),
            })

    if not spikes:
        return []

    spike_months = ", ".join(s["month"][:7] for s in spikes)
    worst = max(spikes, key=lambda s: s["rate"])

    return [Finding(
        product=product,
        parameter=parameter,
        impact="High",
        question=(
            f"{len(spikes)} month(s) have unexplained attrition rate >= {spike_threshold:.0%}: "
            f"{spike_months}. "
            f"Worst: {worst['month'][:7]} with {worst['disappeared']:,}/{worst['active']:,} "
            f"({worst['rate']:.1%}) accounts disappearing. "
            f"Spikes of this magnitude strongly suggest a data migration or source system change — "
            f"verify with the data provider whether a system cutover occurred at these snapshots."
        ),
        check_id="AT10",
        variable="eid",
        stats={"spikes": spikes, "threshold": spike_threshold},
    )]


def _check_censored_trend(
    df: pd.DataFrame, last_records: pd.DataFrame, global_max, product: str,
) -> list[Finding]:
    """AT2: Monthly attrition rate — accounts present this month but gone next month without reason."""
    months = sorted(df["rpt_mth"].unique())
    if len(months) < 2:
        return []

    indicator_cols = [c for c in ["fl_close", "fl_wo", "fl_evt"] if c in df.columns]
    acct_months = df[["eid", "rpt_mth"]].drop_duplicates()
    present = acct_months.groupby("rpt_mth", observed=True)["eid"].count()
    trend = {}
    for i in range(len(months) - 1):
        m_curr, m_next = months[i], months[i + 1]
        n_active = int(present.get(m_curr, 0))
        curr_accts = acct_months.loc[acct_months["rpt_mth"] == m_curr, "eid"]
        next_accts = acct_months.loc[acct_months["rpt_mth"] == m_next, "eid"]
        merged = curr_accts.to_frame().merge(next_accts.to_frame(), on="eid", how="left", indicator=True)
        gone_ids = merged.loc[merged["_merge"] == "left_only", "eid"]
        if len(gone_ids) == 0 or n_active == 0:
            trend[str(m_curr)] = {"disappeared": 0, "active": n_active, "rate": 0.0}
            continue
        dis_records = df.loc[(df["eid"].isin(gone_ids.values)) & (df["rpt_mth"] == m_curr)]
        explained_mask = pd.Series(False, index=dis_records.index)
        for col in indicator_cols:
            explained_mask = explained_mask | (dis_records[col] == 1)
        n_unexplained = int((~explained_mask).sum())
        trend[str(m_curr)] = {
            "disappeared": n_unexplained,
            "active": n_active,
            "rate": round(n_unexplained / n_active, 4),
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
        variable="eid",
        reference_only=True,
        stats={"attrition_trend": trend},
    )]


def _check_single_record(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT3: Accounts with only 1 record in the entire dataset."""
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", ["ERL", "PD"])

    counts = df.groupby("eid", observed=True).size()
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
        variable="eid",
        stats={"single_record_count": single, "rate": round(rate, 4)},
    )]


def _check_record_gaps(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT4: Accounts with gaps in rpt_mth sequence."""
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", ["Data", "ERL"])
    max_gap = cfg.get("max_allowed_gap_months", 1)

    sorted_df = df[["eid", "rpt_mth"]]

    if pd.api.types.is_datetime64_any_dtype(sorted_df["rpt_mth"]):
        month_diff = sorted_df.groupby("eid", observed=True)["rpt_mth"].diff().dt.days / 30.44
    else:
        obs_num = pd.to_datetime(sorted_df["rpt_mth"]).astype(np.int64)
        month_diff = sorted_df.groupby("eid", observed=True)[obs_num.name].diff() / (30.44 * 24 * 3600 * 1e9)

    gap_mask = month_diff > (max_gap + 0.5)
    gap_count = int(gap_mask.sum())

    if gap_count == 0:
        return []

    gap_accounts = int(sorted_df.loc[gap_mask, "eid"].nunique())
    total_accounts = int(sorted_df["eid"].nunique())

    return [Finding(
        product=product,
        parameter=parameter,
        impact=impact,
        question=(
            f"{gap_accounts:,} accounts ({gap_accounts/total_accounts:.1%}) have gaps "
            f"in their rpt_mth sequence exceeding {max_gap} month(s). "
            f"Record gaps break cohort continuity and may cause incorrect "
            f"transition matrix estimation."
        ),
        check_id="AT4",
        variable="eid",
        case_data=_case_sample(df, gap_mask.reindex(df.index, fill_value=False), ["fl_close", "fl_wo", "fl_evt", "past_d"]),
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

    monthly_counts = df.groupby("rpt_mth")["eid"].nunique().sort_index()

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
                    f"rpt_mth {month} has {curr:,} unique accounts, "
                    f"below {min_ratio:.0%} of the median ({median_count:.0f}). "
                    f"This suggests incomplete data extraction for that month."
                ),
                check_id="AT5",
                variable="rpt_mth",
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

    profile_vars = ["past_d", "cur_amt", "sc_orig", "sc_curr"]
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
        variable="eid",
        stats={"profile_comparison": comparison},
    )]


def _check_left_censoring(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT7: Accounts whose dt_start is before the earliest rpt_mth in the dataset (left censoring)."""
    impact = cfg.get("impact", "Medium")
    parameter = cfg.get("parameter", "Data")

    if "dt_start" not in df.columns or "rpt_mth" not in df.columns:
        return []
    if not pd.api.types.is_datetime64_any_dtype(df["dt_start"]):
        return []

    window_start = df["rpt_mth"].min()
    valid = df["dt_start"].notna()
    if not valid.any():
        return []

    left_censored = df.loc[valid, "dt_start"] < window_start
    lc_accts = df.loc[valid][left_censored].groupby("eid", observed=True).ngroups if "eid" in df.columns else int(left_censored.sum())
    total_accts = df["eid"].nunique() if "eid" in df.columns else len(df)
    rate = lc_accts / total_accts if total_accts > 0 else 0

    if lc_accts == 0:
        return []

    return [Finding(
        product=product,
        parameter=parameter,
        impact=impact,
        question=(
            f"{lc_accts:,} accounts ({rate:.1%}) have dt_start before data window start ({window_start}). "
            f"These left-censored accounts lack early-life history, which may bias survival analysis and ERL estimates."
        ),
        check_id="AT7",
        variable="dt_start",
        stats={"left_censored_accounts": lc_accts, "total_accounts": total_accts, "rate": round(rate, 4)},
    )]


def _check_post_default_tracking(df: pd.DataFrame, cfg: dict, product: str) -> list[Finding]:
    """AT8: Whether defaulted accounts have enough post-default observation periods."""
    impact = cfg.get("impact", "High")
    parameter = cfg.get("parameter", "LGD")
    min_months = cfg.get("min_post_default_months", 12)

    if "fl_evt" not in df.columns:
        return []

    defaulted = df[df["fl_evt"] == 1]
    if len(defaulted) == 0:
        return []

    first_default = defaulted.groupby("eid", observed=True)["rpt_mth"].min()
    last_record = df.groupby("eid", observed=True)["rpt_mth"].max()

    common = first_default.index.intersection(last_record.index)
    if len(common) == 0:
        return []

    first_default = first_default[common]
    last_record = last_record[common]

    if pd.api.types.is_datetime64_any_dtype(df["rpt_mth"]):
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
            f"Insufficient workout period may lead to incomplete rcv_amt "
            f"observation and LGD overestimation."
        ),
        check_id="AT8",
        variable="fl_evt",
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
    disappeared = last_records[last_records["rpt_mth"] < global_max]
    if len(disappeared) == 0:
        return []

    total = len(disappeared)
    cats = {}

    dft_mask = pd.Series(False, index=disappeared.index)
    if "fl_evt" in disappeared.columns:
        dft_mask = disappeared["fl_evt"] == 1
    cats["Default"] = int(dft_mask.sum())

    closed_mask = pd.Series(False, index=disappeared.index)
    if "fl_close" in disappeared.columns:
        closed_mask = (disappeared["fl_close"] == 1) & ~dft_mask
    cats["Closed"] = int(closed_mask.sum())

    excl_mask = pd.Series(False, index=disappeared.index)
    if "grp1" in disappeared.columns:
        excl_mask = (disappeared["grp1"] == 3) & ~dft_mask & ~closed_mask
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
        variable="eid",
        reference_only=True,
        stats={"distribution": dist, "total_disappeared": total},
    )]
