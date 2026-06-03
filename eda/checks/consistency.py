# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from eda.models import Finding, VariableInfo


def _enabled(checks_cfg: dict, section: str, name: str) -> bool:
    return checks_cfg.get(section, {}).get(name, {}).get("enabled", False)


def _cfg(checks_cfg: dict, section: str, name: str) -> dict:
    return checks_cfg.get(section, {}).get(name, {})


def _col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


_EXAMPLE_BASE_COLS = ["eid", "rpt_mth"]


def _examples(df: pd.DataFrame, mask, variable: str, extra: list = None, n: int = 20) -> pd.DataFrame:
    """Return a slim examples DataFrame with only relevant columns."""
    cols = [c for c in _EXAMPLE_BASE_COLS if c in df.columns]
    if variable in df.columns and variable not in cols:
        cols.append(variable)
    if extra:
        cols.extend(c for c in extra if c in df.columns and c not in cols)
    return df.loc[mask, cols].head(n)


_total_accounts_cache = {"_df_id": None, "_total": 0}


def _total_accounts(df: pd.DataFrame) -> int:
    df_id = id(df)
    if _total_accounts_cache["_df_id"] != df_id:
        _total_accounts_cache["_df_id"] = df_id
        _total_accounts_cache["_total"] = int(df["eid"].nunique()) if "eid" in df.columns else 0
    return _total_accounts_cache["_total"]


def _acct_info(df: pd.DataFrame, mask) -> tuple[str, dict]:
    """Return (text_suffix, stats_dict) for account proportion."""
    if "eid" not in df.columns:
        return "", {}
    t = _total_accounts(df)
    a = int(df.loc[mask, "eid"].nunique())
    r = a / t if t > 0 else 0
    return (
        f" Affected accounts: {a:,} / {t:,} ({r:.1%}).",
        {"affected_accounts": a, "total_accounts": t, "account_rate": round(r, 4)},
    )


def _case_sample(df: pd.DataFrame, mask, cols: list[str]) -> pd.DataFrame | None:
    """Pick one affected account and return its full history with relevant columns."""
    if "eid" not in df.columns:
        return None
    affected = df.loc[mask, "eid"]
    if len(affected) == 0:
        return None
    sample_id = affected.iloc[0]
    base = ["eid", "rpt_mth"]
    keep = [c for c in base if c in df.columns]
    keep += [c for c in cols if c in df.columns and c not in keep]
    return df.loc[df["eid"] == sample_id, keep].sort_values("rpt_mth") if "rpt_mth" in df.columns else df.loc[df["eid"] == sample_id, keep]


# ================================================================
# Top-level runner
# ================================================================

def run(
    df: pd.DataFrame,
    checks_cfg: dict,
    variables_cfg: dict[str, VariableInfo],
    product: str,
) -> list[Finding]:
    findings: list[Finding] = []
    findings += _run_default_logic(df, checks_cfg, product)
    findings += _run_terminal_events(df, checks_cfg, product)
    findings += _run_lgd_checks(df, checks_cfg, variables_cfg, product)
    findings += _run_perf_lvl(df, checks_cfg, product)
    findings += _run_dt_opened(df, checks_cfg, product)
    findings += _run_origination_checks(df, checks_cfg, product)
    return findings


def run_score_alignment(
    df: pd.DataFrame,
    checks_cfg: dict,
    product: str,
) -> list[Finding]:
    sa = checks_cfg.get("score_alignment", {})
    if not sa.get("enabled", False):
        return []
    return _run_score_alignment_checks(df, sa, product)


def run_term_checks(
    df: pd.DataFrame,
    checks_cfg: dict,
    product: str,
) -> list[Finding]:
    tc = checks_cfg.get("term_checks", {})
    return _run_term(df, tc, product)


def run_revolving_checks(
    df: pd.DataFrame,
    checks_cfg: dict,
    product: str,
) -> list[Finding]:
    rc = checks_cfg.get("revolving_checks", {})
    return _run_revolving(df, rc, product)


# ================================================================
# Default Logic (DF1-DF8)
# ================================================================

def _run_default_logic(df: pd.DataFrame, checks_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []
    c = checks_cfg.get("consistency", {})

    # DF1: past_d missing rate per rpt_mth
    if _col(df, "past_d") and _col(df, "rpt_mth"):
        monthly_miss = df["past_d"].isna().groupby(df["rpt_mth"]).mean()
        overall = df["past_d"].isna().mean()
        if overall > 0.05:
            findings.append(Finding(
                product=product,
                parameter="Data",
                impact="Medium",
                question=(
                    f"DPD has {overall:.1%} missing values overall. "
                    f"Missing DPD affects default identification (past_d>=90 rule) "
                    f"and may cause underestimation of PD."
                ),
                check_id="DF1",
                variable="past_d",
                stats={"overall_rate": round(overall, 4),
                       "per_month": {str(k): round(v, 4) for k, v in monthly_miss.items()}},
            ))
        diffs = monthly_miss.diff().abs()
        jump_months = diffs[diffs > 0.10]
        for month, jump in jump_months.items():
            findings.append(Finding(
                product=product, parameter="Data", impact="Medium",
                question=f"DPD missing rate jumped by {jump:.1%} at {month}. Possible score model or data source change.",
                check_id="DF1", variable="past_d",
                stats={"month": str(month), "jump": round(float(jump), 4)},
            ))

    # DF2: past_d missing with chargeoff
    if c.get("dpd_missing_with_chargeoff", {}).get("enabled") and _col(df, "past_d") and _col(df, "fl_wo"):
        cfg = c["dpd_missing_with_chargeoff"]
        mask = df["past_d"].isna() & (df["fl_wo"] == 1)
        count = mask.sum()
        if count > 0:
            q_text = (
                f"Found {count:,} records where DPD is missing but fl_wo=1 (charge-off). "
                f"These accounts' default status depends entirely on charge-off indicator. "
                f"Is the DPD missing by design or a data gap?"
            )
            extra_stats = {}
            if _col(df, "eid"):
                affected_accts = int(df.loc[mask, "eid"].nunique())
                total_accts = _total_accounts(df)
                acct_pct = affected_accts / total_accts
                q_text += f" Affected accounts: {affected_accts:,} / {total_accts:,} ({acct_pct:.1%})."
                extra_stats = {"affected_accounts": affected_accts, "total_accounts": total_accts, "account_rate": round(acct_pct, 4)}
            findings.append(Finding(
                product=product,
                parameter=cfg.get("parameter", "Data"),
                impact=cfg.get("impact", "High"),
                question=q_text,
                check_id="DF2", variable="past_d",
                examples=_examples(df, mask, variable="past_d"),
                case_data=_case_sample(df, mask, ["past_d", "fl_wo", "fl_evt"]),
                stats={"count": int(count), "rate": round(count / len(df), 4), **extra_stats},
            ))

    # DF6: consecutive defaults without cure
    if c.get("consecutive_defaults", {}).get("enabled") and _col(df, "new_evt") and _col(df, "fl_evt") and _col(df, "eid") and _col(df, "rpt_mth"):
        cfg = c["consecutive_defaults"]
        ntd = df[df["new_evt"] == 1][["eid", "rpt_mth"]].copy()
        multi = ntd.groupby("eid", observed=True).size()
        multi_accts = multi[multi > 1]
        if len(multi_accts) > 0:
            sample_accts = multi_accts.head(20).index.tolist()
            sample = df[df["eid"].isin(sample_accts)][["eid", "rpt_mth", "fl_evt", "new_evt", "past_d"] if _col(df, "past_d") else ["eid", "rpt_mth", "fl_evt", "new_evt"]]
            total_accts = _total_accounts(df)
            acct_pct = len(multi_accts) / total_accts
            findings.append(Finding(
                product=product,
                parameter=cfg.get("parameter", "PD"),
                impact=cfg.get("impact", "High"),
                question=(
                    f"Found {len(multi_accts):,} accounts with multiple new_to_default events "
                    f"(consecutive defaults without cure). The root cause may be DPD fluctuating "
                    f"around the 90-day threshold without triggering a cure. "
                    f"Affected accounts: {len(multi_accts):,} / {total_accts:,} ({acct_pct:.1%})."
                ),
                check_id="DF6", variable="new_evt",
                examples=sample.sort_values(["eid", "rpt_mth"]).head(40),
                case_data=_case_sample(df, df["eid"].isin(multi_accts.index), ["fl_evt", "new_evt", "past_d", "grp1"]),
                stats={"affected_accounts": int(len(multi_accts)), "total_accounts": total_accts, "account_rate": round(acct_pct, 4)},
            ))

    # DF7: 12-month forward default rate trend (grp2=1 / grp1=0)
    if _col(df, "rpt_mth"):
        func_df = df[df["grp1"] == 0] if _col(df, "grp1") else df
        dft_rate = None
        if _col(func_df, "grp2"):
            dft_rate = func_df.groupby("rpt_mth")["grp2"].mean().sort_index()
        ntd_rate = None
        if _col(func_df, "new_evt"):
            ntd_rate = func_df.groupby("rpt_mth")["new_evt"].mean().sort_index()
        if dft_rate is not None or ntd_rate is not None:
            findings.append(Finding(
                product=product, parameter="PD", impact="Low",
                question="Default rate trend data generated. 12-month forward default rate (grp2=1) on functional accounts (grp1=0).",
                check_id="DF7", variable="grp2",
                reference_only=True,
                stats={
                    "default_rate": {str(k): round(float(v), 6) for k, v in dft_rate.items()} if dft_rate is not None else {},
                    "new_to_dft_rate": {str(k): round(float(v), 6) for k, v in ntd_rate.items()} if ntd_rate is not None else {},
                },
            ))

    # DF9: past_c vs past_d consistency — past_c should approximate ceil(past_d/30)
    if _col(df, "past_c") and _col(df, "past_d"):
        valid = df[df["past_c"].notna() & df["past_d"].notna()]
        if len(valid) > 0:
            expected_cpd = np.ceil(valid["past_d"] / 30).astype(int)
            mismatch = (valid["past_c"] != expected_cpd) & (valid["past_d"] > 0)
            mismatch_count = int(mismatch.sum())
            if mismatch_count > 0:
                rate = mismatch_count / len(valid)
                findings.append(Finding(
                    product=product, parameter="Data", impact="Low",
                    question=(
                        f"past_c vs past_d inconsistency: {mismatch_count:,} records ({rate:.1%}) "
                        f"where past_c ≠ ceil(past_d/30). This may indicate different cycle definitions "
                        f"or independent derivation of past_c."
                    + (_acct_info(valid, mismatch)[0] if _col(valid, "eid") else "")
                    ),
                    check_id="DF9", variable="past_c",
                    examples=valid.loc[mismatch].head(20) if mismatch.any() else None,
                    case_data=_case_sample(df, mismatch.reindex(df.index, fill_value=False), ["past_c", "past_d"]) if mismatch.any() else None,
                    stats={"mismatch_count": mismatch_count, "rate": round(rate, 4), **(_acct_info(valid, mismatch)[1] if _col(valid, "eid") else {})},
                ))

    return findings


# ================================================================
# Terminal Events (TE1-TE8)
# ================================================================

def _run_terminal_events(df: pd.DataFrame, checks_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []
    c = checks_cfg.get("consistency", {})

    # TE1: fl_close cummax violation (1 → 0)
    if c.get("closed_temporal", {}).get("enabled") and _col(df, "fl_close") and _col(df, "eid") and _col(df, "rpt_mth"):
        cfg = c["closed_temporal"]
        violations, examples = _check_cummax_violation(df, "eid", "fl_close")
        if violations > 0:
            t = _total_accounts(df)
            findings.append(Finding(
                product=product, parameter=cfg.get("parameter", ["ERL", "LGD"]),
                impact=cfg.get("impact", "High"),
                question=(
                    f"fl_close violates cummax logic: {violations:,} accounts have fl_close "
                    f"reverting from 1 to 0 (re-opening after closure). "
                    f"Affected accounts: {violations:,} / {t:,} ({violations/t:.1%}). "
                    f"This breaks terminal event logic for ERL and may cause LGD miscalculation."
                ),
                check_id="TE1", variable="fl_close",
                examples=examples,
                case_data=_case_sample(df, (df.groupby("eid", observed=True)["fl_close"].shift(1) == 1) & (df["fl_close"] == 0), ["fl_close", "past_d", "cur_amt"]),
                stats={"violation_accounts": violations, "total_accounts": t, "account_rate": round(violations / t, 4)},
            ))

    # TE2: fl_wo cummax violation
    if c.get("chargeoff_temporal", {}).get("enabled") and _col(df, "fl_wo") and _col(df, "eid") and _col(df, "rpt_mth"):
        cfg = c["chargeoff_temporal"]
        violations, examples = _check_cummax_violation(df, "eid", "fl_wo")
        if violations > 0:
            findings.append(Finding(
                product=product, parameter=cfg.get("parameter", ["LGD"]),
                impact=cfg.get("impact", "High"),
                question=(
                    f"fl_wo violates cummax logic: {violations:,} accounts have fl_wo "
                    f"reverting from 1 to 0. Once charged off, the indicator should remain 1. "
                    f"Affected accounts: {violations:,} / {_total_accounts(df):,} ({violations/_total_accounts(df):.1%}). "
                    f"This affects LGD calculation."
                ),
                check_id="TE2", variable="fl_wo",
                examples=examples,
                case_data=_case_sample(df, (df.groupby("eid", observed=True)["fl_wo"].shift(1) == 1) & (df["fl_wo"] == 0), ["fl_wo", "fl_close", "past_d", "cur_amt"]),
                stats={"violation_accounts": violations, "total_accounts": _total_accounts(df), "account_rate": round(violations / _total_accounts(df), 4)},
            ))

    # TE3: closed but cur_amt > 0
    if c.get("closed_vs_balance", {}).get("enabled") and _col(df, "fl_close") and _col(df, "cur_amt"):
        cfg = c["closed_vs_balance"]
        mask = (df["fl_close"] == 1) & (df["cur_amt"] > 0)
        count = mask.sum()
        if count > 0:
            rate = count / (df["fl_close"] == 1).sum() if (df["fl_close"] == 1).sum() > 0 else 0
            findings.append(Finding(
                product=product, parameter=cfg.get("parameter", ["LGD"]),
                impact=cfg.get("impact", "Medium"),
                question=(
                    f"{count:,} records ({rate:.1%} of closed accounts) have fl_close=1 "
                    f"but cur_amt > 0. Closed accounts should typically have zero cur_amt."
                    + _acct_info(df, mask)[0]
                ),
                check_id="TE3", variable="fl_close",
                examples=_examples(df, mask, variable="fl_close"),
                case_data=_case_sample(df, mask, ["fl_close", "cur_amt", "past_d"]),
                stats={"count": int(count), "rate": round(rate, 4), **_acct_info(df, mask)[1]},
            ))

    # TE4: chargeoff after closure
    if c.get("post_close_chargeoff", {}).get("enabled") and _col(df, "fl_close") and _col(df, "fl_wo") and _col(df, "eid") and _col(df, "rpt_mth"):
        cfg = c["post_close_chargeoff"]
        sorted_df = df
        first_closed = sorted_df[sorted_df["fl_close"] == 1].groupby("eid", observed=True)["rpt_mth"].first()
        first_co_after = sorted_df[sorted_df["fl_wo"] == 1].groupby("eid", observed=True)["rpt_mth"].first()
        common = first_closed.index.intersection(first_co_after.index)
        if len(common) > 0:
            post_close_co = first_co_after[common] > first_closed[common]
            violation_count = post_close_co.sum()
            if violation_count > 0:
                findings.append(Finding(
                    product=product, parameter=cfg.get("parameter", ["LGD", "ERL"]),
                    impact=cfg.get("impact", "High"),
                    question=(
                        f"{int(violation_count):,} accounts have charge-off occurring AFTER closure. "
                        f"This creates ambiguity in terminal event definition for ERL. "
                    f"Affected accounts: {int(violation_count):,} / {_total_accounts(df):,} ({int(violation_count)/_total_accounts(df):.1%})."
                    ),
                    check_id="TE4", variable="fl_wo",
                    case_data=_case_sample(df, df["eid"].isin(post_close_co[post_close_co].index), ["fl_close", "fl_wo", "cur_amt", "past_d"]),
                    stats={"violation_accounts": int(violation_count), "total_accounts": _total_accounts(df), "account_rate": round(int(violation_count)/df["eid"].nunique(), 4)},
                ))

    # TE5: fl_wo=1 but fl_close=0
    if c.get("closed_vs_chargeoff", {}).get("enabled") and _col(df, "fl_wo") and _col(df, "fl_close"):
        cfg = c["closed_vs_chargeoff"]
        mask = (df["fl_wo"] == 1) & (df["fl_close"] == 0)
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter=cfg.get("parameter", ["LGD", "ERL"]),
                impact=cfg.get("impact", "Medium"),
                question=(
                    f"{count:,} records have fl_wo=1 but fl_close=0. "
                    f"Charge-off typically implies account closure. "
                    f"Is this intentional for new-to-default charge-off accounts?"
                    + _acct_info(df, mask)[0]
                ),
                check_id="TE5", variable="fl_wo",
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # TE6: chargeoff before dt_start
    if _col(df, "rpt_mth") and _col(df, "dt_start") and _col(df, "fl_wo"):
        both_valid = df["rpt_mth"].notna() & df["dt_start"].notna()
        if both_valid.any() and pd.api.types.is_datetime64_any_dtype(df["rpt_mth"]) and pd.api.types.is_datetime64_any_dtype(df["dt_start"]):
            mask = (df["rpt_mth"] < df["dt_start"]) & both_valid & (df["fl_wo"] == 1)
            count = int(mask.sum())
            if count > 0:
                findings.append(Finding(
                    product=product, parameter="Data", impact="Low",
                    question=(
                        f"{count:,} records have fl_wo=1 before dt_start. "
                        f"Chargeoff should not occur before account opening date."
                    + _acct_info(df, mask)[0]
                    ),
                    check_id="TE6", variable="fl_wo",
                    examples=_examples(df, mask, variable="fl_wo"),
                    case_data=_case_sample(df, mask, ["fl_wo", "dt_start"]),
                    stats={"count": count, **_acct_info(df, mask)[1]},
                ))

    # TE7 & TE8: trend data for fl_close and fl_wo
    for col, check_id in [("fl_close", "TE7"), ("fl_wo", "TE8")]:
        if _col(df, col) and _col(df, "rpt_mth"):
            trend = df.groupby("rpt_mth")[col].agg(["sum", "count"])
            trend.columns = ["value_1", "n_records"]
            trend["value_0"] = trend["n_records"] - trend["value_1"]
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=f"{col} trend data generated by rpt_mth.",
                check_id=check_id, variable=col,
                reference_only=True,
                stats={"trend": {str(k): {"value_0": int(r["value_0"]), "value_1": int(r["value_1"]),
                                           "n_records": int(r["n_records"])}
                                 for k, r in trend.iterrows()}},
            ))

    return findings


def _check_cummax_violation(df: pd.DataFrame, id_col: str, indicator_col: str, time_col: str = "rpt_mth") -> tuple:
    """Return (violation_account_count, examples_df) in one pass."""
    shifted = df.groupby(id_col, observed=True)[indicator_col].shift(1)
    violations = (shifted == 1) & (df[indicator_col] == 0)
    n_accts = int(df.loc[violations, id_col].nunique())
    examples = pd.DataFrame()
    if n_accts > 0:
        accts = df.loc[violations, id_col].unique()[:5]
        cols = [id_col, time_col, indicator_col]
        for c in ("past_d", "cur_amt"):
            if c in df.columns:
                cols.append(c)
        examples = df[df[id_col].isin(accts)][cols].head(20)
    return n_accts, examples


# ================================================================
# Score-Default Alignment (SA1-SA7)
# ================================================================

def _run_score_alignment_checks(df: pd.DataFrame, sa_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []
    score_cols = sa_cfg.get("score_columns", [])
    default_col = sa_cfg.get("default_column", "grp2")
    n_segments = sa_cfg.get("n_segments", 10)
    mono_tol = sa_cfg.get("monotonicity_tolerance", 1)
    max_high_rate = sa_cfg.get("max_high_score_default_rate", 0.02)
    impact = sa_cfg.get("impact", "High")
    parameter = sa_cfg.get("parameter", ["Score_Alignment", "PD", "SICR"])

    if not _col(df, default_col):
        return findings

    # Filter to functional accounts (grp1=0) for score-default alignment
    if _col(df, "grp1"):
        sa_df = df[df["grp1"] == 0]
    else:
        sa_df = df

    for score_col in score_cols:
        if not _col(sa_df, score_col):
            continue

        valid = sa_df[[score_col, default_col]].dropna()
        if len(valid) < 100:
            continue

        n_unique = valid[score_col].nunique()
        effective_segments = min(n_segments, max(3, n_unique // 5))

        # SA1: Monotonicity
        valid["segment"] = pd.qcut(valid[score_col], effective_segments, labels=False, duplicates="drop")
        seg_rates = valid.groupby("segment")[default_col].mean().sort_index()

        violations = 0
        ascending = seg_rates.iloc[-1] > seg_rates.iloc[0]
        for i in range(1, len(seg_rates)):
            if ascending and seg_rates.iloc[i] < seg_rates.iloc[i - 1]:
                violations += 1
            elif not ascending and seg_rates.iloc[i] > seg_rates.iloc[i - 1]:
                violations += 1

        if violations > mono_tol:
            findings.append(Finding(
                product=product, parameter=parameter, impact=impact,
                question=(
                    f"Score `{score_col}` violates monotonicity in {violations} out of "
                    f"{len(seg_rates)-1} adjacent segment pairs (tolerance: {mono_tol}). "
                    f"This suggests the score may not align well with the expected default definition."
                ),
                check_id="SA1", variable=score_col,
                stats={"segment_default_rates": {int(k): round(float(v), 4) for k, v in seg_rates.items()},
                       "violations": violations, "direction": "ascending" if ascending else "descending"},
            ))

        # SA3: Score segment × rpt_mth default rate (for heatmap)
        if _col(sa_df, "rpt_mth"):
            valid_with_month = sa_df[[score_col, default_col, "rpt_mth"]].dropna()
            if len(valid_with_month) > 100:
                sa3_segs = min(effective_segments, 5)
                valid_with_month["segment"] = pd.qcut(valid_with_month[score_col], sa3_segs, labels=False, duplicates="drop")
                heatmap = valid_with_month.groupby(["segment", "rpt_mth"])[default_col].mean().unstack(fill_value=0)
                findings.append(Finding(
                    product=product, parameter=parameter, impact="Low",
                    question=f"Score `{score_col}` segment-level default rate trend generated.",
                    check_id="SA3", variable=score_col,
                    reference_only=True,
                    stats={"heatmap": {int(seg): {str(m): round(float(v), 4) for m, v in row.items()}
                                       for seg, row in heatmap.iterrows()}},
                ))

        # SA4: Extreme score values
        total_valid = len(valid)
        zero_count = int((valid[score_col] == 0).sum())
        over_1000 = int((valid[score_col] > 1000).sum())
        if zero_count > 0 or over_1000 > 0:
            zero_pct = zero_count / total_valid if total_valid > 0 else 0
            over_pct = over_1000 / total_valid if total_valid > 0 else 0
            findings.append(Finding(
                product=product, parameter=parameter, impact="Low",
                question=(
                    f"Score `{score_col}` has {zero_count:,} records ({zero_pct:.2%}) with value=0 and "
                    f"{over_1000:,} records ({over_pct:.2%}) with value>1000. "
                    f"These extreme values should be excluded from final score calculation."
                ),
                check_id="SA4", variable=score_col,
                stats={"zero_count": zero_count, "zero_rate": round(zero_pct, 4),
                       "over_1000": over_1000, "over_1000_rate": round(over_pct, 4)},
            ))

        # SA5/SA6: Missing rate trend on functional accounts (grp1=0)
        if _col(df, "rpt_mth"):
            func_df = df[df["grp1"] == 0] if _col(df, "grp1") else df
            monthly_miss = func_df[score_col].isna().groupby(func_df["rpt_mth"]).mean()
            overall_miss = func_df[score_col].isna().mean()
            is_issue = overall_miss > 0.05
            findings.append(Finding(
                product=product, parameter=parameter, impact="Low",
                question=(
                    f"Score `{score_col}` has {overall_miss:.1%} missing values overall. "
                    + ("Check if score file merge is complete." if is_issue else "Missing rate within acceptable range.")
                ),
                check_id="SA6" if score_col == "sc_curr" else "SA5",
                variable=score_col,
                reference_only=not is_issue,
                stats={"overall_rate": round(overall_miss, 4),
                       "per_month": {str(k): round(float(v), 4) for k, v in monthly_miss.items()}},
            ))

        # SA7: Distribution drift
        if _col(df, "rpt_mth"):
            monthly_mean = df.groupby("rpt_mth")[score_col].mean()
            if len(monthly_mean) > 3:
                z = (monthly_mean - monthly_mean.mean()) / monthly_mean.std()
                jumps = z.abs() > 2.5
                if jumps.any():
                    jump_months = z[jumps]
                    findings.append(Finding(
                        product=product, parameter=parameter, impact="High",
                        question=(
                            f"Score `{score_col}` distribution shows significant drift at: "
                            f"{', '.join(str(m) for m in jump_months.index)}. "
                            f"This may indicate a scorecard model update without backscoring."
                        ),
                        check_id="SA7", variable=score_col,
                        stats={"drift_months": {str(k): round(float(v), 2) for k, v in jump_months.items()},
                               "monthly_mean": {str(k): round(float(v), 2) for k, v in monthly_mean.items()}},
                    ))

        # SA8: Segment default rate trend — visual check for definition alignment
        if _col(sa_df, "rpt_mth"):
            ts_valid = sa_df[[score_col, default_col, "rpt_mth"]].dropna()
            if len(ts_valid) > 200:
                n_seg = min(effective_segments, 5)
                ts_valid["segment"] = pd.qcut(
                    ts_valid[score_col], n_seg, labels=False, duplicates="drop"
                )
                seg_bounds = ts_valid.groupby("segment")[score_col].agg(["min", "max"])
                seg_labels = {
                    int(seg): f"{int(row['min'])}-{int(row['max'])}"
                    for seg, row in seg_bounds.iterrows()
                }
                pivot = ts_valid.groupby(["rpt_mth", "segment"])[default_col].mean().unstack()

                if len(pivot) >= 3 and len(pivot.columns) >= 2:
                    findings.append(Finding(
                        product=product, parameter=parameter, impact="Low",
                        question=(
                            f"Score `{score_col}` segment-level default rate trend by rpt_mth. "
                            f"Review the chart for simultaneous shifts across segments — "
                            f"if all segments move together at the same month, it may indicate "
                            f"a default definition change rather than genuine risk migration."
                        ),
                        check_id="SA8", variable=score_col,
                        reference_only=True,
                        stats={
                            "segment_labels": seg_labels,
                            "segment_rates": {
                                int(seg): {str(m): round(float(v), 4) for m, v in pivot[seg].items()}
                                for seg in pivot.columns
                            },
                        },
                    ))

    return findings


# ================================================================
# LGD Checks (LG1-LG10)
# ================================================================

def _run_lgd_checks(df: pd.DataFrame, checks_cfg: dict, variables_cfg: dict[str, VariableInfo], product: str) -> list[Finding]:
    findings: list[Finding] = []

    # LG1: cur_amt negative
    if _col(df, "cur_amt"):
        mask = df["cur_amt"] < 0
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"Balance has {count:,} negative records ({count/len(df):.2%}). Negative cur_amt may indicate data errors or special account types.",
                check_id="LG1", variable="cur_amt",
                examples=_examples(df, mask, variable="cur_amt"),
                case_data=_case_sample(df, mask, ["cur_amt", "past_d", "fl_evt"]),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # LG2: cur_amt=0 but past_d>0
    if _col(df, "cur_amt") and _col(df, "past_d"):
        mask = (df["cur_amt"] == 0) & (df["past_d"] > 0)
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"{count:,} records have cur_amt=0 but DPD>0. Zero-cur_amt delinquent accounts may affect default definition.",
                check_id="LG2", variable="cur_amt",
                case_data=_case_sample(df, mask, ["cur_amt", "past_d", "fl_evt"]),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # LG3: cur_amt missing at default
    if _col(df, "cur_amt") and _col(df, "fl_evt"):
        mask = (df["fl_evt"] == 1) & df["cur_amt"].isna()
        count = mask.sum()
        total_dft = (df["fl_evt"] == 1).sum()
        if count > 0 and total_dft > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"{count:,} defaulted records ({count/total_dft:.1%} of defaults) have missing cur_amt. Balance at default is required for LGD calculation." + _acct_info(df, mask)[0],
                check_id="LG3", variable="cur_amt",
                case_data=_case_sample(df, mask, ["cur_amt", "fl_evt", "rcv_amt"]),
                stats={"count": int(count), "rate_of_defaults": round(count / total_dft, 4), **_acct_info(df, mask)[1]},
            ))

    # LG4: rcv_amt negative trend
    if _col(df, "rcv_amt") and _col(df, "rpt_mth"):
        neg_mask = df["rcv_amt"] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            monthly = df[neg_mask].groupby("rpt_mth").size()
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"Recovery field has {neg_count:,} negative values. Negative rcv_amt may indicate reversed payments or data errors." + _acct_info(df, neg_mask)[0],
                check_id="LG4", variable="rcv_amt",
                case_data=_case_sample(df, neg_mask, ["rcv_amt", "cur_amt", "fl_evt", "fl_wo"]),
                stats={"total_negative": int(neg_count),
                       "per_month": {str(k): int(v) for k, v in monthly.items()}, **_acct_info(df, neg_mask)[1]},
            ))

    # LG6: rcv_amt trend
    if _col(df, "rcv_amt") and _col(df, "rpt_mth"):
        trend = df.groupby("rpt_mth")["rcv_amt"].agg(["mean", "sum", "count"])
        trend.columns = ["recovery_avg", "recovery_sum", "n_records"]
        findings.append(Finding(
            product=product, parameter="LGD", impact="Medium",
            question="Recovery trend data generated (average and sum by rpt_mth).",
            check_id="LG6", variable="rcv_amt",
            reference_only=True,
            stats={"recovery_trend": {str(k): {"avg": round(float(r["recovery_avg"]), 2),
                                                "sum": round(float(r["recovery_sum"]), 2),
                                                "n": int(r["n_records"])}
                                       for k, r in trend.iterrows()}},
        ))

    # LG7: non-default, non-CO accounts with rcv_amt > 0
    if _col(df, "rcv_amt") and _col(df, "fl_evt") and _col(df, "fl_wo"):
        mask = (df["fl_evt"] == 0) & (df["fl_wo"] == 0) & (df["rcv_amt"].notna()) & (df["rcv_amt"] > 0)
        count = int(mask.sum())
        if count > 0:
            n_accts = df.loc[mask, "eid"].nunique() if _col(df, "eid") else count
            t = _total_accounts(df)
            r = n_accts / t if t > 0 else 0
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=(
                    f"{count:,} records have rcv_amt>0 but fl_evt=0 and fl_wo=0. "
                    f"Non-default, non-chargeoff accounts should not have rcv_amt amounts."
                    f" Affected accounts: {n_accts:,} / {t:,} ({r:.1%})."
                ),
                check_id="LG7", variable="rcv_amt",
                case_data=_case_sample(df, mask, ["rcv_amt", "fl_evt", "fl_wo", "cur_amt"]),
                stats={"records": int(count), "affected_accounts": n_accts, "total_accounts": t, "account_rate": round(r, 4)},
            ))

    # LG8: ann_rate = 0
    if _col(df, "ann_rate"):
        zero_rate = (df["ann_rate"] == 0).sum()
        total_valid = df["ann_rate"].notna().sum()
        if zero_rate > 0 and total_valid > 0:
            ratio = zero_rate / total_valid
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"Interest rate is 0 for {zero_rate:,} records ({ratio:.1%}). Is 0% interest rate valid or does it represent missing data?",
                check_id="LG8", variable="ann_rate",
                stats={"zero_count": int(zero_rate), "rate": round(ratio, 4)},
            ))

    # LG9: ann_rate extreme
    if _col(df, "ann_rate"):
        extreme = (df["ann_rate"] > 0.5) | (df["ann_rate"] < 0)
        count = extreme.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"Interest rate has {count:,} records with extreme values (>50% or <0). Check data quality." + _acct_info(df, extreme)[0],
                check_id="LG9", variable="ann_rate",
                examples=_examples(df, extreme, variable="ann_rate"),
                case_data=_case_sample(df, extreme, ["ann_rate", "cur_amt"]),
                stats={"count": int(count), **_acct_info(df, extreme)[1]},
            ))

    # LG10: cur_amt trend
    if _col(df, "cur_amt") and _col(df, "rpt_mth"):
        trend = df.groupby("rpt_mth")["cur_amt"].agg(["sum", "mean", "median"])
        findings.append(Finding(
            product=product, parameter="Data", impact="Low",
            question="Balance trend data generated (sum, mean, median by rpt_mth).",
            check_id="LG10", variable="cur_amt",
            reference_only=True,
            stats={"balance_trend": {str(k): {"sum": round(float(r["sum"]), 2), "mean": round(float(r["mean"]), 2), "median": round(float(r["median"]), 2)} for k, r in trend.iterrows()}},
        ))

    # LG11: removed (Keep=N in check inventory)

    # LG12 & LG13: LGD workout rcv_amt rate
    _lgd_workout_checks(df, findings, product)


    # NB2: fwd_amt negative
    if _col(df, "fwd_amt"):
        neg = df["fwd_amt"] < 0
        neg_count = int(neg.sum())
        if neg_count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=(
                    f"fwd_amt has {neg_count:,} negative values. "
                    f"Negative cur_amt at default is unusual and will distort LGD calculation."
                ),
                check_id="NB2", variable="fwd_amt",
                examples=_examples(df, neg, variable="fwd_amt"),
                case_data=_case_sample(df, neg, ["fwd_amt", "cur_amt", "fl_evt"]),
                stats={"negative_count": neg_count, **_acct_info(df, neg)[1]},
            ))

    # NB3: fwd_amt vs cur_amt consistency at default
    if _col(df, "fwd_amt") and _col(df, "cur_amt") and _col(df, "new_evt"):
        ntd_mask = df["new_evt"] == 1
        valid = df.loc[ntd_mask].dropna(subset=["fwd_amt", "cur_amt"])
        if len(valid) > 0:
            diff = (valid["fwd_amt"] - valid["cur_amt"]).abs()
            rel_diff = diff / valid["cur_amt"].abs().clip(lower=1)
            large_diff = int((rel_diff > 0.1).sum())
            if large_diff > 0:
                rate = large_diff / len(valid)
                findings.append(Finding(
                    product=product, parameter="LGD", impact="Low",
                    question=(
                        f"{large_diff:,} new-to-default records ({rate:.1%}) have >10% difference "
                        f"between fwd_amt and cur_amt. These should align at the point of default. "
                        f"Large discrepancies suggest a timing or derivation issue."
                    + _acct_info(valid, rel_diff > 0.1)[0]
                    ),
                    check_id="NB3", variable="fwd_amt",
                    case_data=_case_sample(df, df.index.isin(valid.index[rel_diff > 0.1]), ["fwd_amt", "cur_amt", "new_evt", "fl_evt"]),
                    stats={"large_diff_count": large_diff, "rate": round(rate, 4),
                           "median_rel_diff": round(float(rel_diff.median()), 4), **_acct_info(valid, rel_diff > 0.1)[1]},
                ))

    # NB4: fwd_amt = 0 for defaulted accounts
    if _col(df, "fwd_amt") and _col(df, "fl_evt"):
        zero_dft = (df["fwd_amt"] == 0) & (df["fl_evt"] == 1)
        zero_count = int(zero_dft.sum())
        if zero_count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Medium",
                question=(
                    f"{zero_count:,} defaulted records have fwd_amt=0. "
                    f"Zero cur_amt at default means LGD is undefined for these accounts. "
                    f"Check if this is data error or legitimate (e.g., fully recovered before default flag)."
                ),
                check_id="NB4", variable="fwd_amt",
                case_data=_case_sample(df, zero_dft, ["fwd_amt", "cur_amt", "fl_evt"]),
                stats={"zero_count": zero_count, **_acct_info(df, zero_dft)[1]},
            ))

    # MD1: mos_to_evt missing for records before new_evt event
    if _col(df, "mos_to_evt") and _col(df, "new_evt") and _col(df, "eid"):
        ntd_accts = df.loc[df["new_evt"] == 1, "eid"].unique()
        if len(ntd_accts) > 0:
            pre_dft = df[df["eid"].isin(ntd_accts)]
            missing = int(pre_dft["mos_to_evt"].isna().sum())
            total = len(pre_dft)
            if missing > 0:
                rate = missing / total
                findings.append(Finding(
                    product=product, parameter="PD", impact="Low",
                    question=(
                        f"Among accounts that eventually default (new_evt=1), "
                        f"{missing:,} of {total:,} records ({rate:.1%}) have missing mos_to_evt. "
                        f"This affects PD term structure estimation."
                    ),
                    check_id="MD1", variable="mos_to_evt",
                    stats={"missing_count": missing, "total_records": total, "missing_rate": round(rate, 4)},
                ))

    # MD2: mos_to_evt negative
    if _col(df, "mos_to_evt"):
        neg = df["mos_to_evt"] < 0
        neg_count = int(neg.sum())
        if neg_count > 0:
            findings.append(Finding(
                product=product, parameter="PD", impact="Low",
                question=(
                    f"mos_to_evt has {neg_count:,} negative values. "
                    f"Negative months-to-default is logically impossible and indicates "
                    f"a derivation error (possibly dt_next_dft < rpt_mth)."
                ),
                check_id="MD2", variable="mos_to_evt",
                examples=_examples(df, neg, variable="mos_to_evt"),
                case_data=_case_sample(df, neg, ["mos_to_evt", "fl_evt", "new_evt", "mos_bk"]),
                stats={"negative_count": neg_count, **_acct_info(df, neg)[1]},
            ))

    # MD3: mos_to_evt distribution trend
    if _col(df, "mos_to_evt") and _col(df, "rpt_mth"):
        valid = df[df["mos_to_evt"].notna() & (df["mos_to_evt"] >= 0)]
        if len(valid) > 100:
            trend = valid.groupby("rpt_mth")["mos_to_evt"].agg(["mean", "median", "count"])
            findings.append(Finding(
                product=product, parameter="PD", impact="Low",
                question="mos_to_evt distribution trend generated (mean, median by rpt_mth).",
                check_id="MD3", variable="mos_to_evt",
                reference_only=True,
                stats={"mths_to_dft_trend": {
                    str(k): {"mean": round(float(r["mean"]), 2), "median": round(float(r["median"]), 2),
                             "count": int(r["count"])}
                    for k, r in trend.iterrows()
                }},
            ))

    return findings


# ================================================================
# perf_lvl Verification (PL1-PL4)
# ================================================================

def _run_perf_lvl(df: pd.DataFrame, checks_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []
    c = checks_cfg.get("consistency", {})

    if not c.get("perf_lvl_logic", {}).get("enabled"):
        return findings

    # PL2: distribution trend
    if _col(df, "grp1") and _col(df, "rpt_mth"):
        dist = df.groupby("rpt_mth")["grp1"].value_counts(normalize=True).unstack(fill_value=0)
        findings.append(Finding(
            product=product, parameter="Data", impact="Low",
            question="grp1 distribution trend generated.",
            check_id="PL2", variable="grp1",
            reference_only=True,
            stats={"distribution": {str(k): {str(c): round(float(v), 4) for c, v in row.items()} for k, row in dist.iterrows()}},
        ))

    # PL3: functional ratio
    if _col(df, "grp1"):
        func_rate = (df["grp1"] == 0).mean()
        if func_rate < 0.5:
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=f"Only {func_rate:.1%} of records have grp1=0 (functional). Expected >50%. Large exclusion ratio may reduce sample size for downstream models.",
                check_id="PL3", variable="grp1",
                stats={"functional_rate": round(func_rate, 4)},
            ))

    return findings


# ================================================================
# dt_start Checks (DO1-DO3)
# ================================================================

def _run_dt_opened(df: pd.DataFrame, checks_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []

    # DO1: dt_start missing
    if _col(df, "dt_start"):
        rate = df["dt_start"].isna().mean()
        if rate > 0.01:
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=f"dt_start has {rate:.1%} missing values. Missing open dates cause grp1=4 exclusion and affect mos_bk/Loan Term calculation.",
                check_id="DO1", variable="dt_start",
                stats={"missing_rate": round(rate, 4)},
            ))

    # DO2: dt_start month > rpt_mth (compare at month level)
    if _col(df, "dt_start") and _col(df, "rpt_mth"):
        both_valid = df["dt_start"].notna() & df["rpt_mth"].notna()
        if both_valid.any() and pd.api.types.is_datetime64_any_dtype(df["rpt_mth"]) and pd.api.types.is_datetime64_any_dtype(df["dt_start"]):
            dt_month = df["dt_start"].dt.to_period("M")
            obs_month_p = df["rpt_mth"].dt.to_period("M")
            mask = (dt_month > obs_month_p) & both_valid
            count = int(mask.sum())
            if count > 0:
                findings.append(Finding(
                    product=product, parameter="Data", impact="Low",
                    question=f"{count:,} records have dt_start later than rpt_mth. This results in negative mos_bk and indicates data or imputation error." + _acct_info(df, mask)[0],
                    check_id="DO2", variable="dt_start",
                    examples=_examples(df, mask, variable="dt_start"),
                    case_data=_case_sample(df, mask, ["dt_start", "mos_bk"]),
                    stats={"count": count, **_acct_info(df, mask)[1]},
                ))

    # DO3: mos_bk negative
    if _col(df, "mos_bk"):
        mask = df["mos_bk"] < 0
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=f"{count:,} records have negative mos_bk (months on book). This typically results from dt_start > rpt_mth.",
                check_id="DO3", variable="mos_bk",
                stats={"count": int(count)},
            ))

    return findings


# ================================================================
# Origination Variable Checks (LV1-LV3, BA1-BA2, RS1-RS2)
# ================================================================

def _run_origination_checks(df: pd.DataFrame, checks_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []

    # LV1: face_val vs cur_amt — cur_amt should not significantly exceed face_val
    if _col(df, "face_val") and _col(df, "cur_amt"):
        valid = df[df["face_val"].notna() & df["cur_amt"].notna() & (df["face_val"] > 0)]
        if len(valid) > 0:
            over = valid["cur_amt"] > valid["face_val"] * 1.1
            over_count = int(over.sum())
            if over_count > 0:
                rate = over_count / len(valid)
                findings.append(Finding(
                    product=product, parameter="EAD", impact="Low",
                    question=(
                        f"{over_count:,} records ({rate:.1%}) have cur_amt > 110% of face_val. "
                        f"Balance significantly exceeding loan value may indicate capitalized interest, "
                        f"fees, or a data issue."
                        + _acct_info(valid, over)[0]
                    ),
                    check_id="LV1", variable="face_val",
                    case_data=_case_sample(df, over.reindex(df.index, fill_value=False), ["face_val", "cur_amt"]),
                    stats={"count": over_count, "rate": round(rate, 4), **_acct_info(valid, over)[1]},
                ))

    # LV2: face_val should be constant per account (tolerance 0.1%)
    if _col(df, "face_val") and _col(df, "eid"):
        valid = df[df["face_val"].notna()]
        agg = valid.groupby("eid", observed=True)["face_val"].agg(["min", "max"])
        agg["range_pct"] = (agg["max"] - agg["min"]) / agg["max"].replace(0, np.nan)
        changing = agg[agg["range_pct"] > 0.001]
        if len(changing) > 0:
            findings.append(Finding(
                product=product, parameter="EAD", impact="Low",
                question=(
                    f"{len(changing):,} accounts have changing face_val over time. "
                    f"Loan value is typically fixed at origination. Changes may indicate "
                    f"restructuring, top-ups, or data quality issues."
                    f" Affected accounts: {len(changing):,} / {int(_total_accounts(df)):,} ({len(changing)/_total_accounts(df):.1%})."
                ),
                check_id="LV2", variable="face_val",
                case_data=_case_sample(df, df["eid"].isin(changing.index), ["face_val", "cur_amt"]),
                stats={"accounts_with_changes": len(changing), "total_accounts": _total_accounts(df),
                       "account_rate": round(len(changing) / df["eid"].nunique(), 4)},
            ))

    # LV3: face_val trend by rpt_mth
    if _col(df, "face_val") and _col(df, "rpt_mth"):
        valid = df[df["face_val"].notna() & (df["face_val"] > 0)]
        if len(valid) > 0:
            trend = valid.groupby("rpt_mth")["face_val"].agg(["mean", "median", "count"])
            findings.append(Finding(
                product=product, parameter="EAD", impact="Low",
                question="Loan value (face_val) trend data generated (mean, median by rpt_mth).",
                check_id="LV3", variable="face_val",
                reference_only=True,
                stats={"ln_value_trend": {
                    str(k): {"mean": round(float(r["mean"]), 2), "median": round(float(r["median"]), 2),
                             "count": int(r["count"])}
                    for k, r in trend.iterrows()
                }},
            ))

    # BA1: init_amt should be constant per account (tolerance 0.1%)
    if _col(df, "init_amt") and _col(df, "eid"):
        valid = df[df["init_amt"].notna()]
        agg = valid.groupby("eid", observed=True)["init_amt"].agg(["min", "max"])
        agg["range_pct"] = (agg["max"] - agg["min"]) / agg["max"].replace(0, np.nan)
        changing = agg[agg["range_pct"] > 0.001]
        if len(changing) > 0:
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=(
                    f"{len(changing):,} accounts have changing init_amt over time. "
                    f"Booked amount is fixed at origination and should not change. "
                    f"Changes suggest data joins or restructuring issues."
                    f" Affected accounts: {len(changing):,} / {int(_total_accounts(df)):,} ({len(changing)/_total_accounts(df):.1%})."
                ),
                check_id="BA1", variable="init_amt",
                case_data=_case_sample(df, df["eid"].isin(changing.index), ["init_amt", "face_val"]),
                stats={"accounts_with_changes": len(changing), "total_accounts": _total_accounts(df),
                       "account_rate": round(len(changing) / df["eid"].nunique(), 4)},
            ))

    # BA2: init_amt vs face_val consistency
    if _col(df, "init_amt") and _col(df, "face_val"):
        valid = df[df["init_amt"].notna() & df["face_val"].notna() & (df["face_val"] > 0)]
        if len(valid) > 0:
            diff = (valid["init_amt"] - valid["face_val"]).abs()
            rel_diff = diff / valid["face_val"]
            large = int((rel_diff > 0.01).sum())
            if large > 0:
                rate = large / len(valid)
                acct_info = ""
                acct_stats = {}
                if _col(df, "eid"):
                    a = int(valid.loc[rel_diff > 0.01, "eid"].nunique())
                    t = _total_accounts(df)
                    acct_info = f" Affected accounts: {a:,} / {t:,} ({a/t:.1%})."
                    acct_stats = {"affected_accounts": a, "total_accounts": t, "account_rate": round(a/t, 4)}
                findings.append(Finding(
                    product=product, parameter="Data", impact="Low",
                    question=(
                        f"{large:,} records ({rate:.1%}) have >1% difference between "
                        f"init_amt and face_val. These should typically align for non-revolving products."
                        + acct_info
                    ),
                    check_id="BA2", variable="init_amt",
                    case_data=_case_sample(df, (rel_diff > 0.01).reindex(df.index, fill_value=False), ["init_amt", "face_val"]),
                    stats={"count": large, "rate": round(rate, 4), **acct_stats},
                ))

    # RS1: fl_restr trend by rpt_mth
    if _col(df, "fl_restr") and _col(df, "rpt_mth"):
        rate_trend = df.groupby("rpt_mth")["fl_restr"].mean().sort_index()
        overall = float(df["fl_restr"].mean())
        findings.append(Finding(
            product=product, parameter="Score_Alignment", impact="Low",
            question=(
                f"Restructure rate trend generated. Overall rate: {overall:.2%}. "
                f"Sudden spikes may indicate policy changes or economic events."
            ),
            check_id="RS1", variable="fl_restr",
            reference_only=True,
            stats={"overall_rate": round(overall, 4),
                   "trend": {str(k): round(float(v), 4) for k, v in rate_trend.items()}},
        ))

    # RS2: restructured accounts — tot_term/maturity changes should align with fl_restr=1
    if _col(df, "fl_restr") and _col(df, "tot_term") and _col(df, "eid") and _col(df, "rpt_mth"):
        term_diff = df.groupby("eid", observed=True)["tot_term"].diff().abs()
        changed_accts = df.loc[term_diff > 0, "eid"].unique()
        if len(changed_accts) > 0:
            changed_records = df[df["eid"].isin(changed_accts)]
            no_restructure = changed_records.groupby("eid", observed=True)["fl_restr"].max() == 0
            unexplained = int(no_restructure.sum())
            if unexplained > 0:
                findings.append(Finding(
                    product=product, parameter="Score_Alignment", impact="Low",
                    question=(
                        f"{unexplained:,} accounts have tot_term changes but fl_restr is never 1. "
                        f"Term changes without restructure flag suggest data issues or unreported restructuring."
                        f" Affected accounts: {unexplained:,} / {int(_total_accounts(df)):,} ({unexplained/_total_accounts(df):.1%})."
                    ),
                    check_id="RS2", variable="fl_restr",
                    case_data=_case_sample(df, df["eid"].isin(no_restructure[no_restructure].index), ["fl_restr", "tot_term", "dt_end"]),
                    stats={"unexplained_term_changes": unexplained,
                           "total_term_changed_accounts": len(changed_accts),
                           "affected_accounts": unexplained, "total_accounts": _total_accounts(df),
                           "account_rate": round(unexplained / df["eid"].nunique(), 4)},
                ))

    return findings


# ================================================================
# Term Product Checks (TM1-TM9)
# ================================================================

def _run_term(df: pd.DataFrame, tc: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []

    # TM1: dt_end missing
    if tc.get("maturity_missing", {}).get("enabled") and _col(df, "dt_end"):
        rate = df["dt_end"].isna().mean()
        if rate > 0.05:
            findings.append(Finding(
                product=product, parameter=tc["maturity_missing"].get("parameter", ["DF", "ERL"]),
                impact=tc["maturity_missing"].get("impact", "High"),
                question=f"dt_end has {rate:.1%} missing values. Term products require maturity dates for DF cohort and ERL terminal event calculation.",
                check_id="TM1", variable="dt_end",
                stats={"missing_rate": round(rate, 4)},
            ))

    # TM2: maturity < opened
    if tc.get("maturity_before_opened", {}).get("enabled") and _col(df, "dt_end") and _col(df, "dt_start"):
        both_valid = df["dt_end"].notna() & df["dt_start"].notna()
        if not both_valid.any():
            count = 0
            mask = pd.Series(False, index=df.index)
        elif pd.api.types.is_datetime64_any_dtype(df["dt_end"]) and pd.api.types.is_datetime64_any_dtype(df["dt_start"]):
            mask = (df["dt_end"] < df["dt_start"]) & both_valid
            count = int(mask.sum())
        else:
            count = 0
            mask = pd.Series(False, index=df.index)
        if count > 0:
            findings.append(Finding(
                product=product, parameter=tc["maturity_before_opened"].get("parameter", ["DF", "ERL"]),
                impact="Low",
                question=f"{count:,} records have dt_end before dt_start. This is a data error that invalidates Loan Term and rem_term." + _acct_info(df, mask)[0],
                check_id="TM2", variable="dt_end",
                examples=_examples(df, mask, variable="dt_end"),
                case_data=_case_sample(df, mask, ["dt_end", "dt_start", "tot_term"]),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # TM3: maturity passed but open
    if tc.get("maturity_passed_but_open", {}).get("enabled") and _col(df, "dt_end") and _col(df, "rpt_mth") and _col(df, "fl_close"):
        both_dt = df["dt_end"].notna() & df["rpt_mth"].notna()
        if not (both_dt.any() and pd.api.types.is_datetime64_any_dtype(df["dt_end"]) and pd.api.types.is_datetime64_any_dtype(df["rpt_mth"])):
            both_dt = pd.Series(False, index=df.index)
        mask = both_dt & (df["dt_end"] < df["rpt_mth"]) & (df["fl_close"] == 0)
        if _col(df, "grp1"):
            mask = mask & (df["grp1"] == 0)
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter=tc["maturity_passed_but_open"].get("parameter", ["ERL"]),
                impact="Low",
                question=f"{count:,} functional, open accounts have dt_end earlier than rpt_mth. These accounts have passed maturity but are not closed, affecting ERL terminal event." + _acct_info(df, mask)[0],
                check_id="TM3", variable="dt_end",
                examples=_examples(df, mask, variable="dt_end"),
                case_data=_case_sample(df, mask, ["dt_end", "fl_close", "cur_amt", "rem_term"]),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # TM4: maturity temporal consistency
    if tc.get("maturity_temporal_consistency", {}).get("enabled") and _col(df, "dt_end") and _col(df, "eid"):
        mat_per_acct = df[df["dt_end"].notna()].groupby("eid", observed=True)["dt_end"].nunique()
        inconsistent = mat_per_acct[mat_per_acct > 1]
        if _col(df, "fl_restr"):
            restructured = df[df["fl_restr"] == 1]["eid"].unique()
            inconsistent = inconsistent[~inconsistent.index.isin(restructured)]
        count = len(inconsistent)
        if count > 0:
            findings.append(Finding(
                product=product, parameter=tc["maturity_temporal_consistency"].get("parameter", ["DF"]),
                impact="Medium",
                question=f"{count:,} non-restructured accounts have different dt_end values across obs_months. Maturity should be stable for term products. Affected accounts: {count:,} / {int(_total_accounts(df)):,} ({count/_total_accounts(df):.1%}).",
                check_id="TM4", variable="dt_end",
                case_data=_case_sample(df, df["eid"].isin(inconsistent.index), ["dt_end", "tot_term", "fl_restr"]),
                stats={"inconsistent_accounts": int(count), "affected_accounts": int(count),
                       "total_accounts": _total_accounts(df),
                       "account_rate": round(count / df["eid"].nunique(), 4)},
            ))

    # TM5: loan term distribution (12-month buckets) + average
    if _col(df, "tot_term"):
        valid = df["tot_term"].dropna()
        if len(valid) > 0:
            avg_term = float(valid.mean())
            max_term = int(valid.max())
            bins = list(range(0, max_term + 13, 12))
            labels = [f"{b+1}-{b+12}" for b in bins[:-1]]
            labels[0] = "0-12"
            bucketed = pd.cut(valid, bins=bins, labels=labels, right=True, include_lowest=True)
            dist = bucketed.value_counts().sort_index()
            total = len(valid)
            bucket_stats = {
                str(k): {"count": int(v), "pct": round(v / total, 4)}
                for k, v in dist.items() if v > 0
            }
            acct_avg = None
            if _col(df, "eid"):
                acct_avg = float(df.groupby("eid", observed=True)["tot_term"].first().dropna().mean())

            desc_parts = [f"{k}: {v['count']:,} ({v['pct']:.1%})" for k, v in bucket_stats.items()]
            findings.append(Finding(
                product=product, parameter=["DF", "ERL"],
                impact="Medium",
                question=(
                    f"Loan term distribution (months): {', '.join(desc_parts)}. "
                    f"Average term across all records: {avg_term:.1f} months"
                    + (f", average per account: {acct_avg:.1f} months" if acct_avg else "")
                    + "."
                ),
                check_id="TM5", variable="tot_term",
                reference_only=True,
                stats={"distribution": bucket_stats, "avg_term": round(avg_term, 1),
                       "avg_term_per_account": round(acct_avg, 1) if acct_avg else None},
            ))

    # TM6: loan term growth
    if tc.get("loan_term_growth", {}).get("enabled") and _col(df, "tot_term") and _col(df, "eid") and _col(df, "rpt_mth"):
        sorted_df = df[df["tot_term"].notna()]
        prev = sorted_df.groupby("eid", observed=True)["tot_term"].shift(1)
        growing = sorted_df["tot_term"] > prev
        if _col(df, "fl_restr"):
            growing = growing & (sorted_df["fl_restr"] != 1)
        growing_accts = sorted_df.loc[growing, "eid"].nunique()
        total_accts = _total_accounts(df)
        if growing_accts > 0:
            rate = growing_accts / total_accts
            findings.append(Finding(
                product=product, parameter=tc["loan_term_growth"].get("parameter", ["ERL"]),
                impact="High",
                question=f"{growing_accts:,} accounts ({rate:.1%}) have increasing tot_term without restructure flag. This may cause ERL overestimation.",
                check_id="TM6", variable="tot_term",
                stats={"growing_accounts": int(growing_accts), "rate": round(rate, 4)},
            ))

    # TM7: rem_term negative
    if tc.get("remaining_term_negative", {}).get("enabled") and _col(df, "rem_term"):
        mask = df["rem_term"] < 0
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter=tc["remaining_term_negative"].get("parameter", ["DF"]),
                impact="Low",
                question=f"{count:,} records have negative rem_term." + _acct_info(df, mask)[0],
                check_id="TM7", variable="rem_term",
                case_data=_case_sample(df, mask, ["rem_term", "tot_term", "mos_bk", "dt_end"]),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # TM8: mos_bk + rem_term = tot_term (must be exactly equal)
    if tc.get("term_consistency", {}).get("enabled") and _col(df, "mos_bk") and _col(df, "rem_term") and _col(df, "tot_term"):
        valid = df[df["mos_bk"].notna() & df["rem_term"].notna() & df["tot_term"].notna()]
        diff = (valid["mos_bk"] + valid["rem_term"] - valid["tot_term"]).abs()
        inconsistent = int((diff > 0).sum())
        if inconsistent > 0:
            rate = inconsistent / len(valid) if len(valid) > 0 else 0
            acct_info = ""
            acct_stats = {}
            if _col(df, "eid"):
                a = int(valid.loc[diff > 0, "eid"].nunique())
                t = _total_accounts(df)
                acct_info = f" Affected accounts: {a:,} / {t:,} ({a/t:.1%})."
                acct_stats = {"affected_accounts": a, "total_accounts": t, "account_rate": round(a/t, 4)}
            findings.append(Finding(
                product=product, parameter=tc["term_consistency"].get("parameter", ["DF"]),
                impact="Low",
                question=f"{inconsistent:,} records ({rate:.1%}) have mos_bk + rem_term ≠ tot_term. Term variables must be exactly consistent." + acct_info,
                check_id="TM8", variable="rem_term",
                case_data=_case_sample(df, (diff > 0).reindex(df.index, fill_value=False), ["rem_term", "tot_term", "mos_bk"]),
                stats={"inconsistent_count": inconsistent, "rate": round(rate, 4), **acct_stats},
            ))

    # TM9: rem_term vs maturity
    if tc.get("remaining_term_vs_maturity", {}).get("enabled") and _col(df, "rem_term") and _col(df, "dt_end") and _col(df, "rpt_mth"):
        valid = df[df["rem_term"].notna() & df["dt_end"].notna() & df["rpt_mth"].notna()].copy()
        if len(valid) > 0:
            if pd.api.types.is_datetime64_any_dtype(valid["dt_end"]) and pd.api.types.is_datetime64_any_dtype(valid["rpt_mth"]):
                expected = ((valid["dt_end"] - valid["rpt_mth"]).dt.days / 30.44).round()
                diff = (valid["rem_term"] - expected).abs()
                inconsistent = (diff > 2).sum()
                if inconsistent > 0:
                    findings.append(Finding(
                        product=product, parameter=tc["remaining_term_vs_maturity"].get("parameter", ["DF", "ERL"]),
                        impact="Low",
                        question=f"{inconsistent:,} records have rem_term inconsistent with (dt_end - rpt_mth) by more than 2 months." + _acct_info(valid, diff > 2)[0],
                        check_id="TM9", variable="rem_term",
                        case_data=_case_sample(df, (diff > 2).reindex(df.index, fill_value=False), ["rem_term", "dt_end", "mos_bk"]),
                        stats={"inconsistent_count": int(inconsistent), **_acct_info(valid, diff > 2)[1]},
                    ))

    return findings


# ================================================================
# Revolving Product Checks (RV1-RV5)
# ================================================================

def _run_revolving(df: pd.DataFrame, rc: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []

    if not _col(df, "max_lim"):
        findings.append(Finding(
            product=product, parameter=["EAD"], impact="Low",
            question="max_lim column not found in data. Revolving product checks (utilization, EAD) cannot be performed. Check column_mapping in project config.",
            check_id="RV0", variable="max_lim",
        ))
        return findings

    # RV1: missing
    if rc.get("limit_missing", {}).get("enabled"):
        rate = df["max_lim"].isna().mean()
        if rate > 0.01:
            findings.append(Finding(
                product=product, parameter=rc["limit_missing"].get("parameter", ["EAD"]),
                impact=rc["limit_missing"].get("impact", "High"),
                question=f"max_lim has {rate:.1%} missing values. Utilization and EAD cannot be calculated for these records.",
                check_id="RV1", variable="max_lim",
                stats={"missing_rate": round(rate, 4)},
            ))

    # RV2: zero or negative
    if rc.get("limit_zero_or_negative", {}).get("enabled"):
        mask = df["max_lim"] <= 0
        mask = mask & df["max_lim"].notna()
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter=rc["limit_zero_or_negative"].get("parameter", ["EAD"]),
                impact=rc["limit_zero_or_negative"].get("impact", "High"),
                question=f"{count:,} records have max_lim <= 0. This causes division-by-zero in utilization calculation." + _acct_info(df, mask)[0],
                check_id="RV2", variable="max_lim",
                examples=_examples(df, mask, variable="max_lim"),
                case_data=_case_sample(df, mask, ["max_lim", "cur_amt"]),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # RV3: extreme values
    if rc.get("limit_extreme", {}).get("enabled"):
        valid = df["max_lim"].dropna()
        if len(valid) > 0:
            p999 = valid.quantile(0.999)
            extreme = df["max_lim"] > p999
            count = extreme.sum()
            if count > 0:
                max_val = valid.max()
                findings.append(Finding(
                    product=product, parameter=rc["limit_extreme"].get("parameter", ["EAD"]),
                    impact=rc["limit_extreme"].get("impact", "Medium"),
                    question=f"{count:,} records have max_lim above 99.9th percentile ({p999:,.0f}). Max value: {max_val:,.0f}. Check for currency conversion issues.",
                    check_id="RV3", variable="max_lim",
                    stats={"count": int(count), "p999": float(p999), "max": float(max_val)},
                ))

    # RV4: utilization distribution (reference chart)
    if rc.get("utilization_distribution", {}).get("enabled") and _col(df, "cur_amt"):
        valid = df[(df["max_lim"].notna()) & (df["max_lim"] > 0)].copy()
        if len(valid) > 0:
            valid["utilization"] = valid["cur_amt"] / valid["max_lim"]
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")]
            labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                      "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", ">100%"]
            bucketed = pd.cut(valid["utilization"], bins=bins, labels=labels, right=True, include_lowest=True)
            dist = bucketed.value_counts().sort_index()
            total = len(valid)
            bucket_stats = {str(k): {"count": int(v), "pct": round(v / total, 4)} for k, v in dist.items() if v > 0}
            over_1_pct = round((valid["utilization"] > 1).mean(), 4)
            below_0_pct = round((valid["utilization"] <= 0).mean(), 4)
            desc_parts = [f"{k}: {v['count']:,} ({v['pct']:.1%})" for k, v in bucket_stats.items()]
            findings.append(Finding(
                product=product, parameter=["EAD"], impact="Low",
                question=(
                    f"Utilization (cur_amt/max_lim) distribution: {', '.join(desc_parts)}. "
                    f"Utilization >100%: {over_1_pct:.1%}, ≤0%: {below_0_pct:.1%}. "
                    f"Mean: {valid['utilization'].mean():.2%}, Median: {valid['utilization'].median():.2%}. "
                    f"Remind: utilization >1 indicates cur_amt exceeds credit limit — review CCF/EAD impact."
                ),
                check_id="RV4", variable="max_lim",
                reference_only=True,
                stats={"distribution": bucket_stats, "over_1_pct": over_1_pct, "below_0_pct": below_0_pct,
                       "mean_util": round(float(valid["utilization"].mean()), 4),
                       "median_util": round(float(valid["utilization"].median()), 4)},
            ))

    # RV5: limit temporal
    if rc.get("limit_temporal", {}).get("enabled") and _col(df, "eid") and _col(df, "rpt_mth"):
        sorted_df = df[df["max_lim"].notna()].sort_values(["eid", "rpt_mth"])
        prev = sorted_df.groupby("eid", observed=True)["max_lim"].shift(1)
        change = ((sorted_df["max_lim"] - prev) / prev).abs()
        big_change = change > 0.5
        if _col(df, "fl_restr"):
            big_change = big_change & (sorted_df["fl_restr"] != 1)
        count = big_change.sum()
        limit_trend = sorted_df.groupby("rpt_mth")["max_lim"].agg(["mean", "median"]).to_dict("index")
        limit_trend_stats = {str(k): {"mean": round(float(v["mean"]), 2), "median": round(float(v["median"]), 2)} for k, v in limit_trend.items()}
        if count > 0:
            findings.append(Finding(
                product=product, parameter=rc["limit_temporal"].get("parameter", ["EAD"]),
                impact=rc["limit_temporal"].get("impact", "Low"),
                question=f"{count:,} records have max_lim changing by >50% month-over-month (excluding restructured accounts).",
                check_id="RV5", variable="max_lim",
                case_data=_case_sample(df, big_change.reindex(df.index, fill_value=False), ["max_lim", "cur_amt", "fl_restr"]),
                stats={"count": int(count), "limit_trend": limit_trend_stats},
            ))

    return findings


def _lgd_workout_checks(df: pd.DataFrame, findings: list, product: str) -> None:
    """LG12 (actual rcv_amt) and LG13 (imputed from cur_amt Δ)."""
    WINDOW = 36
    need = ["new_evt", "eid", "rpt_mth"]
    missing_need = [c for c in need if not _col(df, c)]
    if missing_need:
        print(f"  [LG12/LG13] Skipped: missing columns {missing_need}")
        return
    if not (_col(df, "rcv_amt") or _col(df, "cur_amt")):
        print(f"  [LG12/LG13] Skipped: missing both rcv_amt and cur_amt")
        return

    dft_events = df[df["new_evt"] == 1][["eid", "rpt_mth"]].copy()
    if len(dft_events) == 0:
        print(f"  [LG12/LG13] Skipped: no new_evt=1 events found")
        return

    bal_col = "fwd_amt" if _col(df, "fwd_amt") else "cur_amt"
    dft_events = dft_events.rename(columns={"rpt_mth": "dft_month"})
    dft_events["dft_bal"] = df.loc[dft_events.index, bal_col].values
    dft_events = dft_events[dft_events["dft_bal"].notna() & (dft_events["dft_bal"] > 0)]
    if len(dft_events) == 0:
        return

    dft_idx = dft_events.set_index("eid")
    dft_idx = dft_idx[~dft_idx.index.duplicated(keep="first")]

    merge_cols = ["eid", "rpt_mth"]
    has_interest = _col(df, "ann_rate")
    has_recovery = _col(df, "rcv_amt")
    has_balance = _col(df, "cur_amt")
    if has_interest:
        merge_cols.append("ann_rate")
    if has_recovery:
        merge_cols.append("rcv_amt")
    if has_balance:
        merge_cols.append("cur_amt")
    merged = dft_events.merge(df[merge_cols], on="eid")
    merged = merged[merged["rpt_mth"] >= merged["dft_month"]]
    merged["months_since"] = ((merged["rpt_mth"] - merged["dft_month"]).dt.days / 30.44).round().astype(int)
    merged = merged[merged["months_since"] <= WINDOW]
    merged = merged.sort_values(["eid", "rpt_mth"])

    if has_interest:
        monthly_rate = merged["ann_rate"].clip(lower=0) / 12
        discount = (1 + monthly_rate) ** merged["months_since"]
    else:
        discount = 1

    def _summarize(rates):
        if len(rates) == 0:
            return {}
        return {
            "mean": round(float(rates.mean()), 4),
            "median": round(float(rates.median()), 4),
            "count": len(rates),
            "p10": round(float(rates.quantile(0.1)), 4),
            "p25": round(float(rates.quantile(0.25)), 4),
            "p75": round(float(rates.quantile(0.75)), 4),
            "p90": round(float(rates.quantile(0.9)), 4),
        }

    def _build_stats(recovery_rate, dft_months_s, label):
        cohort_rates = {}
        for acct, rate in recovery_rate.items():
            m = str(dft_months_s.get(acct, "Unknown"))
            cohort_rates.setdefault(m, []).append(float(rate))
        cohort_summary = {}
        for m in sorted(cohort_rates.keys()):
            vals = cohort_rates[m]
            cohort_summary[m] = {"mean": round(sum(vals) / len(vals), 4), "count": len(vals)}
        question = (
            f"LGD workout rcv_amt rate{label} for {len(recovery_rate)} defaulted accounts "
            f"({WINDOW}-month window, mean {recovery_rate.mean():.2%})."
        )
        return {
            "overall": _summarize(recovery_rate),
            "cohort": cohort_summary,
        }, question

    acct_bal = dft_idx["dft_bal"]
    dft_months_s = dft_idx["dft_month"].to_dict()

    # LG12: actual rcv_amt
    if has_recovery:
        merged["pv_recovery_actual"] = merged["rcv_amt"].fillna(0) / discount
        acct_pv = merged.groupby("eid", observed=True)["pv_recovery_actual"].sum()
        rr = (acct_pv / acct_bal).dropna()
        rr = rr[rr.between(-0.5, 2.0)]
        if len(rr) > 0:
            stats, question = _build_stats(rr, dft_months_s, "")
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=question, check_id="LG12", variable="rcv_amt",
                reference_only=True, stats=stats,
            ))

    # LG13: imputed rcv_amt (cur_amt Δ when rcv_amt is 0/NA)
    if has_balance:
        prev_bal = merged.groupby("eid", observed=True)["cur_amt"].shift(1)
        bal_diff = (prev_bal - merged["cur_amt"]).clip(lower=0)
        if has_recovery:
            raw = merged["rcv_amt"]
            use_actual = raw.notna() & (raw > 0)
            imputed = bal_diff.copy()
            imputed.loc[use_actual] = raw[use_actual]
        else:
            imputed = bal_diff
        merged["pv_recovery_imputed"] = imputed.fillna(0) / discount
        acct_pv = merged.groupby("eid", observed=True)["pv_recovery_imputed"].sum()
        rr = (acct_pv / acct_bal).dropna()
        rr = rr[rr.between(-0.5, 2.0)]
        if len(rr) > 0:
            stats, question = _build_stats(rr, dft_months_s, " (imputed from cur_amt Δ)")
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=question, check_id="LG13", variable="rcv_amt",
                reference_only=True, stats=stats,
            ))

    # LG14: recovery pattern distribution over 36-month window
    if has_recovery:
        rcv = merged["rcv_amt"]
        merged["_rcv_pos"] = (rcv > 0).astype(int)
        merged["_rcv_zero"] = (rcv.isna() | (rcv == 0)).astype(int)
        merged["_rcv_neg"] = (rcv < 0).astype(int)

        acct_pat = merged.groupby("eid", observed=True).agg(
            n_months=("months_since", "count"),
            n_pos=("_rcv_pos", "sum"),
            n_zero=("_rcv_zero", "sum"),
            n_neg=("_rcv_neg", "sum"),
            dft_month=("dft_month", "first"),
        )
        acct_pat["pct_pos"] = acct_pat["n_pos"] / acct_pat["n_months"]
        acct_pat["pct_zero"] = acct_pat["n_zero"] / acct_pat["n_months"]
        acct_pat["pct_neg"] = acct_pat["n_neg"] / acct_pat["n_months"]

        cohort_pat = acct_pat.groupby("dft_month").agg(
            pct_pos=("pct_pos", "mean"),
            pct_zero=("pct_zero", "mean"),
            pct_neg=("pct_neg", "mean"),
            count=("n_months", "count"),
        )

        lg14_stats: dict = {"cohort": {}}
        for month, row in cohort_pat.iterrows():
            m = str(month)[:7]
            lg14_stats["cohort"][m] = {
                "pct_pos": round(float(row["pct_pos"]), 4),
                "pct_zero": round(float(row["pct_zero"]), 4),
                "pct_neg": round(float(row["pct_neg"]), 4),
                "count": int(row["count"]),
            }

        findings.append(Finding(
            product=product, parameter="LGD", impact="Low",
            question=(
                f"Recovery pattern distribution for {len(acct_pat)} defaulted accounts "
                f"over {WINDOW}-month window. Average proportion of months with "
                f"positive recovery: {acct_pat['pct_pos'].mean():.1%}, "
                f"zero/NA: {acct_pat['pct_zero'].mean():.1%}, "
                f"negative: {acct_pat['pct_neg'].mean():.1%}."
            ),
            check_id="LG14", variable="rcv_amt",
            reference_only=True, stats=lg14_stats,
        ))
        merged.drop(columns=["_rcv_pos", "_rcv_zero", "_rcv_neg"], inplace=True)
