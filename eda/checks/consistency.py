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


_EXAMPLE_BASE_COLS = ["acct_id", "obs_month"]


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
        _total_accounts_cache["_total"] = int(df["acct_id"].nunique()) if "acct_id" in df.columns else 0
    return _total_accounts_cache["_total"]


def _acct_info(df: pd.DataFrame, mask) -> tuple[str, dict]:
    """Return (text_suffix, stats_dict) for account proportion."""
    if "acct_id" not in df.columns:
        return "", {}
    t = _total_accounts(df)
    a = int(df.loc[mask, "acct_id"].nunique())
    r = a / t if t > 0 else 0
    return (
        f" Affected accounts: {a:,} / {t:,} ({r:.1%}).",
        {"affected_accounts": a, "total_accounts": t, "account_rate": round(r, 4)},
    )


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

    # DF1: dpd missing rate per obs_month
    if _col(df, "dpd") and _col(df, "obs_month"):
        monthly_miss = df["dpd"].isna().groupby(df["obs_month"]).mean()
        overall = df["dpd"].isna().mean()
        if overall > 0.05:
            findings.append(Finding(
                product=product,
                parameter="Data",
                impact="Medium",
                question=(
                    f"DPD has {overall:.1%} missing values overall. "
                    f"Missing DPD affects default identification (dpd>=90 rule) "
                    f"and may cause underestimation of PD."
                ),
                check_id="DF1",
                variable="dpd",
                stats={"overall_rate": round(overall, 4),
                       "per_month": {str(k): round(v, 4) for k, v in monthly_miss.items()}},
            ))
        diffs = monthly_miss.diff().abs()
        jump_months = diffs[diffs > 0.10]
        for month, jump in jump_months.items():
            findings.append(Finding(
                product=product, parameter="Data", impact="Medium",
                question=f"DPD missing rate jumped by {jump:.1%} at {month}. Possible score model or data source change.",
                check_id="DF1", variable="dpd",
                stats={"month": str(month), "jump": round(float(jump), 4)},
            ))

    # DF2: dpd missing with chargeoff
    if c.get("dpd_missing_with_chargeoff", {}).get("enabled") and _col(df, "dpd") and _col(df, "ind_CO"):
        cfg = c["dpd_missing_with_chargeoff"]
        mask = df["dpd"].isna() & (df["ind_CO"] == 1)
        count = mask.sum()
        if count > 0:
            q_text = (
                f"Found {count:,} records where DPD is missing but ind_CO=1 (charge-off). "
                f"These accounts' default status depends entirely on charge-off indicator. "
                f"Is the DPD missing by design or a data gap?"
            )
            extra_stats = {}
            if _col(df, "acct_id"):
                affected_accts = int(df.loc[mask, "acct_id"].nunique())
                total_accts = _total_accounts(df)
                acct_pct = affected_accts / total_accts
                q_text += f" Affected accounts: {affected_accts:,} / {total_accts:,} ({acct_pct:.1%})."
                extra_stats = {"affected_accounts": affected_accts, "total_accounts": total_accts, "account_rate": round(acct_pct, 4)}
            findings.append(Finding(
                product=product,
                parameter=cfg.get("parameter", "Data"),
                impact=cfg.get("impact", "High"),
                question=q_text,
                check_id="DF2", variable="dpd",
                examples=_examples(df, mask, variable="dpd"),
                stats={"count": int(count), "rate": round(count / len(df), 4), **extra_stats},
            ))

    # DF6: consecutive defaults without cure
    if c.get("consecutive_defaults", {}).get("enabled") and _col(df, "new_to_dft") and _col(df, "ind_dft") and _col(df, "acct_id") and _col(df, "obs_month"):
        cfg = c["consecutive_defaults"]
        ntd = df[df["new_to_dft"] == 1][["acct_id", "obs_month"]].copy()
        multi = ntd.groupby("acct_id", observed=True).size()
        multi_accts = multi[multi > 1]
        if len(multi_accts) > 0:
            sample_accts = multi_accts.head(20).index.tolist()
            sample = df[df["acct_id"].isin(sample_accts)][["acct_id", "obs_month", "ind_dft", "new_to_dft", "dpd"] if _col(df, "dpd") else ["acct_id", "obs_month", "ind_dft", "new_to_dft"]]
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
                check_id="DF6", variable="new_to_dft",
                examples=sample.sort_values(["acct_id", "obs_month"]).head(40),
                stats={"affected_accounts": int(len(multi_accts)), "total_accounts": total_accts, "account_rate": round(acct_pct, 4)},
            ))

    # DF7: 12-month forward default rate trend (perf_lvl2=1 / perf_lvl1=0)
    if _col(df, "obs_month"):
        func_df = df[df["perf_lvl1"] == 0] if _col(df, "perf_lvl1") else df
        dft_rate = None
        if _col(func_df, "perf_lvl2"):
            dft_rate = func_df.groupby("obs_month")["perf_lvl2"].mean().sort_index()
        ntd_rate = None
        if _col(func_df, "new_to_dft"):
            ntd_rate = func_df.groupby("obs_month")["new_to_dft"].mean().sort_index()
        if dft_rate is not None or ntd_rate is not None:
            findings.append(Finding(
                product=product, parameter="PD", impact="Low",
                question="Default rate trend data generated. 12-month forward default rate (perf_lvl2=1) on functional accounts (perf_lvl1=0).",
                check_id="DF7", variable="perf_lvl2",
                reference_only=True,
                stats={
                    "default_rate": {str(k): round(float(v), 6) for k, v in dft_rate.items()} if dft_rate is not None else {},
                    "new_to_dft_rate": {str(k): round(float(v), 6) for k, v in ntd_rate.items()} if ntd_rate is not None else {},
                },
            ))

    # DF9: cpd vs dpd consistency — cpd should approximate ceil(dpd/30)
    if _col(df, "cpd") and _col(df, "dpd"):
        valid = df[df["cpd"].notna() & df["dpd"].notna()]
        if len(valid) > 0:
            expected_cpd = np.ceil(valid["dpd"] / 30).astype(int)
            mismatch = (valid["cpd"] != expected_cpd) & (valid["dpd"] > 0)
            mismatch_count = int(mismatch.sum())
            if mismatch_count > 0:
                rate = mismatch_count / len(valid)
                findings.append(Finding(
                    product=product, parameter="Data", impact="Low",
                    question=(
                        f"cpd vs dpd inconsistency: {mismatch_count:,} records ({rate:.1%}) "
                        f"where cpd ≠ ceil(dpd/30). This may indicate different cycle definitions "
                        f"or independent derivation of cpd."
                    + (_acct_info(valid, mismatch)[0] if _col(valid, "acct_id") else "")
                    ),
                    check_id="DF9", variable="cpd",
                    examples=valid.loc[mismatch].head(20) if mismatch.any() else None,
                    stats={"mismatch_count": mismatch_count, "rate": round(rate, 4), **(_acct_info(valid, mismatch)[1] if _col(valid, "acct_id") else {})},
                ))

    return findings


# ================================================================
# Terminal Events (TE1-TE8)
# ================================================================

def _run_terminal_events(df: pd.DataFrame, checks_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []
    c = checks_cfg.get("consistency", {})

    # TE1: ind_closed cummax violation (1 → 0)
    if c.get("closed_temporal", {}).get("enabled") and _col(df, "ind_closed") and _col(df, "acct_id") and _col(df, "obs_month"):
        cfg = c["closed_temporal"]
        violations, examples = _check_cummax_violation(df, "acct_id", "ind_closed")
        if violations > 0:
            t = _total_accounts(df)
            findings.append(Finding(
                product=product, parameter=cfg.get("parameter", ["ERL", "LGD"]),
                impact=cfg.get("impact", "High"),
                question=(
                    f"ind_closed violates cummax logic: {violations:,} accounts have ind_closed "
                    f"reverting from 1 to 0 (re-opening after closure). "
                    f"Affected accounts: {violations:,} / {t:,} ({violations/t:.1%}). "
                    f"This breaks terminal event logic for ERL and may cause LGD miscalculation."
                ),
                check_id="TE1", variable="ind_closed",
                examples=examples,
                stats={"violation_accounts": violations, "total_accounts": t, "account_rate": round(violations / t, 4)},
            ))

    # TE2: ind_CO cummax violation
    if c.get("chargeoff_temporal", {}).get("enabled") and _col(df, "ind_CO") and _col(df, "acct_id") and _col(df, "obs_month"):
        cfg = c["chargeoff_temporal"]
        violations, examples = _check_cummax_violation(df, "acct_id", "ind_CO")
        if violations > 0:
            findings.append(Finding(
                product=product, parameter=cfg.get("parameter", ["LGD"]),
                impact=cfg.get("impact", "High"),
                question=(
                    f"ind_CO violates cummax logic: {violations:,} accounts have ind_CO "
                    f"reverting from 1 to 0. Once charged off, the indicator should remain 1. "
                    f"Affected accounts: {violations:,} / {_total_accounts(df):,} ({violations/_total_accounts(df):.1%}). "
                    f"This affects LGD calculation."
                ),
                check_id="TE2", variable="ind_CO",
                examples=examples,
                stats={"violation_accounts": violations, "total_accounts": _total_accounts(df), "account_rate": round(violations / _total_accounts(df), 4)},
            ))

    # TE3: closed but balance > 0
    if c.get("closed_vs_balance", {}).get("enabled") and _col(df, "ind_closed") and _col(df, "balance"):
        cfg = c["closed_vs_balance"]
        mask = (df["ind_closed"] == 1) & (df["balance"] > 0)
        count = mask.sum()
        if count > 0:
            rate = count / (df["ind_closed"] == 1).sum() if (df["ind_closed"] == 1).sum() > 0 else 0
            findings.append(Finding(
                product=product, parameter=cfg.get("parameter", ["LGD"]),
                impact=cfg.get("impact", "Medium"),
                question=(
                    f"{count:,} records ({rate:.1%} of closed accounts) have ind_closed=1 "
                    f"but balance > 0. Closed accounts should typically have zero balance."
                    + _acct_info(df, mask)[0]
                ),
                check_id="TE3", variable="ind_closed",
                examples=_examples(df, mask, variable="ind_closed"),
                stats={"count": int(count), "rate": round(rate, 4), **_acct_info(df, mask)[1]},
            ))

    # TE4: chargeoff after closure
    if c.get("post_close_chargeoff", {}).get("enabled") and _col(df, "ind_closed") and _col(df, "ind_CO") and _col(df, "acct_id") and _col(df, "obs_month"):
        cfg = c["post_close_chargeoff"]
        sorted_df = df
        first_closed = sorted_df[sorted_df["ind_closed"] == 1].groupby("acct_id", observed=True)["obs_month"].first()
        first_co_after = sorted_df[sorted_df["ind_CO"] == 1].groupby("acct_id", observed=True)["obs_month"].first()
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
                    check_id="TE4", variable="ind_CO",
                    stats={"violation_accounts": int(violation_count), "total_accounts": _total_accounts(df), "account_rate": round(int(violation_count)/df["acct_id"].nunique(), 4)},
                ))

    # TE5: ind_CO=1 but ind_closed=0
    if c.get("closed_vs_chargeoff", {}).get("enabled") and _col(df, "ind_CO") and _col(df, "ind_closed"):
        cfg = c["closed_vs_chargeoff"]
        mask = (df["ind_CO"] == 1) & (df["ind_closed"] == 0)
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter=cfg.get("parameter", ["LGD", "ERL"]),
                impact=cfg.get("impact", "Medium"),
                question=(
                    f"{count:,} records have ind_CO=1 but ind_closed=0. "
                    f"Charge-off typically implies account closure. "
                    f"Is this intentional for new-to-default charge-off accounts?"
                    + _acct_info(df, mask)[0]
                ),
                check_id="TE5", variable="ind_CO",
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # TE6: chargeoff before dt_opened
    if _col(df, "obs_month") and _col(df, "dt_opened") and _col(df, "ind_CO"):
        both_valid = df["obs_month"].notna() & df["dt_opened"].notna()
        if both_valid.any() and pd.api.types.is_datetime64_any_dtype(df["obs_month"]) and pd.api.types.is_datetime64_any_dtype(df["dt_opened"]):
            mask = (df["obs_month"] < df["dt_opened"]) & both_valid & (df["ind_CO"] == 1)
            count = int(mask.sum())
            if count > 0:
                findings.append(Finding(
                    product=product, parameter="Data", impact="Low",
                    question=(
                        f"{count:,} records have ind_CO=1 before dt_opened. "
                        f"Chargeoff should not occur before account opening date."
                    + _acct_info(df, mask)[0]
                    ),
                    check_id="TE6", variable="ind_CO",
                    examples=_examples(df, mask, variable="ind_CO"),
                    stats={"count": count, **_acct_info(df, mask)[1]},
                ))

    # TE7 & TE8: trend data for ind_closed and ind_CO
    for col, check_id in [("ind_closed", "TE7"), ("ind_CO", "TE8")]:
        if _col(df, col) and _col(df, "obs_month"):
            trend = df.groupby("obs_month")[col].agg(["sum", "count"])
            trend.columns = ["value_1", "n_records"]
            trend["value_0"] = trend["n_records"] - trend["value_1"]
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=f"{col} trend data generated by obs_month.",
                check_id=check_id, variable=col,
                reference_only=True,
                stats={"trend": {str(k): {"value_0": int(r["value_0"]), "value_1": int(r["value_1"]),
                                           "n_records": int(r["n_records"])}
                                 for k, r in trend.iterrows()}},
            ))

    return findings


def _check_cummax_violation(df: pd.DataFrame, id_col: str, indicator_col: str, time_col: str = "obs_month") -> tuple:
    """Return (violation_account_count, examples_df) in one pass."""
    shifted = df.groupby(id_col, observed=True)[indicator_col].shift(1)
    violations = (shifted == 1) & (df[indicator_col] == 0)
    n_accts = int(df.loc[violations, id_col].nunique())
    examples = pd.DataFrame()
    if n_accts > 0:
        accts = df.loc[violations, id_col].unique()[:5]
        cols = [id_col, time_col, indicator_col]
        for c in ("dpd", "balance"):
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
    default_col = sa_cfg.get("default_column", "perf_lvl2")
    n_segments = sa_cfg.get("n_segments", 10)
    mono_tol = sa_cfg.get("monotonicity_tolerance", 1)
    max_high_rate = sa_cfg.get("max_high_score_default_rate", 0.02)
    impact = sa_cfg.get("impact", "High")
    parameter = sa_cfg.get("parameter", ["Score_Alignment", "PD", "SICR"])

    if not _col(df, default_col):
        return findings

    # Filter to functional accounts (perf_lvl1=0) for score-default alignment
    if _col(df, "perf_lvl1"):
        sa_df = df[df["perf_lvl1"] == 0]
    else:
        sa_df = df

    for score_col in score_cols:
        if not _col(sa_df, score_col):
            continue

        valid = sa_df[[score_col, default_col]].dropna()
        if len(valid) < 100:
            continue

        # SA1: Monotonicity
        valid["segment"] = pd.qcut(valid[score_col], n_segments, labels=False, duplicates="drop")
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

        # SA3: Score segment × obs_month default rate (for heatmap)
        if _col(sa_df, "obs_month"):
            valid_with_month = sa_df[[score_col, default_col, "obs_month"]].dropna()
            if len(valid_with_month) > 100:
                valid_with_month["segment"] = pd.qcut(valid_with_month[score_col], min(n_segments, 5), labels=False, duplicates="drop")
                heatmap = valid_with_month.groupby(["segment", "obs_month"])[default_col].mean().unstack(fill_value=0)
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

        # SA5/SA6: Missing rate trend on functional accounts (perf_lvl1=0)
        if _col(df, "obs_month"):
            func_df = df[df["perf_lvl1"] == 0] if _col(df, "perf_lvl1") else df
            monthly_miss = func_df[score_col].isna().groupby(func_df["obs_month"]).mean()
            overall_miss = func_df[score_col].isna().mean()
            is_issue = overall_miss > 0.05
            findings.append(Finding(
                product=product, parameter=parameter, impact="Low",
                question=(
                    f"Score `{score_col}` has {overall_miss:.1%} missing values overall. "
                    + ("Check if score file merge is complete." if is_issue else "Missing rate within acceptable range.")
                ),
                check_id="SA6" if score_col == "score_bhv" else "SA5",
                variable=score_col,
                reference_only=not is_issue,
                stats={"overall_rate": round(overall_miss, 4),
                       "per_month": {str(k): round(float(v), 4) for k, v in monthly_miss.items()}},
            ))

        # SA7: Distribution drift
        if _col(df, "obs_month"):
            monthly_mean = df.groupby("obs_month")[score_col].mean()
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
        if _col(sa_df, "obs_month"):
            ts_valid = sa_df[[score_col, default_col, "obs_month"]].dropna()
            if len(ts_valid) > 200:
                n_seg = min(n_segments, 5)
                ts_valid["segment"] = pd.qcut(
                    ts_valid[score_col], n_seg, labels=False, duplicates="drop"
                )
                seg_bounds = ts_valid.groupby("segment")[score_col].agg(["min", "max"])
                seg_labels = {
                    int(seg): f"{int(row['min'])}-{int(row['max'])}"
                    for seg, row in seg_bounds.iterrows()
                }
                pivot = ts_valid.groupby(["obs_month", "segment"])[default_col].mean().unstack()

                if len(pivot) >= 3 and len(pivot.columns) >= 2:
                    findings.append(Finding(
                        product=product, parameter=parameter, impact="Low",
                        question=(
                            f"Score `{score_col}` segment-level default rate trend by obs_month. "
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

    # LG1: balance negative
    if _col(df, "balance"):
        mask = df["balance"] < 0
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"Balance has {count:,} negative records ({count/len(df):.2%}). Negative balance may indicate data errors or special account types.",
                check_id="LG1", variable="balance",
                examples=_examples(df, mask, variable="balance"),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # LG2: balance=0 but dpd>0
    if _col(df, "balance") and _col(df, "dpd"):
        mask = (df["balance"] == 0) & (df["dpd"] > 0)
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"{count:,} records have balance=0 but DPD>0. Zero-balance delinquent accounts may affect default definition.",
                check_id="LG2", variable="balance",
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # LG3: balance missing at default
    if _col(df, "balance") and _col(df, "ind_dft"):
        mask = (df["ind_dft"] == 1) & df["balance"].isna()
        count = mask.sum()
        total_dft = (df["ind_dft"] == 1).sum()
        if count > 0 and total_dft > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"{count:,} defaulted records ({count/total_dft:.1%} of defaults) have missing balance. Balance at default is required for LGD calculation." + _acct_info(df, mask)[0],
                check_id="LG3", variable="balance",
                stats={"count": int(count), "rate_of_defaults": round(count / total_dft, 4), **_acct_info(df, mask)[1]},
            ))

    # LG4: recovery negative trend
    if _col(df, "recovery") and _col(df, "obs_month"):
        neg_mask = df["recovery"] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            monthly = df[neg_mask].groupby("obs_month").size()
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"Recovery field has {neg_count:,} negative values. Negative recovery may indicate reversed payments or data errors." + _acct_info(df, neg_mask)[0],
                check_id="LG4", variable="recovery",
                stats={"total_negative": int(neg_count),
                       "per_month": {str(k): int(v) for k, v in monthly.items()}, **_acct_info(df, neg_mask)[1]},
            ))

    # LG6: recovery trend
    if _col(df, "recovery") and _col(df, "obs_month"):
        trend = df.groupby("obs_month")["recovery"].agg(["mean", "sum", "count"])
        trend.columns = ["recovery_avg", "recovery_sum", "n_records"]
        findings.append(Finding(
            product=product, parameter="LGD", impact="Medium",
            question="Recovery trend data generated (average and sum by obs_month).",
            check_id="LG6", variable="recovery",
            reference_only=True,
            stats={"recovery_trend": {str(k): {"avg": round(float(r["recovery_avg"]), 2),
                                                "sum": round(float(r["recovery_sum"]), 2),
                                                "n": int(r["n_records"])}
                                       for k, r in trend.iterrows()}},
        ))

    # LG7: non-default, non-CO accounts with recovery > 0
    if _col(df, "recovery") and _col(df, "ind_dft") and _col(df, "ind_CO"):
        mask = (df["ind_dft"] == 0) & (df["ind_CO"] == 0) & (df["recovery"].notna()) & (df["recovery"] > 0)
        count = int(mask.sum())
        if count > 0:
            n_accts = df.loc[mask, "acct_id"].nunique() if _col(df, "acct_id") else count
            t = _total_accounts(df)
            r = n_accts / t if t > 0 else 0
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=(
                    f"{count:,} records have recovery>0 but ind_dft=0 and ind_CO=0. "
                    f"Non-default, non-chargeoff accounts should not have recovery amounts."
                    f" Affected accounts: {n_accts:,} / {t:,} ({r:.1%})."
                ),
                check_id="LG7", variable="recovery",
                stats={"records": int(count), "affected_accounts": n_accts, "total_accounts": t, "account_rate": round(r, 4)},
            ))

    # LG8: interest_rate = 0
    if _col(df, "interest_rate"):
        zero_rate = (df["interest_rate"] == 0).sum()
        total_valid = df["interest_rate"].notna().sum()
        if zero_rate > 0 and total_valid > 0:
            ratio = zero_rate / total_valid
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"Interest rate is 0 for {zero_rate:,} records ({ratio:.1%}). Is 0% interest rate valid or does it represent missing data?",
                check_id="LG8", variable="interest_rate",
                stats={"zero_count": int(zero_rate), "rate": round(ratio, 4)},
            ))

    # LG9: interest_rate extreme
    if _col(df, "interest_rate"):
        extreme = (df["interest_rate"] > 0.5) | (df["interest_rate"] < 0)
        count = extreme.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=f"Interest rate has {count:,} records with extreme values (>50% or <0). Check data quality." + _acct_info(df, extreme)[0],
                check_id="LG9", variable="interest_rate",
                examples=_examples(df, extreme, variable="interest_rate"),
                stats={"count": int(count), **_acct_info(df, extreme)[1]},
            ))

    # LG10: balance trend
    if _col(df, "balance") and _col(df, "obs_month"):
        trend = df.groupby("obs_month")["balance"].agg(["sum", "mean", "median"])
        findings.append(Finding(
            product=product, parameter="Data", impact="Low",
            question="Balance trend data generated (sum, mean, median by obs_month).",
            check_id="LG10", variable="balance",
            reference_only=True,
            stats={"balance_trend": {str(k): {"sum": round(float(r["sum"]), 2), "mean": round(float(r["mean"]), 2), "median": round(float(r["median"]), 2)} for k, r in trend.iterrows()}},
        ))

    # LG11: removed (Keep=N in check inventory)

    # LG12 & LG13: LGD workout recovery rate
    _lgd_workout_checks(df, findings, product)


    # NB2: next_dft_bal negative
    if _col(df, "next_dft_bal"):
        neg = df["next_dft_bal"] < 0
        neg_count = int(neg.sum())
        if neg_count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=(
                    f"next_dft_bal has {neg_count:,} negative values. "
                    f"Negative balance at default is unusual and will distort LGD calculation."
                ),
                check_id="NB2", variable="next_dft_bal",
                examples=_examples(df, neg, variable="next_dft_bal"),
                stats={"negative_count": neg_count, **_acct_info(df, neg)[1]},
            ))

    # NB3: next_dft_bal vs balance consistency at default
    if _col(df, "next_dft_bal") and _col(df, "balance") and _col(df, "new_to_dft"):
        ntd_mask = df["new_to_dft"] == 1
        valid = df.loc[ntd_mask].dropna(subset=["next_dft_bal", "balance"])
        if len(valid) > 0:
            diff = (valid["next_dft_bal"] - valid["balance"]).abs()
            rel_diff = diff / valid["balance"].abs().clip(lower=1)
            large_diff = int((rel_diff > 0.1).sum())
            if large_diff > 0:
                rate = large_diff / len(valid)
                findings.append(Finding(
                    product=product, parameter="LGD", impact="Low",
                    question=(
                        f"{large_diff:,} new-to-default records ({rate:.1%}) have >10% difference "
                        f"between next_dft_bal and balance. These should align at the point of default. "
                        f"Large discrepancies suggest a timing or derivation issue."
                    + _acct_info(valid, rel_diff > 0.1)[0]
                    ),
                    check_id="NB3", variable="next_dft_bal",
                    stats={"large_diff_count": large_diff, "rate": round(rate, 4),
                           "median_rel_diff": round(float(rel_diff.median()), 4), **_acct_info(valid, rel_diff > 0.1)[1]},
                ))

    # NB4: next_dft_bal = 0 for defaulted accounts
    if _col(df, "next_dft_bal") and _col(df, "ind_dft"):
        zero_dft = (df["next_dft_bal"] == 0) & (df["ind_dft"] == 1)
        zero_count = int(zero_dft.sum())
        if zero_count > 0:
            findings.append(Finding(
                product=product, parameter="LGD", impact="Medium",
                question=(
                    f"{zero_count:,} defaulted records have next_dft_bal=0. "
                    f"Zero balance at default means LGD is undefined for these accounts. "
                    f"Check if this is data error or legitimate (e.g., fully recovered before default flag)."
                ),
                check_id="NB4", variable="next_dft_bal",
                stats={"zero_count": zero_count, **_acct_info(df, zero_dft)[1]},
            ))

    # MD1: mths_to_dft missing for defaulted accounts
    if _col(df, "mths_to_dft") and _col(df, "ind_dft"):
        dft_mask = df["ind_dft"] == 1
        total_dft = int(dft_mask.sum())
        if total_dft > 0:
            missing = int(df.loc[dft_mask, "mths_to_dft"].isna().sum())
            if missing > 0:
                rate = missing / total_dft
                findings.append(Finding(
                    product=product, parameter="PD", impact="Low",
                    question=(
                        f"mths_to_dft is missing for {missing:,} defaulted records ({rate:.1%}). "
                        f"This variable is needed for PD term structure estimation. "
                        f"Missing values reduce the usable default sample."
                    + _acct_info(df, df["ind_dft"].eq(1) & df["mths_to_dft"].isna())[0]
                    ),
                    check_id="MD1", variable="mths_to_dft",
                    stats={"missing_count": missing, "total_defaults": total_dft, "rate": round(rate, 4),
                           **_acct_info(df, df["ind_dft"].eq(1) & df["mths_to_dft"].isna())[1]},
                ))

    # MD2: mths_to_dft negative
    if _col(df, "mths_to_dft"):
        neg = df["mths_to_dft"] < 0
        neg_count = int(neg.sum())
        if neg_count > 0:
            findings.append(Finding(
                product=product, parameter="PD", impact="Low",
                question=(
                    f"mths_to_dft has {neg_count:,} negative values. "
                    f"Negative months-to-default is logically impossible and indicates "
                    f"a derivation error (possibly dt_next_dft < obs_month)."
                ),
                check_id="MD2", variable="mths_to_dft",
                examples=_examples(df, neg, variable="mths_to_dft"),
                stats={"negative_count": neg_count, **_acct_info(df, neg)[1]},
            ))

    # MD3: mths_to_dft distribution trend
    if _col(df, "mths_to_dft") and _col(df, "obs_month"):
        valid = df[df["mths_to_dft"].notna() & (df["mths_to_dft"] >= 0)]
        if len(valid) > 100:
            trend = valid.groupby("obs_month")["mths_to_dft"].agg(["mean", "median", "count"])
            findings.append(Finding(
                product=product, parameter="PD", impact="Low",
                question="mths_to_dft distribution trend generated (mean, median by obs_month).",
                check_id="MD3", variable="mths_to_dft",
                reference_only=True,
                stats={"mths_to_dft_trend": {
                    str(k): {"mean": round(float(r["mean"]), 2), "median": round(float(r["median"]), 2),
                             "count": int(r["count"])}
                    for k, r in trend.iterrows()
                }},
            ))

    # MD4: mths_to_dft vs mob consistency
    if _col(df, "mths_to_dft") and _col(df, "mob") and _col(df, "new_to_dft"):
        ntd = df[(df["new_to_dft"] == 1) & df["mths_to_dft"].notna() & df["mob"].notna()]
        if len(ntd) > 0:
            diff = (ntd["mths_to_dft"] - ntd["mob"]).abs()
            large = int((diff > 2).sum())
            if large > 0:
                rate = large / len(ntd)
                findings.append(Finding(
                    product=product, parameter="PD", impact="Low",
                    question=(
                        f"{large:,} new-to-default records ({rate:.1%}) have |mths_to_dft - mob| > 2. "
                        f"At the point of first default, mths_to_dft should approximate mob. "
                        f"Large gaps suggest different reference dates in derivation."
                    + _acct_info(ntd, diff > 2)[0]
                    ),
                    check_id="MD4", variable="mths_to_dft",
                    stats={"inconsistent_count": large, "rate": round(rate, 4),
                           **_acct_info(ntd, diff > 2)[1]},
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
    if _col(df, "perf_lvl1") and _col(df, "obs_month"):
        dist = df.groupby("obs_month")["perf_lvl1"].value_counts(normalize=True).unstack(fill_value=0)
        findings.append(Finding(
            product=product, parameter="Data", impact="Low",
            question="perf_lvl1 distribution trend generated.",
            check_id="PL2", variable="perf_lvl1",
            reference_only=True,
            stats={"distribution": {str(k): {str(c): round(float(v), 4) for c, v in row.items()} for k, row in dist.iterrows()}},
        ))

    # PL3: functional ratio
    if _col(df, "perf_lvl1"):
        func_rate = (df["perf_lvl1"] == 0).mean()
        if func_rate < 0.5:
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=f"Only {func_rate:.1%} of records have perf_lvl1=0 (functional). Expected >50%. Large exclusion ratio may reduce sample size for downstream models.",
                check_id="PL3", variable="perf_lvl1",
                stats={"functional_rate": round(func_rate, 4)},
            ))

    return findings


# ================================================================
# dt_opened Checks (DO1-DO3)
# ================================================================

def _run_dt_opened(df: pd.DataFrame, checks_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []

    # DO1: dt_opened missing
    if _col(df, "dt_opened"):
        rate = df["dt_opened"].isna().mean()
        if rate > 0.01:
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=f"dt_opened has {rate:.1%} missing values. Missing open dates cause perf_lvl1=4 exclusion and affect mob/Loan Term calculation.",
                check_id="DO1", variable="dt_opened",
                stats={"missing_rate": round(rate, 4)},
            ))

    # DO2: dt_opened > obs_month
    if _col(df, "dt_opened") and _col(df, "obs_month"):
        both_valid = df["dt_opened"].notna() & df["obs_month"].notna()
        if both_valid.any() and pd.api.types.is_datetime64_any_dtype(df["obs_month"]) and pd.api.types.is_datetime64_any_dtype(df["dt_opened"]):
            mask = (df["dt_opened"] > df["obs_month"]) & both_valid
            count = int(mask.sum())
            if count > 0:
                findings.append(Finding(
                    product=product, parameter="Data", impact="Low",
                    question=f"{count:,} records have dt_opened later than obs_month. This results in negative mob and indicates data or imputation error." + _acct_info(df, mask)[0],
                    check_id="DO2", variable="dt_opened",
                    examples=_examples(df, mask, variable="dt_opened"),
                    stats={"count": count, **_acct_info(df, mask)[1]},
                ))

    # DO3: mob negative
    if _col(df, "mob"):
        mask = df["mob"] < 0
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=f"{count:,} records have negative mob (months on book). This typically results from dt_opened > obs_month.",
                check_id="DO3", variable="mob",
                stats={"count": int(count)},
            ))

    return findings


# ================================================================
# Origination Variable Checks (LV1-LV3, BA1-BA2, RS1-RS2)
# ================================================================

def _run_origination_checks(df: pd.DataFrame, checks_cfg: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []

    # LV1: ln_value vs balance — balance should not significantly exceed ln_value
    if _col(df, "ln_value") and _col(df, "balance"):
        valid = df[df["ln_value"].notna() & df["balance"].notna() & (df["ln_value"] > 0)]
        if len(valid) > 0:
            over = valid["balance"] > valid["ln_value"] * 1.1
            over_count = int(over.sum())
            if over_count > 0:
                rate = over_count / len(valid)
                findings.append(Finding(
                    product=product, parameter="EAD", impact="Low",
                    question=(
                        f"{over_count:,} records ({rate:.1%}) have balance > 110% of ln_value. "
                        f"Balance significantly exceeding loan value may indicate capitalized interest, "
                        f"fees, or a data issue."
                        + _acct_info(valid, over)[0]
                    ),
                    check_id="LV1", variable="ln_value",
                    stats={"count": over_count, "rate": round(rate, 4), **_acct_info(valid, over)[1]},
                ))

    # LV2: ln_value should be constant per account
    if _col(df, "ln_value") and _col(df, "acct_id"):
        per_acct = df[df["ln_value"].notna()].groupby("acct_id", observed=True)["ln_value"].nunique()
        changing = per_acct[per_acct > 1]
        if len(changing) > 0:
            findings.append(Finding(
                product=product, parameter="EAD", impact="Low",
                question=(
                    f"{len(changing):,} accounts have changing ln_value over time. "
                    f"Loan value is typically fixed at origination. Changes may indicate "
                    f"restructuring, top-ups, or data quality issues."
                    f" Affected accounts: {len(changing):,} / {int(_total_accounts(df)):,} ({len(changing)/_total_accounts(df):.1%})."
                ),
                check_id="LV2", variable="ln_value",
                stats={"accounts_with_changes": len(changing), "total_accounts": _total_accounts(df),
                       "account_rate": round(len(changing) / df["acct_id"].nunique(), 4)},
            ))

    # LV3: ln_value trend by obs_month
    if _col(df, "ln_value") and _col(df, "obs_month"):
        valid = df[df["ln_value"].notna() & (df["ln_value"] > 0)]
        if len(valid) > 0:
            trend = valid.groupby("obs_month")["ln_value"].agg(["mean", "median", "count"])
            findings.append(Finding(
                product=product, parameter="EAD", impact="Low",
                question="Loan value (ln_value) trend data generated (mean, median by obs_month).",
                check_id="LV3", variable="ln_value",
                reference_only=True,
                stats={"ln_value_trend": {
                    str(k): {"mean": round(float(r["mean"]), 2), "median": round(float(r["median"]), 2),
                             "count": int(r["count"])}
                    for k, r in trend.iterrows()
                }},
            ))

    # BA1: booked_amt should be constant per account
    if _col(df, "booked_amt") and _col(df, "acct_id"):
        per_acct = df[df["booked_amt"].notna()].groupby("acct_id", observed=True)["booked_amt"].nunique()
        changing = per_acct[per_acct > 1]
        if len(changing) > 0:
            findings.append(Finding(
                product=product, parameter="Data", impact="Low",
                question=(
                    f"{len(changing):,} accounts have changing booked_amt over time. "
                    f"Booked amount is fixed at origination and should not change. "
                    f"Changes suggest data joins or restructuring issues."
                    f" Affected accounts: {len(changing):,} / {int(_total_accounts(df)):,} ({len(changing)/_total_accounts(df):.1%})."
                ),
                check_id="BA1", variable="booked_amt",
                stats={"accounts_with_changes": len(changing), "total_accounts": _total_accounts(df),
                       "account_rate": round(len(changing) / df["acct_id"].nunique(), 4)},
            ))

    # BA2: booked_amt vs ln_value consistency
    if _col(df, "booked_amt") and _col(df, "ln_value"):
        valid = df[df["booked_amt"].notna() & df["ln_value"].notna() & (df["ln_value"] > 0)]
        if len(valid) > 0:
            diff = (valid["booked_amt"] - valid["ln_value"]).abs()
            rel_diff = diff / valid["ln_value"]
            large = int((rel_diff > 0.01).sum())
            if large > 0:
                rate = large / len(valid)
                acct_info = ""
                acct_stats = {}
                if _col(df, "acct_id"):
                    a = int(valid.loc[rel_diff > 0.01, "acct_id"].nunique())
                    t = _total_accounts(df)
                    acct_info = f" Affected accounts: {a:,} / {t:,} ({a/t:.1%})."
                    acct_stats = {"affected_accounts": a, "total_accounts": t, "account_rate": round(a/t, 4)}
                findings.append(Finding(
                    product=product, parameter="Data", impact="Low",
                    question=(
                        f"{large:,} records ({rate:.1%}) have >1% difference between "
                        f"booked_amt and ln_value. These should typically align for non-revolving products."
                        + acct_info
                    ),
                    check_id="BA2", variable="booked_amt",
                    stats={"count": large, "rate": round(rate, 4), **acct_stats},
                ))

    # RS1: ind_restructure trend by obs_month
    if _col(df, "ind_restructure") and _col(df, "obs_month"):
        rate_trend = df.groupby("obs_month")["ind_restructure"].mean().sort_index()
        overall = float(df["ind_restructure"].mean())
        findings.append(Finding(
            product=product, parameter="Score_Alignment", impact="Low",
            question=(
                f"Restructure rate trend generated. Overall rate: {overall:.2%}. "
                f"Sudden spikes may indicate policy changes or economic events."
            ),
            check_id="RS1", variable="ind_restructure",
            reference_only=True,
            stats={"overall_rate": round(overall, 4),
                   "trend": {str(k): round(float(v), 4) for k, v in rate_trend.items()}},
        ))

    # RS2: restructured accounts — ln_term/maturity changes should align with ind_restructure=1
    if _col(df, "ind_restructure") and _col(df, "ln_term") and _col(df, "acct_id") and _col(df, "obs_month"):
        term_diff = df.groupby("acct_id", observed=True)["ln_term"].diff().abs()
        changed_accts = df.loc[term_diff > 0, "acct_id"].unique()
        if len(changed_accts) > 0:
            changed_records = df[df["acct_id"].isin(changed_accts)]
            no_restructure = changed_records.groupby("acct_id", observed=True)["ind_restructure"].max() == 0
            unexplained = int(no_restructure.sum())
            if unexplained > 0:
                findings.append(Finding(
                    product=product, parameter="Score_Alignment", impact="Low",
                    question=(
                        f"{unexplained:,} accounts have ln_term changes but ind_restructure is never 1. "
                        f"Term changes without restructure flag suggest data issues or unreported restructuring."
                        f" Affected accounts: {unexplained:,} / {int(_total_accounts(df)):,} ({unexplained/_total_accounts(df):.1%})."
                    ),
                    check_id="RS2", variable="ind_restructure",
                    stats={"unexplained_term_changes": unexplained,
                           "total_term_changed_accounts": len(changed_accts),
                           "affected_accounts": unexplained, "total_accounts": _total_accounts(df),
                           "account_rate": round(unexplained / df["acct_id"].nunique(), 4)},
                ))

    return findings


# ================================================================
# Term Product Checks (TM1-TM9)
# ================================================================

def _run_term(df: pd.DataFrame, tc: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []

    # TM1: maturity_dt missing
    if tc.get("maturity_missing", {}).get("enabled") and _col(df, "maturity_dt"):
        rate = df["maturity_dt"].isna().mean()
        if rate > 0.05:
            findings.append(Finding(
                product=product, parameter=tc["maturity_missing"].get("parameter", ["DF", "ERL"]),
                impact=tc["maturity_missing"].get("impact", "High"),
                question=f"maturity_dt has {rate:.1%} missing values. Term products require maturity dates for DF cohort and ERL terminal event calculation.",
                check_id="TM1", variable="maturity_dt",
                stats={"missing_rate": round(rate, 4)},
            ))

    # TM2: maturity < opened
    if tc.get("maturity_before_opened", {}).get("enabled") and _col(df, "maturity_dt") and _col(df, "dt_opened"):
        both_valid = df["maturity_dt"].notna() & df["dt_opened"].notna()
        if not both_valid.any():
            count = 0
            mask = pd.Series(False, index=df.index)
        elif pd.api.types.is_datetime64_any_dtype(df["maturity_dt"]) and pd.api.types.is_datetime64_any_dtype(df["dt_opened"]):
            mask = (df["maturity_dt"] < df["dt_opened"]) & both_valid
            count = int(mask.sum())
        else:
            count = 0
            mask = pd.Series(False, index=df.index)
        if count > 0:
            findings.append(Finding(
                product=product, parameter=tc["maturity_before_opened"].get("parameter", ["DF", "ERL"]),
                impact="Low",
                question=f"{count:,} records have maturity_dt before dt_opened. This is a data error that invalidates Loan Term and remaining_term." + _acct_info(df, mask)[0],
                check_id="TM2", variable="maturity_dt",
                examples=_examples(df, mask, variable="maturity_dt"),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # TM3: maturity passed but open
    if tc.get("maturity_passed_but_open", {}).get("enabled") and _col(df, "maturity_dt") and _col(df, "obs_month") and _col(df, "ind_closed"):
        both_dt = df["maturity_dt"].notna() & df["obs_month"].notna()
        if not (both_dt.any() and pd.api.types.is_datetime64_any_dtype(df["maturity_dt"]) and pd.api.types.is_datetime64_any_dtype(df["obs_month"])):
            both_dt = pd.Series(False, index=df.index)
        mask = both_dt & (df["maturity_dt"] < df["obs_month"]) & (df["ind_closed"] == 0)
        if _col(df, "perf_lvl1"):
            mask = mask & (df["perf_lvl1"] == 0)
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter=tc["maturity_passed_but_open"].get("parameter", ["ERL"]),
                impact="Low",
                question=f"{count:,} functional, open accounts have maturity_dt earlier than obs_month. These accounts have passed maturity but are not closed, affecting ERL terminal event." + _acct_info(df, mask)[0],
                check_id="TM3", variable="maturity_dt",
                examples=_examples(df, mask, variable="maturity_dt"),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # TM4: maturity temporal consistency
    if tc.get("maturity_temporal_consistency", {}).get("enabled") and _col(df, "maturity_dt") and _col(df, "acct_id"):
        mat_per_acct = df[df["maturity_dt"].notna()].groupby("acct_id", observed=True)["maturity_dt"].nunique()
        inconsistent = mat_per_acct[mat_per_acct > 1]
        if _col(df, "ind_restructure"):
            restructured = df[df["ind_restructure"] == 1]["acct_id"].unique()
            inconsistent = inconsistent[~inconsistent.index.isin(restructured)]
        count = len(inconsistent)
        if count > 0:
            findings.append(Finding(
                product=product, parameter=tc["maturity_temporal_consistency"].get("parameter", ["DF"]),
                impact="Medium",
                question=f"{count:,} non-restructured accounts have different maturity_dt values across obs_months. Maturity should be stable for term products. Affected accounts: {count:,} / {int(_total_accounts(df)):,} ({count/_total_accounts(df):.1%}).",
                check_id="TM4", variable="maturity_dt",
                stats={"inconsistent_accounts": int(count), "affected_accounts": int(count),
                       "total_accounts": _total_accounts(df),
                       "account_rate": round(count / df["acct_id"].nunique(), 4)},
            ))

    # TM5: loan term distribution (12-month buckets) + average
    if _col(df, "ln_term"):
        valid = df["ln_term"].dropna()
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
            if _col(df, "acct_id"):
                acct_avg = float(df.groupby("acct_id", observed=True)["ln_term"].first().dropna().mean())

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
                check_id="TM5", variable="ln_term",
                reference_only=True,
                stats={"distribution": bucket_stats, "avg_term": round(avg_term, 1),
                       "avg_term_per_account": round(acct_avg, 1) if acct_avg else None},
            ))

    # TM6: loan term growth
    if tc.get("loan_term_growth", {}).get("enabled") and _col(df, "ln_term") and _col(df, "acct_id") and _col(df, "obs_month"):
        sorted_df = df[df["ln_term"].notna()]
        prev = sorted_df.groupby("acct_id", observed=True)["ln_term"].shift(1)
        growing = sorted_df["ln_term"] > prev
        if _col(df, "ind_restructure"):
            growing = growing & (sorted_df["ind_restructure"] != 1)
        growing_accts = sorted_df.loc[growing, "acct_id"].nunique()
        total_accts = _total_accounts(df)
        if growing_accts > 0:
            rate = growing_accts / total_accts
            findings.append(Finding(
                product=product, parameter=tc["loan_term_growth"].get("parameter", ["ERL"]),
                impact="High",
                question=f"{growing_accts:,} accounts ({rate:.1%}) have increasing ln_term without restructure flag. This may cause ERL overestimation.",
                check_id="TM6", variable="ln_term",
                stats={"growing_accounts": int(growing_accts), "rate": round(rate, 4)},
            ))

    # TM7: remaining_term negative
    if tc.get("remaining_term_negative", {}).get("enabled") and _col(df, "remaining_term"):
        mask = df["remaining_term"] < 0
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter=tc["remaining_term_negative"].get("parameter", ["DF"]),
                impact="Low",
                question=f"{count:,} records have negative remaining_term." + _acct_info(df, mask)[0],
                check_id="TM7", variable="remaining_term",
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # TM8: mob + remaining_term = ln_term (must be exactly equal)
    if tc.get("term_consistency", {}).get("enabled") and _col(df, "mob") and _col(df, "remaining_term") and _col(df, "ln_term"):
        valid = df[df["mob"].notna() & df["remaining_term"].notna() & df["ln_term"].notna()]
        diff = (valid["mob"] + valid["remaining_term"] - valid["ln_term"]).abs()
        inconsistent = int((diff > 0).sum())
        if inconsistent > 0:
            rate = inconsistent / len(valid) if len(valid) > 0 else 0
            acct_info = ""
            acct_stats = {}
            if _col(df, "acct_id"):
                a = int(valid.loc[diff > 0, "acct_id"].nunique())
                t = _total_accounts(df)
                acct_info = f" Affected accounts: {a:,} / {t:,} ({a/t:.1%})."
                acct_stats = {"affected_accounts": a, "total_accounts": t, "account_rate": round(a/t, 4)}
            findings.append(Finding(
                product=product, parameter=tc["term_consistency"].get("parameter", ["DF"]),
                impact="Low",
                question=f"{inconsistent:,} records ({rate:.1%}) have mob + remaining_term ≠ ln_term. Term variables must be exactly consistent." + acct_info,
                check_id="TM8", variable="remaining_term",
                stats={"inconsistent_count": inconsistent, "rate": round(rate, 4), **acct_stats},
            ))

    # TM9: remaining_term vs maturity
    if tc.get("remaining_term_vs_maturity", {}).get("enabled") and _col(df, "remaining_term") and _col(df, "maturity_dt") and _col(df, "obs_month"):
        valid = df[df["remaining_term"].notna() & df["maturity_dt"].notna() & df["obs_month"].notna()].copy()
        if len(valid) > 0:
            if pd.api.types.is_datetime64_any_dtype(valid["maturity_dt"]) and pd.api.types.is_datetime64_any_dtype(valid["obs_month"]):
                expected = ((valid["maturity_dt"] - valid["obs_month"]).dt.days / 30.44).round()
                diff = (valid["remaining_term"] - expected).abs()
                inconsistent = (diff > 2).sum()
                if inconsistent > 0:
                    findings.append(Finding(
                        product=product, parameter=tc["remaining_term_vs_maturity"].get("parameter", ["DF", "ERL"]),
                        impact="Low",
                        question=f"{inconsistent:,} records have remaining_term inconsistent with (maturity_dt - obs_month) by more than 2 months." + _acct_info(valid, diff > 2)[0],
                        check_id="TM9", variable="remaining_term",
                        stats={"inconsistent_count": int(inconsistent), **_acct_info(valid, diff > 2)[1]},
                    ))

    return findings


# ================================================================
# Revolving Product Checks (RV1-RV5)
# ================================================================

def _run_revolving(df: pd.DataFrame, rc: dict, product: str) -> list[Finding]:
    findings: list[Finding] = []

    if not _col(df, "credit_limit"):
        findings.append(Finding(
            product=product, parameter=["EAD"], impact="Low",
            question="credit_limit column not found in data. Revolving product checks (utilization, EAD) cannot be performed. Check column_mapping in project config.",
            check_id="RV0", variable="credit_limit",
        ))
        return findings

    # RV1: missing
    if rc.get("limit_missing", {}).get("enabled"):
        rate = df["credit_limit"].isna().mean()
        if rate > 0.01:
            findings.append(Finding(
                product=product, parameter=rc["limit_missing"].get("parameter", ["EAD"]),
                impact=rc["limit_missing"].get("impact", "High"),
                question=f"credit_limit has {rate:.1%} missing values. Utilization and EAD cannot be calculated for these records.",
                check_id="RV1", variable="credit_limit",
                stats={"missing_rate": round(rate, 4)},
            ))

    # RV2: zero or negative
    if rc.get("limit_zero_or_negative", {}).get("enabled"):
        mask = df["credit_limit"] <= 0
        mask = mask & df["credit_limit"].notna()
        count = mask.sum()
        if count > 0:
            findings.append(Finding(
                product=product, parameter=rc["limit_zero_or_negative"].get("parameter", ["EAD"]),
                impact=rc["limit_zero_or_negative"].get("impact", "High"),
                question=f"{count:,} records have credit_limit <= 0. This causes division-by-zero in utilization calculation." + _acct_info(df, mask)[0],
                check_id="RV2", variable="credit_limit",
                examples=_examples(df, mask, variable="credit_limit"),
                stats={"count": int(count), **_acct_info(df, mask)[1]},
            ))

    # RV3: extreme values
    if rc.get("limit_extreme", {}).get("enabled"):
        valid = df["credit_limit"].dropna()
        if len(valid) > 0:
            p999 = valid.quantile(0.999)
            extreme = df["credit_limit"] > p999
            count = extreme.sum()
            if count > 0:
                max_val = valid.max()
                findings.append(Finding(
                    product=product, parameter=rc["limit_extreme"].get("parameter", ["EAD"]),
                    impact=rc["limit_extreme"].get("impact", "Medium"),
                    question=f"{count:,} records have credit_limit above 99.9th percentile ({p999:,.0f}). Max value: {max_val:,.0f}. Check for currency conversion issues.",
                    check_id="RV3", variable="credit_limit",
                    stats={"count": int(count), "p999": float(p999), "max": float(max_val)},
                ))

    # RV4: utilization distribution (reference chart)
    if rc.get("utilization_distribution", {}).get("enabled") and _col(df, "balance"):
        valid = df[(df["credit_limit"].notna()) & (df["credit_limit"] > 0)].copy()
        if len(valid) > 0:
            valid["utilization"] = valid["balance"] / valid["credit_limit"]
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
                    f"Utilization (balance/credit_limit) distribution: {', '.join(desc_parts)}. "
                    f"Utilization >100%: {over_1_pct:.1%}, ≤0%: {below_0_pct:.1%}. "
                    f"Mean: {valid['utilization'].mean():.2%}, Median: {valid['utilization'].median():.2%}. "
                    f"Remind: utilization >1 indicates balance exceeds credit limit — review CCF/EAD impact."
                ),
                check_id="RV4", variable="credit_limit",
                reference_only=True,
                stats={"distribution": bucket_stats, "over_1_pct": over_1_pct, "below_0_pct": below_0_pct,
                       "mean_util": round(float(valid["utilization"].mean()), 4),
                       "median_util": round(float(valid["utilization"].median()), 4)},
            ))

    # RV5: limit temporal
    if rc.get("limit_temporal", {}).get("enabled") and _col(df, "acct_id") and _col(df, "obs_month"):
        sorted_df = df[df["credit_limit"].notna()].sort_values(["acct_id", "obs_month"])
        prev = sorted_df.groupby("acct_id", observed=True)["credit_limit"].shift(1)
        change = ((sorted_df["credit_limit"] - prev) / prev).abs()
        big_change = change > 0.5
        if _col(df, "ind_restructure"):
            big_change = big_change & (sorted_df["ind_restructure"] != 1)
        count = big_change.sum()
        limit_trend = sorted_df.groupby("obs_month")["credit_limit"].agg(["mean", "median"]).to_dict("index")
        limit_trend_stats = {str(k): {"mean": round(float(v["mean"]), 2), "median": round(float(v["median"]), 2)} for k, v in limit_trend.items()}
        if count > 0:
            findings.append(Finding(
                product=product, parameter=rc["limit_temporal"].get("parameter", ["EAD"]),
                impact=rc["limit_temporal"].get("impact", "Low"),
                question=f"{count:,} records have credit_limit changing by >50% month-over-month (excluding restructured accounts).",
                check_id="RV5", variable="credit_limit",
                stats={"count": int(count), "limit_trend": limit_trend_stats},
            ))

    return findings


def _lgd_workout_checks(df: pd.DataFrame, findings: list, product: str) -> None:
    """LG12 (actual recovery) and LG13 (imputed from balance Δ)."""
    WINDOW = 36
    need = ["new_to_dft", "interest_rate", "acct_id", "obs_month"]
    if not all(_col(df, c) for c in need):
        return
    if not (_col(df, "recovery") or _col(df, "balance")):
        return

    dft_events = df[df["new_to_dft"] == 1][["acct_id", "obs_month"]].copy()
    if len(dft_events) == 0:
        return

    bal_col = "next_dft_bal" if _col(df, "next_dft_bal") else "balance"
    dft_events = dft_events.rename(columns={"obs_month": "dft_month"})
    dft_events["dft_bal"] = df.loc[dft_events.index, bal_col].values
    dft_events = dft_events[dft_events["dft_bal"].notna() & (dft_events["dft_bal"] > 0)]
    if len(dft_events) == 0:
        return

    dft_idx = dft_events.set_index("acct_id")
    dft_idx = dft_idx[~dft_idx.index.duplicated(keep="first")]

    merge_cols = ["acct_id", "obs_month", "interest_rate"]
    has_recovery = _col(df, "recovery")
    has_balance = _col(df, "balance")
    if has_recovery:
        merge_cols.append("recovery")
    if has_balance:
        merge_cols.append("balance")
    merged = dft_events.merge(df[merge_cols], on="acct_id")
    merged = merged[merged["obs_month"] >= merged["dft_month"]]
    merged["months_since"] = ((merged["obs_month"] - merged["dft_month"]).dt.days / 30.44).round().astype(int)
    merged = merged[merged["months_since"] <= WINDOW]
    merged = merged.sort_values(["acct_id", "obs_month"])

    monthly_rate = merged["interest_rate"].clip(lower=0) / 12
    discount = (1 + monthly_rate) ** merged["months_since"]

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
            f"LGD workout recovery rate{label} for {len(recovery_rate)} defaulted accounts "
            f"({WINDOW}-month window, mean {recovery_rate.mean():.2%})."
        )
        return {
            "overall": _summarize(recovery_rate),
            "cohort": cohort_summary,
        }, question

    acct_bal = dft_idx["dft_bal"]
    dft_months_s = dft_idx["dft_month"].to_dict()

    # LG12: actual recovery
    if has_recovery:
        merged["pv_recovery_actual"] = merged["recovery"].fillna(0) / discount
        acct_pv = merged.groupby("acct_id", observed=True)["pv_recovery_actual"].sum()
        rr = (acct_pv / acct_bal).dropna()
        rr = rr[rr.between(-0.5, 2.0)]
        if len(rr) > 0:
            stats, question = _build_stats(rr, dft_months_s, "")
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=question, check_id="LG12", variable="recovery",
                reference_only=True, stats=stats,
            ))

    # LG13: imputed recovery (balance Δ when recovery is 0/NA)
    if has_balance:
        prev_bal = merged.groupby("acct_id", observed=True)["balance"].shift(1)
        bal_diff = (prev_bal - merged["balance"]).clip(lower=0)
        if has_recovery:
            raw = merged["recovery"]
            use_actual = raw.notna() & (raw > 0)
            imputed = bal_diff.copy()
            imputed.loc[use_actual] = raw[use_actual]
        else:
            imputed = bal_diff
        merged["pv_recovery_imputed"] = imputed.fillna(0) / discount
        acct_pv = merged.groupby("acct_id", observed=True)["pv_recovery_imputed"].sum()
        rr = (acct_pv / acct_bal).dropna()
        rr = rr[rr.between(-0.5, 2.0)]
        if len(rr) > 0:
            stats, question = _build_stats(rr, dft_months_s, " (imputed from balance Δ)")
            findings.append(Finding(
                product=product, parameter="LGD", impact="Low",
                question=question, check_id="LG13", variable="recovery",
                reference_only=True, stats=stats,
            ))
