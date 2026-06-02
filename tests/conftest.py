from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Build a synthetic DataFrame with known issues for testing."""
    np.random.seed(42)
    n_accounts = 15
    n_months = 20
    acct_ids = [f"ACCT_{i:04d}" for i in range(n_accounts)]
    months = pd.date_range("2022-01-01", periods=n_months, freq="MS")

    rows = []
    for acct_id in acct_ids:
        dt_opened = months[0] - pd.DateOffset(months=np.random.randint(1, 24))
        for month in months:
            rows.append({
                "acct_id": acct_id,
                "obs_month": month,
                "dt_opened": dt_opened,
            })

    df = pd.DataFrame(rows)

    df["dpd"] = np.random.choice([0, 0, 0, 0, 0, 30, 60, 90, 120], size=len(df))
    df["cpd"] = np.where(df["dpd"] >= 90, np.random.choice([4, 5, 6], size=len(df)), 0)
    df["ind_CO"] = 0
    df["ind_closed"] = 0
    df["ind_dft"] = np.where((df["dpd"] >= 90) | (df["cpd"] >= 4) | (df["ind_CO"] == 1), 1, 0)
    df["balance"] = np.random.uniform(1000, 50000, size=len(df)).round(2)
    df["recovery"] = np.where(df["ind_dft"] == 1, np.random.uniform(0, 5000, size=len(df)), 0).round(2)
    df["score_orig"] = np.random.uniform(300, 850, size=len(df)).round(0)
    df["score_bhv"] = np.random.uniform(300, 850, size=len(df)).round(0)
    df["interest_rate"] = np.random.uniform(0.05, 0.30, size=len(df)).round(4)
    df["perf_lvl1"] = 0
    df.loc[df["ind_dft"] == 1, "perf_lvl1"] = 1
    df.loc[df["ind_closed"] == 1, "perf_lvl1"] = 2
    df["perf_lvl2"] = df["perf_lvl1"]
    df["mob"] = ((df["obs_month"] - df["dt_opened"]).dt.days / 30.44).round().astype(int)
    df["new_to_dft"] = 0
    df["lag_ind_dft"] = 0

    # next_dft_bal: should be populated for defaults, NaN for non-defaults
    df["next_dft_bal"] = np.nan
    df.loc[df["ind_dft"] == 1, "next_dft_bal"] = df.loc[df["ind_dft"] == 1, "balance"] * np.random.uniform(0.8, 1.2, size=int((df["ind_dft"] == 1).sum()))

    # mths_to_dft: months until default, populated for defaults
    df["mths_to_dft"] = np.nan
    df.loc[df["ind_dft"] == 1, "mths_to_dft"] = df.loc[df["ind_dft"] == 1, "mob"] + np.random.randint(-1, 2, size=int((df["ind_dft"] == 1).sum()))

    # --- Inject known issues ---

    # 1) Right-censored account: ACCT_0000 disappears at month 10, not defaulted/closed
    acct0_mask = (df["acct_id"] == "ACCT_0000") & (df["obs_month"] > months[9])
    df = df[~acct0_mask].copy()
    acct0_remaining = df["acct_id"] == "ACCT_0000"
    df.loc[acct0_remaining, "ind_dft"] = 0
    df.loc[acct0_remaining, "ind_CO"] = 0
    df.loc[acct0_remaining, "ind_closed"] = 0
    df.loc[acct0_remaining, "dpd"] = 0

    # 2) ind_closed cummax violation: ACCT_0001 goes 1 → 0
    acct1_mask = df["acct_id"] == "ACCT_0001"
    acct1_idx = df.loc[acct1_mask].index
    if len(acct1_idx) > 10:
        df.loc[acct1_idx[8], "ind_closed"] = 1
        df.loc[acct1_idx[9], "ind_closed"] = 0

    # 3) DPD missing with chargeoff: ACCT_0002
    acct2_mask = df["acct_id"] == "ACCT_0002"
    acct2_idx = df.loc[acct2_mask].index
    if len(acct2_idx) > 5:
        df.loc[acct2_idx[3], "dpd"] = np.nan
        df.loc[acct2_idx[3], "ind_CO"] = 1
        df.loc[acct2_idx[4], "dpd"] = np.nan
        df.loc[acct2_idx[4], "ind_CO"] = 1

    # 4) Negative balance: ACCT_0003
    acct3_mask = df["acct_id"] == "ACCT_0003"
    acct3_idx = df.loc[acct3_mask].index
    if len(acct3_idx) > 3:
        df.loc[acct3_idx[2], "balance"] = -500.0
        df.loc[acct3_idx[3], "balance"] = -200.0

    # 5) Missing score_bhv for 15% of records
    missing_idx = df.sample(frac=0.15, random_state=42).index
    df.loc[missing_idx, "score_bhv"] = np.nan

    # 6) Score monotonicity violation: make high scores have high default rate
    high_score = df["score_orig"] > 750
    df.loc[high_score & (np.random.random(len(df)) < 0.3), "ind_dft"] = 1

    # 7) Single-record account: ACCT_0014 only 1 record
    acct14_mask = df["acct_id"] == "ACCT_0014"
    acct14_idx = df.loc[acct14_mask].index
    if len(acct14_idx) > 1:
        df = df.drop(acct14_idx[1:]).copy()

    # 8) Interest rate = 0 for some records
    zero_ir_idx = df.sample(n=10, random_state=99).index
    df.loc[zero_ir_idx, "interest_rate"] = 0.0

    # 9) NB2: negative next_dft_bal for ACCT_0004
    acct4_mask = df["acct_id"] == "ACCT_0004"
    acct4_idx = df.loc[acct4_mask].index
    if len(acct4_idx) > 5:
        df.loc[acct4_idx[4], "next_dft_bal"] = -1000.0
        df.loc[acct4_idx[4], "ind_dft"] = 1

    # 10) NB1b: ACCT_0005 never defaults but has next_dft_bal populated
    acct5_all = df["acct_id"] == "ACCT_0005"
    df.loc[acct5_all, "ind_dft"] = 0
    df.loc[acct5_all, "dpd"] = 0
    df.loc[acct5_all, "cpd"] = 0
    df.loc[acct5_all, "ind_CO"] = 0
    df.loc[acct5_all, "next_dft_bal"] = np.nan
    acct5_idx = df.loc[acct5_all].index
    if len(acct5_idx) > 3:
        df.loc[acct5_idx[0:3], "next_dft_bal"] = 5000.0

    # 11) MD2: negative mths_to_dft for ACCT_0006
    acct6_mask = df["acct_id"] == "ACCT_0006"
    acct6_idx = df.loc[acct6_mask].index
    if len(acct6_idx) > 5:
        df.loc[acct6_idx[5], "mths_to_dft"] = -3
        df.loc[acct6_idx[5], "ind_dft"] = 1

    # 12) MD4: mths_to_dft vs mob large gap for ACCT_0007
    acct7_mask = df["acct_id"] == "ACCT_0007"
    acct7_idx = df.loc[acct7_mask].index
    if len(acct7_idx) > 6:
        df.loc[acct7_idx[6], "new_to_dft"] = 1
        df.loc[acct7_idx[6], "ind_dft"] = 1
        df.loc[acct7_idx[6], "mths_to_dft"] = df.loc[acct7_idx[6], "mob"] + 10

    df = df.reset_index(drop=True)
    return df


@pytest.fixture
def variables_cfg():
    from eda.models import VariableInfo
    return {
        "obs_month": VariableInfo(name="obs_month", var_type="Date", downstream=["ALL"]),
        "acct_id": VariableInfo(name="acct_id", var_type="ID", downstream=["ALL"]),
        "dpd": VariableInfo(name="dpd", var_type="Value", downstream=["PD", "DF"], constraints={"min": 0}),
        "cpd": VariableInfo(name="cpd", var_type="Value", downstream=["PD", "DF"], constraints={"min": 0}),
        "balance": VariableInfo(name="balance", var_type="Value", downstream=["LGD", "EAD"], constraints={"min": 0}),
        "recovery": VariableInfo(name="recovery", var_type="Value", downstream=["LGD"]),
        "ind_dft": VariableInfo(name="ind_dft", var_type="Status", downstream=["PD", "DF"], valid_values=[0, 1]),
        "ind_CO": VariableInfo(name="ind_CO", var_type="Status", downstream=["LGD", "ERL"], valid_values=[0, 1]),
        "ind_closed": VariableInfo(name="ind_closed", var_type="Status", downstream=["ERL", "LGD"], valid_values=[0, 1]),
        "score_orig": VariableInfo(name="score_orig", var_type="Score", downstream=["Score_Alignment", "PD"]),
        "score_bhv": VariableInfo(name="score_bhv", var_type="Score", downstream=["Score_Alignment", "SICR"]),
        "interest_rate": VariableInfo(name="interest_rate", var_type="Rate", downstream=["LGD"]),
        "perf_lvl1": VariableInfo(name="perf_lvl1", var_type="Status", downstream=["ALL"], valid_values=[0, 1, 2, 3, 4]),
        "perf_lvl2": VariableInfo(name="perf_lvl2", var_type="Status", downstream=["ALL"]),
        "dt_opened": VariableInfo(name="dt_opened", var_type="Date", downstream=["ALL"]),
        "mob": VariableInfo(name="mob", var_type="Term", downstream=["DF", "ERL"]),
        "new_to_dft": VariableInfo(name="new_to_dft", var_type="Status", downstream=["PD"], valid_values=[0, 1]),
        "lag_ind_dft": VariableInfo(name="lag_ind_dft", var_type="Status", downstream=["PD"], valid_values=[0, 1]),
        "next_dft_bal": VariableInfo(name="next_dft_bal", var_type="Value", downstream=["LGD", "EAD"]),
        "mths_to_dft": VariableInfo(name="mths_to_dft", var_type="Term", downstream=["PD", "DF"], constraints={"min": 0}),
    }


@pytest.fixture
def checks_cfg():
    import os
    from eda.loader import load_checks_config
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return load_checks_config(os.path.join(base, "config", "checks.yaml"))
