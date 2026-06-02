"""Generate realistic synthetic parquet data with NaN edge cases."""
import numpy as np
import pandas as pd
import os

np.random.seed(2024)

N_ACCOUNTS = 500
N_MONTHS = 24
months = pd.date_range("2022-01-01", periods=N_MONTHS, freq="MS")

rows = []
for i in range(N_ACCOUNTS):
    acct_id = f"MX_{i:06d}"
    dt_opened = months[0] - pd.DateOffset(months=np.random.randint(0, 60))
    # Some accounts appear for fewer months (right censoring)
    if i < 30:
        active_months = months[:np.random.randint(3, 15)]
    else:
        active_months = months

    for m in active_months:
        rows.append({"acct_id": acct_id, "obs_month": m, "dt_opened": dt_opened})

df = pd.DataFrame(rows)
print(f"Base shape: {df.shape}")

# --- Core variables ---
df["dpd"] = np.random.choice([0]*70 + [30]*10 + [60]*8 + [90]*6 + [120]*4 + [180]*2, size=len(df))
df["cpd"] = np.where(df["dpd"] >= 90, np.minimum(df["dpd"] // 30, 8), 0)
df["ind_CO"] = np.where((df["dpd"] >= 180) & (np.random.random(len(df)) < 0.7), 1, 0)
df["ind_closed"] = 0
df["ind_dft"] = np.where((df["dpd"] >= 90) | (df["cpd"] >= 4) | (df["ind_CO"] == 1), 1, 0)
df["new_to_dft"] = 0
df["lag_ind_dft"] = 0
df["balance"] = np.random.lognormal(mean=9, sigma=1.2, size=len(df)).round(2)
df["recovery"] = np.where(df["ind_dft"] == 1, np.random.uniform(0, 3000, size=len(df)), 0).round(2)
df["score_orig"] = np.random.normal(600, 100, size=len(df)).clip(200, 999).round(0)
df["score_bhv"] = np.random.normal(550, 120, size=len(df)).clip(200, 999).round(0)
df["interest_rate"] = np.random.uniform(0.08, 0.45, size=len(df)).round(4)
df["mob"] = ((df["obs_month"] - df["dt_opened"]).dt.days / 30.44).round().astype(int)

# perf_lvl1 derivation
df["perf_lvl1"] = 0
df.loc[df["ind_dft"] == 1, "perf_lvl1"] = 1
df.loc[df["ind_closed"] == 1, "perf_lvl1"] = 2
df["perf_lvl2"] = df["perf_lvl1"]

# --- Inject NaN edge cases ---

# 1) DPD missing (~8% overall, concentrated in some months)
dpd_nan_idx = df.sample(frac=0.08, random_state=1).index
df.loc[dpd_nan_idx, "dpd"] = np.nan

# 2) Balance missing (~5%)
bal_nan_idx = df.sample(frac=0.05, random_state=2).index
df.loc[bal_nan_idx, "balance"] = np.nan

# 3) Score missing (~12% for score_bhv, ~6% for score_orig)
bhv_nan = df.sample(frac=0.12, random_state=3).index
df.loc[bhv_nan, "score_bhv"] = np.nan
orig_nan = df.sample(frac=0.06, random_state=4).index
df.loc[orig_nan, "score_orig"] = np.nan

# 4) Recovery NaN where balance exists and ind_dft=1 (NaN propagation scenario)
rec_nan_idx = df[(df["ind_dft"] == 1) & df["balance"].notna()].sample(frac=0.2, random_state=5).index
df.loc[rec_nan_idx, "recovery"] = np.nan

# 5) dt_opened missing for ~3%
dt_nan = df.sample(frac=0.03, random_state=6).index
df.loc[dt_nan, "dt_opened"] = pd.NaT

# 6) Negative balance
neg_idx = df.sample(n=15, random_state=7).index
df.loc[neg_idx, "balance"] = np.random.uniform(-5000, -100, size=15).round(2)

# 7) interest_rate = 0 for some
zero_ir = df.sample(n=50, random_state=8).index
df.loc[zero_ir, "interest_rate"] = 0.0

# 8) interest_rate extreme (>0.5)
ext_ir = df.sample(n=10, random_state=9).index
df.loc[ext_ir, "interest_rate"] = np.random.uniform(0.6, 1.5, size=10).round(4)

# 9) ind_closed cummax violation: a few accounts go 1 -> 0
for acct_idx in range(5, 10):
    acct = f"MX_{acct_idx:06d}"
    mask = df["acct_id"] == acct
    idxs = df.loc[mask].index
    if len(idxs) > 12:
        df.loc[idxs[8], "ind_closed"] = 1
        df.loc[idxs[9], "ind_closed"] = 0  # violation

# 10) DPD missing with chargeoff
co_mask = df["ind_CO"] == 1
co_sample = df[co_mask].sample(frac=0.3, random_state=10).index
df.loc[co_sample, "dpd"] = np.nan

# 11) Score = 0 or > 1000 (edge)
df.loc[df.sample(n=8, random_state=11).index, "score_orig"] = 0
df.loc[df.sample(n=5, random_state=12).index, "score_orig"] = 1200

# 12) mob with NaN (from dt_opened NaN)
df.loc[dt_nan, "mob"] = np.nan

# 13) Entirely NaN columns should NOT exist, but some records all NaN
all_nan_idx = df.sample(n=3, random_state=13).index
for col in ["dpd", "balance", "score_orig", "score_bhv", "recovery", "interest_rate"]:
    df.loc[all_nan_idx, col] = np.nan

# 14) cpd as float (NaN makes int->float)
cpd_nan = df.sample(n=20, random_state=14).index
df.loc[cpd_nan, "cpd"] = np.nan

# 15) obs_month before dt_opened (left censoring)
left_cens = df.sample(n=10, random_state=15).index
df.loc[left_cens, "dt_opened"] = df.loc[left_cens, "obs_month"] + pd.DateOffset(months=3)

# --- Save ---
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "mx_cc_synthetic.parquet")
df.to_parquet(out_path, index=False)
print(f"Saved to {out_path}")
print(f"Shape: {df.shape}")
print(f"\nNaN counts:")
print(df.isna().sum())
print(f"\nDtypes:")
print(df.dtypes)
