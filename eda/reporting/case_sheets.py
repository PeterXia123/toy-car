from __future__ import annotations

import os

import pandas as pd

from eda.models import Finding


_DISPLAY_NAMES = {
    "rpt_mth": "obs_month", "eid": "acct_id", "past_d": "dpd",
    "past_c": "cpd", "cur_amt": "balance", "fl_close": "ind_closed",
    "fl_wo": "ind_CO", "fl_evt": "ind_dft", "lag_fl_evt": "lag_ind_dft",
    "new_evt": "new_to_dft", "lag_fl_wo": "lag_ind_CO", "new_wo": "new_to_CO",
    "fl_excl": "ind_excl", "fl_restr": "ind_restructure",
    "sc_orig": "score_orig", "sc_curr": "score_bhv",
    "grp1": "perf_lvl1", "grp2": "perf_lvl2",
    "dt_start": "dt_opened", "dt_end": "maturity_dt",
    "tot_term": "ln_term", "rem_term": "remaining_term",
    "mos_bk": "mob", "rcv_amt": "recovery", "ann_rate": "interest_rate",
    "face_val": "ln_value", "init_amt": "booked_amt",
    "fwd_amt": "next_dft_bal", "mos_to_evt": "mths_to_dft", "max_lim": "credit_limit",
}


def generate(findings: list[Finding], output_path: str) -> int:
    sheets: dict[str, pd.DataFrame] = {}
    for f in findings:
        if f.case_data is None or f.case_data.empty:
            continue
        display_var = _DISPLAY_NAMES.get(f.variable, f.variable)
        sheet_name = f"{f.check_id}_{display_var}"[:31]
        if sheet_name in sheets:
            continue
        sheets[sheet_name] = f.case_data.rename(columns=_DISPLAY_NAMES)

    if not sheets:
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    return len(sheets)
