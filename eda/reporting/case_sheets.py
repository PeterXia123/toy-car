from __future__ import annotations

import os

import pandas as pd

from eda.models import Finding


def generate(findings: list[Finding], output_path: str) -> int:
    sheets: dict[str, pd.DataFrame] = {}
    for f in findings:
        if f.case_data is None or f.case_data.empty:
            continue
        sheet_name = f"{f.check_id}_{f.variable}"[:31]
        if sheet_name in sheets:
            continue
        sheets[sheet_name] = f.case_data

    if not sheets:
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    return len(sheets)
