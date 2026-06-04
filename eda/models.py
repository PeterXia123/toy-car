from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


SECTION_ORDER = {
    "DQ": (1, "Accounts & Indicator Flags"),
    "AT": (1, "Accounts & Indicator Flags"),
    "DF": (1, "Accounts & Indicator Flags"),
    "TE": (1, "Accounts & Indicator Flags"),
    "PL": (1, "Accounts & Indicator Flags"),
    "DO": (1, "Accounts & Indicator Flags"),
    "SA": (2, "Score Alignment"),
    "LG": (3, "Recovery, Balance & LGD"),
    "NB": (3, "Recovery, Balance & LGD"),
    "MD": (3, "Recovery, Balance & LGD"),
    "LV": (3, "Recovery, Balance & LGD"),
    "BA": (4, "Origination"),
    "RS": (1, "Accounts & Indicator Flags"),
    "RV": (5, "Revolving Product"),
    "TM": (6, "Term Product"),
    "TS": (7, "Time Series"),
}

_PREFIX_SUBORDER = {
    "DQ": 0, "AT": 1, "DF": 2, "TE": 3, "PL": 4, "DO": 5,
    "LG": 0, "NB": 1, "MD": 2, "LV": 3,
    "BA": 0, "RS": 6,
}


def finding_sort_key(f: "Finding") -> tuple:
    m = re.match(r"([A-Z]+)", f.check_id)
    prefix = m.group(1) if m else ""
    section = SECTION_ORDER.get(prefix, (99, "Other"))[0]
    sub = _PREFIX_SUBORDER.get(prefix, 0)
    num_m = re.search(r"\d+", f.check_id)
    num = int(num_m.group()) if num_m else 0
    suffix = 1 if f.check_id.endswith(("_HEATMAP", "_TREND")) else 0
    return (section, sub, num, suffix)


@dataclass
class Finding:
    product: str
    parameter: str | list[str]
    impact: str
    question: str
    check_id: str
    variable: str
    downstream: list[str] = field(default_factory=list)
    date: str = field(default_factory=lambda: date.today().strftime("%Y-%m-%d"))
    chart_path: str | None = None
    examples: pd.DataFrame | None = None
    stats: dict[str, Any] | None = None
    reference_only: bool = False
    case_data: pd.DataFrame | None = None

    @property
    def parameter_str(self) -> str:
        if isinstance(self.parameter, list):
            return ", ".join(self.parameter)
        return self.parameter

    @property
    def downstream_str(self) -> str:
        return ", ".join(self.downstream)


@dataclass
class VariableInfo:
    name: str
    var_type: str
    downstream: list[str] = field(default_factory=list)
    description: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    valid_values: list | None = None
    expected_dtype: str = ""
    format_hint: str = ""


@dataclass
class ProjectConfig:
    name: str
    country: str
    product: str
    product_type: str
    entity: str
    data_path: str
    data_file: str
    data_format: str
    column_mapping: dict[str, str]
    filters: dict[str, Any]
    output_directory: str
    issue_log_file: str
    charts_dir: str

    @classmethod
    def from_dict(cls, d: dict) -> ProjectConfig:
        proj = d["project"]
        data = d["data"]
        output = d.get("output", {})
        return cls(
            name=proj.get("name", ""),
            country=proj.get("country", ""),
            product=proj.get("product", ""),
            product_type=proj.get("product_type", "term"),
            entity=proj.get("entity", ""),
            data_path=data.get("path", ""),
            data_file=data.get("file", ""),
            data_format=data.get("format", "parquet"),
            column_mapping=d.get("column_mapping", {}),
            filters=d.get("filters", {}),
            output_directory=output.get("directory", "output/"),
            issue_log_file=output.get("issue_log", "Issue_Log.xlsx"),
            charts_dir=output.get("charts_dir", "output/charts/"),
        )
