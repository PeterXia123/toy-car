from __future__ import annotations

import os
import sys
from datetime import datetime

from eda.loader import (
    load_checks_config,
    load_data,
    load_project_config,
    load_variables_config,
)
from eda.models import Finding, finding_sort_key
from eda.checks import data_quality, consistency, trends
from eda.reporting import charts, issue_log, html_report, case_sheets


def run_validation(
    project_config_path: str,
    checks_config_path: str | None = None,
    variables_config_path: str | None = None,
    only: list[str] | None = None,
    extra_findings: list[Finding] | None = None,
) -> list[Finding]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if checks_config_path is None:
        checks_config_path = os.path.join(base_dir, "config", "checks.yaml")
    if variables_config_path is None:
        variables_config_path = os.path.join(base_dir, "config", "variables.yaml")

    project_cfg = load_project_config(project_config_path)
    checks_cfg = load_checks_config(checks_config_path)
    variables_cfg = load_variables_config(variables_config_path)

    print(f"{'='*60}")
    print(f"EDA Toolkit — {project_cfg.name}")
    print(f"{'='*60}")
    print(f"Country: {project_cfg.country} | Product: {project_cfg.product} | Type: {project_cfg.product_type}")
    print(f"Loading data...")

    df = load_data(project_cfg)

    # Keep only columns used by checks — drop everything else to save memory
    keep_cols = set(variables_cfg.keys()) | {
        "obs_month", "acct_id", "perf_lvl1", "perf_lvl2",
        "ind_dft", "ind_closed", "ind_CO", "ind_excl",
        "new_to_dft", "lag_ind_dft", "lag_ind_CO", "new_to_CO",
        "ind_restructure", "dpd", "cpd", "balance", "recovery",
        "next_dft_bal", "mths_to_dft", "dt_opened", "mob",
        "score_orig", "score_bhv", "interest_rate",
        "ln_term", "ln_value", "booked_amt", "maturity_dt",
        "remaining_term", "credit_limit",
    }
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "acct_id" in df.columns and "obs_month" in df.columns:
        df = df.sort_values(["acct_id", "obs_month"])

    print(f"Data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Observation months: {df['obs_month'].nunique() if 'obs_month' in df.columns else 'N/A'}")

    mapped = [v for v in variables_cfg if v in df.columns]
    missing = [v for v in variables_cfg if v not in df.columns]
    print(f"Variables mapped: {len(mapped)} / {len(variables_cfg)}")
    if missing:
        print(f"Variables missing: {', '.join(missing)}")
    print()

    findings: list[Finding] = []
    categories_run = []

    if _should_run("data_quality", only):
        print("Running data quality checks (DQ1-DQ5)...")
        findings += data_quality.run(df, checks_cfg, variables_cfg, project_cfg.product)
        categories_run.append("data_quality")

    if _should_run("consistency", only):
        print("Running consistency checks (DF, TE, LG, PL, DO)...")
        findings += consistency.run(df, checks_cfg, variables_cfg, project_cfg.product)
        categories_run.append("consistency")

    if _should_run("score_alignment", only):
        print("Running score alignment checks (SA1-SA7)...")
        findings += consistency.run_score_alignment(df, checks_cfg, project_cfg.product)
        categories_run.append("score_alignment")

    if _should_run("trends", only):
        print("Running trend analysis...")
        findings += trends.run(df, checks_cfg, variables_cfg, project_cfg.product)
        categories_run.append("trends")

    if _should_run("account_tracking", only):
        print("Running account tracking checks (AT1-AT8)...")
        findings += trends.run_account_tracking(df, checks_cfg, project_cfg.product)
        categories_run.append("account_tracking")

    if _should_run("term_checks", only) and project_cfg.product_type == "term":
        print("Running term product checks (TM1-TM9)...")
        findings += consistency.run_term_checks(df, checks_cfg, project_cfg.product)
        categories_run.append("term_checks")

    if _should_run("revolving_checks", only) and project_cfg.product_type == "revolving":
        print("Running revolving product checks (RV1-RV5)...")
        findings += consistency.run_revolving_checks(df, checks_cfg, project_cfg.product)
        categories_run.append("revolving_checks")

    if extra_findings:
        for ef in extra_findings:
            ef.product = project_cfg.product
        findings = extra_findings + findings

    _enrich_downstream(findings, variables_cfg)

    # Write case sheets before releasing data
    case_path = os.path.join(project_cfg.output_directory, "Case_Sheets.xlsx")
    n_cases = case_sheets.generate(findings, case_path)
    if n_cases:
        print(f"  {n_cases} case sheets saved to {case_path}")

    # Clear case_data from findings to free memory, then release data
    for f in findings:
        f.case_data = None

    del df
    import gc; gc.collect()

    findings.sort(key=finding_sort_key)

    _ensure_output_dirs(project_cfg)

    print(f"\nGenerating charts...")
    charts.generate_all_charts(findings, project_cfg.charts_dir)
    chart_count = sum(1 for f in findings if f.chart_path)
    print(f"  {chart_count} charts generated in {project_cfg.charts_dir}")

    output_path = os.path.join(project_cfg.output_directory, project_cfg.issue_log_file)
    print(f"Generating Issue Log...")
    issue_log.generate(findings, output_path)
    print(f"  Saved to {output_path}")

    html_path = os.path.join(project_cfg.output_directory, project_cfg.issue_log_file.replace(".xlsx", ".html"))
    print(f"Generating HTML Report...")
    html_report.generate(findings, html_path, project_name=project_cfg.name)
    print(f"  Saved to {html_path}")

    high = sum(1 for f in findings if f.impact == "High")
    med = sum(1 for f in findings if f.impact == "Medium")
    low = sum(1 for f in findings if f.impact == "Low")

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(findings)} findings ({high} High, {med} Medium, {low} Low)")
    print(f"Categories run: {', '.join(categories_run)}")
    print(f"{'='*60}")

    return findings


def _should_run(category: str, only: list[str] | None) -> bool:
    if only is None:
        return True
    return category in only


def _enrich_downstream(findings: list[Finding], variables_cfg: dict) -> None:
    for f in findings:
        if f.variable in variables_cfg:
            var_info = variables_cfg[f.variable]
            for ds in var_info.downstream:
                if ds not in f.downstream:
                    f.downstream.append(ds)


def _ensure_output_dirs(project_cfg) -> None:
    os.makedirs(project_cfg.output_directory, exist_ok=True)
    os.makedirs(project_cfg.charts_dir, exist_ok=True)
