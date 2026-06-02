#!/usr/bin/env python3
"""MVA Validator — IFRS 9 ECL Model Validation Automation"""

from __future__ import annotations

import argparse
import sys

from mva.engine import run_validation
from mva.loader import load_checks_config


def main():
    parser = argparse.ArgumentParser(
        description="MVA Validator: Automated IFRS 9 ECL data quality and consistency checks",
    )
    parser.add_argument(
        "--project", required=True,
        help="Path to project YAML config (e.g., config/projects/mx_cc.yaml)",
    )
    parser.add_argument(
        "--checks", default=None,
        help="Path to checks YAML config (default: config/checks.yaml)",
    )
    parser.add_argument(
        "--variables", default=None,
        help="Path to variables YAML config (default: config/variables.yaml)",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated categories to run: data_quality,consistency,score_alignment,trends,account_tracking,term_checks,revolving_checks",
    )
    parser.add_argument(
        "--list-checks", action="store_true",
        help="List all available checks and exit",
    )

    args = parser.parse_args()

    if args.list_checks:
        _list_checks(args.checks)
        return

    only = None
    if args.only:
        only = [c.strip() for c in args.only.split(",")]

    try:
        findings = run_validation(
            project_config_path=args.project,
            checks_config_path=args.checks,
            variables_config_path=args.variables,
            only=only,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


def _list_checks(checks_path: str | None = None):
    import os
    if checks_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        checks_path = os.path.join(base, "config", "checks.yaml")

    cfg = load_checks_config(checks_path)

    print("Available checks:")
    print("=" * 60)

    for section, section_cfg in cfg.items():
        if not isinstance(section_cfg, dict):
            continue
        print(f"\n[{section}]")
        for name, check_cfg in section_cfg.items():
            if not isinstance(check_cfg, dict):
                continue
            enabled = check_cfg.get("enabled", False)
            status = "ON " if enabled else "OFF"
            impact = check_cfg.get("impact", "—")
            print(f"  [{status}] {name:40s} Impact: {impact}")


if __name__ == "__main__":
    main()
