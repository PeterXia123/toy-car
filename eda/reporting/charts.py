from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from eda.models import Finding


_COLORS = {
    "blue": "#2196F3",
    "red": "#F44336",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "grey": "#9E9E9E",
    "dark_blue": "#1565C0",
    "light_blue": "#90CAF9",
}


def setup_style():
    plt.rcParams.update({
        "figure.figsize": (20, 6),
        "figure.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def generate_all_charts(findings: list[Finding], charts_dir: str) -> None:
    setup_style()
    os.makedirs(charts_dir, exist_ok=True)

    for f in findings:
        if f.stats is None:
            continue

        path = None
        try:
            if f.check_id == "TE7" or f.check_id == "TE8":
                path = _plot_binary_indicator_trend(f, charts_dir)
            elif f.check_id == "LG6":
                path = _plot_recovery_trend(f, charts_dir)
            elif f.check_id == "LG10":
                path = _plot_balance_trend(f, charts_dir)
            elif f.check_id == "DF7":
                path = _plot_default_rate_trend(f, charts_dir)
            elif f.check_id == "SA7" and "monthly_mean" in f.stats:
                path = _plot_score_distribution(f, charts_dir)
            elif f.check_id == "DQ3_TREND":
                path = _plot_account_count(f, charts_dir)
            elif f.check_id == "AT2":
                path = _plot_censored_accounts(f, charts_dir)
            elif f.check_id == "DQ1_HEATMAP":
                path = _plot_missing_heatmap(f, charts_dir)
            elif f.check_id == "SA1":
                path = _plot_score_default_monotonicity(f, charts_dir)
            elif f.check_id == "DF8":
                path = _plot_dpd_trend(f, charts_dir)
            elif f.check_id == "PL2":
                path = _plot_perf_lvl_distribution(f, charts_dir)
            elif f.check_id == "SA8":
                path = _plot_segment_default_trend(f, charts_dir)
            elif f.check_id == "AT9":
                path = _plot_disappear_classification(f, charts_dir)
            elif f.check_id == "TM5":
                path = _plot_loan_term_distribution(f, charts_dir)
            elif f.check_id == "RV4":
                path = _plot_utilization_distribution(f, charts_dir)
            elif f.check_id == "MD3":
                path = _plot_mths_to_dft_trend(f, charts_dir)
            elif f.check_id in ("SA5", "SA6"):
                path = _plot_score_missing_trend(f, charts_dir)
            elif f.check_id == "LV3":
                path = _plot_loan_value_trend(f, charts_dir)
            elif f.check_id == "RS1":
                path = _plot_restructure_trend(f, charts_dir)
            elif f.check_id == "LG11":
                path = _plot_interest_rate_trend(f, charts_dir)
            elif f.check_id == "RV5":
                path = _plot_credit_limit_trend(f, charts_dir)
            elif f.check_id in ("LG12", "LG13"):
                path = _plot_lgd_workout(f, charts_dir)
        except Exception:
            pass

        if path:
            f.chart_path = path


def _plot_binary_indicator_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    v0 = [trend[m]["value_0"] for m in months]
    v1 = [trend[m]["value_1"] for m in months]
    n = [trend[m]["n_records"] for m in months]
    rate = [trend[m]["value_1"] / trend[m]["n_records"] if trend[m]["n_records"] > 0 else 0 for m in months]

    fig, ax1 = plt.subplots(figsize=(20, 6))

    x = np.arange(len(months))
    width = 0.6
    ax1.bar(x, v0, width, label=f"{f.variable}=0", color=_COLORS["light_blue"])
    ax1.bar(x, v1, width, bottom=v0, label=f"{f.variable}=1", color=_COLORS["red"])
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Observation Month")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, rate, color=_COLORS["dark_blue"], marker="o", linewidth=2, label=f"{f.variable}=1 Rate")
    ax2.set_ylabel(f"{f.variable}=1 Rate")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(f"{f.variable} Trend by Observation Month")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_{f.variable}_trend.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_recovery_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("recovery_trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    avg = [trend[m]["avg"] for m in months]
    total = [trend[m]["sum"] for m in months]

    fig, ax1 = plt.subplots(figsize=(20, 6))

    x = np.arange(len(months))
    ax1.bar(x, total, color=_COLORS["light_blue"], alpha=0.7, label="Recovery Sum")
    ax1.set_ylabel("Recovery Sum")
    ax1.set_xlabel("Observation Month")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))

    ax2 = ax1.twinx()
    ax2.plot(x, avg, color=_COLORS["red"], marker="o", linewidth=2, label="Recovery Avg")
    ax2.set_ylabel("Recovery Average")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Recovery Trend by Observation Month")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_recovery_trend.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_balance_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("balance_trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    means = [trend[m]["mean"] for m in months]
    medians = [trend[m]["median"] for m in months]

    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(range(len(months)), means, marker="o", linewidth=2, color=_COLORS["blue"], label="Mean Balance")
    ax.plot(range(len(months)), medians, marker="s", linewidth=2, color=_COLORS["green"], label="Median Balance")
    ax.set_ylabel("Balance")
    ax.set_xlabel("Observation Month")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.legend()

    plt.title("Balance Trend by Observation Month")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_balance_trend.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_default_rate_trend(f: Finding, charts_dir: str) -> str | None:
    dft_rate = f.stats.get("default_rate", {})
    ntd_rate = f.stats.get("new_to_dft_rate", {})
    if not dft_rate:
        return None

    months = sorted(dft_rate.keys())
    dft_vals = [dft_rate[m] for m in months]

    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(range(len(months)), dft_vals, marker="o", linewidth=2, color=_COLORS["red"], label="Default Rate (ind_dft)")

    if ntd_rate:
        ntd_vals = [ntd_rate.get(m, 0) for m in months]
        ax.plot(range(len(months)), ntd_vals, marker="s", linewidth=2, color=_COLORS["orange"], label="New-to-Default Rate")

    ax.set_ylabel("Rate")
    ax.set_xlabel("Observation Month")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()

    plt.title("Default Rate Trend by Observation Month")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_default_rate_trend.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_score_distribution(f: Finding, charts_dir: str) -> str | None:
    monthly_mean = f.stats.get("monthly_mean", {})
    if not monthly_mean:
        return None

    months = sorted(monthly_mean.keys())
    vals = [monthly_mean[m] for m in months]

    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(range(len(months)), vals, marker="o", linewidth=2, color=_COLORS["purple"])
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Observation Month")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")

    drift_months = f.stats.get("drift_months", {})
    if drift_months:
        for m in drift_months:
            if m in months:
                idx = months.index(m)
                ax.axvline(x=idx, color=_COLORS["red"], linestyle="--", alpha=0.7)
                ax.annotate(f"drift", (idx, vals[idx]), fontsize=8, color=_COLORS["red"])

    plt.title(f"{f.variable} Score Distribution Drift")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_{f.variable}_drift.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_account_count(f: Finding, charts_dir: str) -> str | None:
    counts = f.stats.get("account_counts", {})
    if not counts:
        return None

    months = sorted(counts.keys())
    vals = [counts[m] for m in months]

    fig, ax = plt.subplots(figsize=(20, 6))

    ax.bar(range(len(months)), vals, color=_COLORS["blue"], alpha=0.8)
    ax.set_ylabel("Record Count")
    ax.set_xlabel("Observation Month")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))

    plt.title("Record Count by Observation Month")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_account_count.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_censored_accounts(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("attrition_trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    rates = [trend[m]["rate"] for m in months]
    disappeared = [trend[m]["disappeared"] for m in months]
    active = [trend[m]["active"] for m in months]

    fig, ax1 = plt.subplots(figsize=(20, 6))

    x = np.arange(len(months))
    bars = ax1.bar(x, rates, color=_COLORS["orange"], alpha=0.8, width=0.7)
    ax1.set_ylabel("Attrition Rate (unexplained)")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")

    for bar, d, a in zip(bars, disappeared, active):
        if d > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{d}/{a}", ha="center", va="bottom", fontsize=8, color="#666")

    avg_rate = sum(rates) / len(rates) if rates else 0
    ax1.axhline(y=avg_rate, color=_COLORS["red"], linestyle="--", alpha=0.7,
                label=f"Avg: {avg_rate:.2%}/month")

    ax1.set_title("Monthly Unexplained Account Attrition Rate", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_attrition_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_missing_heatmap(f: Finding, charts_dir: str) -> str | None:
    heatmap_data = f.stats.get("missing_heatmap", {})
    if not heatmap_data:
        return None

    import pandas as pd
    df_heat = pd.DataFrame(heatmap_data).T
    if df_heat.empty:
        return None

    df_heat = df_heat.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(max(14, len(df_heat.columns) * 0.8), max(6, len(df_heat) * 0.5)))

    im = ax.imshow(df_heat.values, cmap="Reds", aspect="auto", vmin=0, vmax=max(0.1, df_heat.values.max()))

    ax.set_xticks(range(len(df_heat.columns)))
    ax.set_xticklabels([str(c)[:7] for c in df_heat.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df_heat.index)))
    ax.set_yticklabels(df_heat.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Missing Rate")

    for i in range(len(df_heat.index)):
        for j in range(len(df_heat.columns)):
            val = df_heat.iloc[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.5 else "black")

    plt.title("Missing Value Heatmap (Variable × Observation Month)")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_missing_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_score_default_monotonicity(f: Finding, charts_dir: str) -> str | None:
    seg_rates = f.stats.get("segment_default_rates", {})
    if not seg_rates:
        return None

    segments = sorted(seg_rates.keys(), key=lambda x: int(x))
    rates = [seg_rates[s] for s in segments]

    fig, ax = plt.subplots(figsize=(20, 6))

    colors = [_COLORS["green"] if i == 0 or rates[i] <= rates[i-1]
              else _COLORS["red"] for i in range(len(rates))]
    direction = f.stats.get("direction", "unknown")
    if direction == "ascending":
        colors = [_COLORS["green"] if i == 0 or rates[i] >= rates[i-1]
                  else _COLORS["red"] for i in range(len(rates))]

    ax.bar(range(len(segments)), rates, color=colors, alpha=0.8)
    ax.set_ylabel("Default Rate")
    ax.set_xlabel("Score Segment (Decile)")
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels([f"D{int(s)+1}" for s in segments])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    for i, r in enumerate(rates):
        ax.text(i, r + 0.002, f"{r:.1%}", ha="center", va="bottom", fontsize=8)

    plt.title(f"{f.variable} Score-Default Monotonicity")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_{f.variable}_monotonicity.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_dpd_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("dpd_trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    means = [trend[m]["mean"] for m in months]
    medians = [trend[m]["median"] for m in months]
    p90s = [trend[m]["p90"] for m in months]

    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(range(len(months)), means, marker="o", linewidth=2, color=_COLORS["blue"], label="Mean DPD")
    ax.plot(range(len(months)), medians, marker="s", linewidth=2, color=_COLORS["green"], label="Median DPD")
    ax.plot(range(len(months)), p90s, marker="^", linewidth=2, color=_COLORS["red"], label="90th Percentile")
    ax.set_ylabel("Days Past Due")
    ax.set_xlabel("Observation Month")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.legend()

    plt.title("DPD Distribution Trend by Observation Month")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_dpd_trend.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_perf_lvl_distribution(f: Finding, charts_dir: str) -> str | None:
    dist = f.stats.get("distribution", {})
    if not dist:
        return None

    import pandas as pd
    df_dist = pd.DataFrame(dist).T
    df_dist = df_dist.sort_index()

    fig, ax = plt.subplots(figsize=(20, 6))

    colors = [_COLORS["green"], _COLORS["red"], _COLORS["orange"], _COLORS["grey"], _COLORS["purple"]]
    bottom = np.zeros(len(df_dist))
    for i, col in enumerate(sorted(df_dist.columns, key=lambda x: int(x))):
        vals = df_dist[col].values
        color = colors[int(col) % len(colors)]
        ax.bar(range(len(df_dist)), vals, bottom=bottom, label=f"perf_lvl1={col}", color=color, alpha=0.8)
        bottom += vals

    ax.set_ylabel("Proportion")
    ax.set_xlabel("Observation Month")
    ax.set_xticks(range(len(df_dist)))
    ax.set_xticklabels([str(m)[:7] for m in df_dist.index], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right")

    plt.title("perf_lvl1 Distribution Trend by Observation Month")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{f.check_id}_perf_lvl_dist.png")
    fig.savefig(path)
    plt.close(fig)
    return path




def _plot_segment_default_trend(f: Finding, charts_dir: str) -> str | None:
    seg_rates = f.stats.get("segment_rates", {})
    seg_labels = f.stats.get("segment_labels", {})
    if not seg_rates:
        return None

    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_facecolor("#FAFAFA")

    all_vals = []
    for i, (seg, rates) in enumerate(sorted(seg_rates.items(), key=lambda x: int(x[0]))):
        months = sorted(rates.keys())
        vals = [rates[m] for m in months]
        all_vals.extend(vals)
        label = seg_labels.get(str(seg), seg_labels.get(int(seg), f"Seg {seg}"))
        ax.plot(range(len(months)), vals, linewidth=2, color=palette[i % len(palette)],
                label=f"{label}", alpha=0.85, marker="o", markersize=3)

    months = sorted(list(seg_rates.values())[0].keys())
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m[:7] for m in months], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Default Rate", fontsize=10)
    if all_vals:
        y_max = max(all_vals)
        ax.set_ylim(bottom=0, top=max(y_max * 1.3, 0.01))
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.grid(axis="x", alpha=0.1, linewidth=0.5)

    score_name = f.variable or "score"
    ax.set_title(f"Default Rate by {score_name} Segment", fontsize=12, fontweight="bold", pad=12)

    ax.legend(title="Score Range", loc="upper right", fontsize=8, title_fontsize=9,
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_{score_name}_segment_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_disappear_classification(f: Finding, charts_dir: str) -> str | None:
    dist = f.stats.get("distribution", {})
    if not dist:
        return None

    palette = {"Default": "#C44E52", "Closed": "#4C72B0", "Exclusion": "#8172B2", "Censored": "#CCB974"}
    categories = ["Default", "Closed", "Exclusion", "Censored"]
    counts = [dist.get(c, {}).get("count", 0) for c in categories]
    pcts = [dist.get(c, {}).get("pct", 0) for c in categories]
    colors = [palette[c] for c in categories]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_facecolor("#FAFAFA")

    bars = ax.barh(categories, counts, color=colors, height=0.55, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        w = bar.get_width()
        ax.text(w + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}  ({pct:.1%})", va="center", fontsize=9)

    ax.set_xlabel("Number of Accounts")
    ax.set_title("Disappeared Accounts by Final Status", fontsize=12, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_disappear_classification.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_loan_term_distribution(f: Finding, charts_dir: str) -> str | None:
    dist = f.stats.get("distribution", {})
    if not dist:
        return None

    buckets = list(dist.keys())
    counts = [dist[b]["count"] for b in buckets]
    pcts = [dist[b]["pct"] for b in buckets]
    avg_term = f.stats.get("avg_term")
    avg_acct = f.stats.get("avg_term_per_account")

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_facecolor("#FAFAFA")

    bars = ax.bar(range(len(buckets)), counts, color="#4C72B0", width=0.7, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        if pct >= 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(counts) * 0.01,
                    f"{pct:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Records")
    ax.set_xlabel("Loan Term (months)")
    ax.set_title("Loan Term Distribution", fontsize=12, fontweight="bold", pad=12)

    note = f"Average term: {avg_term:.1f} months" if avg_term else ""
    if avg_acct:
        note += f"  |  Average per account: {avg_acct:.1f} months"
    if note:
        fig.text(0.5, -0.01, note, ha="center", fontsize=10, fontstyle="italic", color="#555555")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(charts_dir, f"{f.check_id}_loan_term_dist.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_utilization_distribution(f: Finding, charts_dir: str) -> str | None:
    dist = f.stats.get("distribution", {})
    if not dist:
        return None

    buckets = list(dist.keys())
    counts = [dist[b]["count"] for b in buckets]
    pcts = [dist[b]["pct"] for b in buckets]
    mean_u = f.stats.get("mean_util")
    median_u = f.stats.get("median_util")

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_facecolor("#FAFAFA")

    colors = ["#4C72B0"] * (len(buckets) - 1) + ["#C44E52"] if len(buckets) > 1 else ["#4C72B0"]
    if len(colors) < len(buckets):
        colors = ["#4C72B0"] * len(buckets)

    bars = ax.bar(range(len(buckets)), counts, color=colors, width=0.7, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        if pct >= 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(counts) * 0.01,
                    f"{pct:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Records")
    ax.set_xlabel("Utilization (balance / credit_limit)")
    ax.set_title("Utilization Distribution", fontsize=12, fontweight="bold", pad=12)

    note_parts = []
    if mean_u is not None:
        note_parts.append(f"Mean: {mean_u:.2%}")
    if median_u is not None:
        note_parts.append(f"Median: {median_u:.2%}")
    over_1 = f.stats.get("over_1_pct")
    if over_1 is not None:
        note_parts.append(f">100%: {over_1:.1%}")
    if note_parts:
        fig.text(0.5, -0.01, "  |  ".join(note_parts),
                 ha="center", fontsize=10, fontstyle="italic", color="#555555")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    fig.text(0.5, -0.05, "Remind: utilization >1 indicates balance exceeds credit limit — review CCF/EAD impact.",
             ha="center", fontsize=9, color="#888888")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(charts_dir, f"{f.check_id}_utilization_dist.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_mths_to_dft_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("mths_to_dft_trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    means = [trend[m]["mean"] for m in months]
    medians = [trend[m]["median"] for m in months]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_facecolor("#FAFAFA")

    ax.plot(range(len(months)), means, color="#4C72B0", linewidth=1.5, label="Mean")
    ax.plot(range(len(months)), medians, color="#55A868", linewidth=1.5, label="Median")

    ax.set_xticks(range(len(months)))
    labels = [str(m)[:7] if len(str(m)) > 7 else str(m) for m in months]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Months to Default")
    ax.set_title("Months-to-Default Trend by Observation Month", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_mths_to_dft_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_score_missing_trend(f: Finding, charts_dir: str) -> str | None:
    per_month = f.stats.get("per_month", {})
    if not per_month:
        return None

    months = sorted(per_month.keys())
    vals = [per_month[m] for m in months]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(range(len(months)), vals, marker="o", linewidth=2, color=_COLORS["red"])
    ax.fill_between(range(len(months)), vals, alpha=0.15, color=_COLORS["red"])
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Missing Rate")
    ax.set_title(f"{f.variable} Missing Rate Trend", fontsize=12, fontweight="bold", pad=12)

    overall = f.stats.get("overall_rate", 0)
    ax.axhline(y=overall, color=_COLORS["grey"], linestyle="--", alpha=0.7, label=f"Overall: {overall:.1%}")
    ax.legend(fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_{f.variable}_missing_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_loan_value_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("ln_value_trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    means = [trend[m]["mean"] for m in months]
    medians = [trend[m]["median"] for m in months]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(range(len(months)), means, marker="o", linewidth=2, color=_COLORS["blue"], label="Mean")
    ax.plot(range(len(months)), medians, marker="s", linewidth=2, color=_COLORS["green"], label="Median")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.set_ylabel("Loan Value")
    ax.set_title("Loan Value Trend by Observation Month", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_ln_value_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_restructure_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    vals = [trend[m] for m in months]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(len(months)), vals, color=_COLORS["orange"], alpha=0.8, width=0.7)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Restructure Rate")
    ax.set_title("Restructure Rate Trend by Observation Month", fontsize=12, fontweight="bold", pad=12)

    overall = f.stats.get("overall_rate", 0)
    ax.axhline(y=overall, color=_COLORS["red"], linestyle="--", alpha=0.7, label=f"Overall: {overall:.2%}")
    ax.legend(fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_restructure_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_interest_rate_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("ir_trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    means = [trend[m]["mean"] for m in months]
    medians = [trend[m]["median"] for m in months]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(range(len(months)), means, marker="o", linewidth=2, color=_COLORS["blue"], label="Mean")
    ax.plot(range(len(months)), medians, marker="s", linewidth=2, color=_COLORS["green"], label="Median")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Interest Rate")
    ax.set_title("Interest Rate Trend by Observation Month", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_interest_rate_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_credit_limit_trend(f: Finding, charts_dir: str) -> str | None:
    trend = f.stats.get("limit_trend", {})
    if not trend:
        return None

    months = sorted(trend.keys())
    means = [trend[m]["mean"] for m in months]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(range(len(months)), means, marker="o", linewidth=2, color=_COLORS["purple"])
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.set_ylabel("Credit Limit (Mean)")
    ax.set_title("Credit Limit Trend by Observation Month", fontsize=12, fontweight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_credit_limit_trend.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_lgd_workout(f: Finding, charts_dir: str) -> str | None:
    cohort = f.stats.get("cohort", {})
    overall = f.stats.get("overall", {})
    if not cohort and not overall:
        return None

    n_total = overall.get("count", 0)
    fig, ax1 = plt.subplots(figsize=(20, 6))

    months = sorted(cohort.keys())
    x = np.arange(len(months))
    means = [cohort[m].get("mean", 0) for m in months]
    counts = [cohort[m].get("count", 0) for m in months]

    bars = ax1.bar(x, means, 0.7, color=_COLORS["blue"], alpha=0.85)
    for bar, c in zip(bars, counts):
        if c > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"n={c}",
                     ha="center", va="bottom", fontsize=8, color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m)[:7] for m in months], rotation=45, ha="right")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    overall_mean = overall.get("mean")
    if overall_mean is not None:
        ax1.axhline(y=overall_mean, color=_COLORS["orange"], linestyle="--", alpha=0.6,
                     label=f"Overall Mean: {overall_mean:.1%}")
    ax1.set_ylabel("Recovery Rate (PV / Default Balance)")
    title_suffix = " — Imputed from Balance Δ" if f.check_id == "LG13" else ""
    ax1.set_title(f"LGD Workout Recovery Rate by Default Cohort{title_suffix}\n(n={n_total} accounts, 36-month discounted window)",
                  fontsize=11, fontweight="bold", pad=12)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(charts_dir, f"{f.check_id}_lgd_workout.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
