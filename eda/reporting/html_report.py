from __future__ import annotations

import base64
import os
from datetime import date

import re

from eda.models import Finding, SECTION_ORDER


def generate(findings: list[Finding], output_path: str, project_name: str = "") -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    issue_findings = [f for f in findings if not f.reference_only]
    chart_findings = [f for f in findings if f.chart_path and os.path.exists(f.chart_path)]

    high = sum(1 for f in issue_findings if f.impact == "High")
    med = sum(1 for f in issue_findings if f.impact == "Medium")
    low = sum(1 for f in issue_findings if f.impact == "Low")

    parameters = sorted(set(f.parameter_str for f in issue_findings))
    check_ids = sorted(set(f.check_id for f in issue_findings))

    html = _TEMPLATE.format(
        project_name=_esc(project_name),
        date=date.today().strftime("%Y-%m-%d"),
        total=len(issue_findings),
        high=high,
        med=med,
        low=low,
        param_options=_build_options(parameters),
        check_options=_build_options(check_ids),
        rows=_build_rows(issue_findings),
        charts=_build_charts(chart_findings),
        summary_table=_build_summary(issue_findings, parameters),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _build_options(items: list[str]) -> str:
    return "\n".join(f'<option value="{_esc(v)}">{_esc(v)}</option>' for v in items)


def _get_section(f: Finding) -> str:
    m = re.match(r"([A-Z]+)", f.check_id)
    prefix = m.group(1) if m else ""
    return SECTION_ORDER.get(prefix, (99, "Other"))[1]


def _build_rows(findings: list[Finding]) -> str:
    rows = []
    current_section = None
    for f in findings:
        section = _get_section(f)
        if section != current_section:
            current_section = section
            rows.append(f'<tr class="section-row"><td colspan="7">{_esc(section)}</td></tr>')

        impact_class = {"High": "impact-high", "Medium": "impact-med", "Low": "impact-low"}.get(f.impact, "")
        chart_link = ""
        if f.chart_path:
            anchor = f"chart-{os.path.basename(f.chart_path).replace('.png','')}"
            chart_link = f'<a href="#{anchor}">View</a>'

        rows.append(f"""<tr class="finding-row" data-impact="{_esc(f.impact)}" data-param="{_esc(f.parameter_str)}" data-check="{_esc(f.check_id)}" data-section="{_esc(section)}">
  <td class="{impact_class}">{_esc(f.impact)}</td>
  <td>{_esc(f.check_id)}</td>
  <td>{_esc(f.variable)}</td>
  <td>{_esc(f.parameter_str)}</td>
  <td class="question-cell">{_esc(f.question)}</td>
  <td>{_esc(f.downstream_str)}</td>
  <td>{chart_link}</td>
</tr>""")
    return "\n".join(rows)


def _build_charts(findings: list[Finding]) -> str:
    parts = []
    current_section = None
    for f in findings:
        if not f.chart_path or not os.path.exists(f.chart_path):
            continue

        section = _get_section(f)
        if section != current_section:
            current_section = section
            parts.append(f'<h2 class="chart-section-header">{_esc(section)}</h2>')

        with open(f.chart_path, "rb") as img:
            b64 = base64.b64encode(img.read()).decode()
        anchor = f"chart-{os.path.basename(f.chart_path).replace('.png','')}"
        title = f"[{f.check_id}] {f.variable}"
        parts.append(f"""<div class="chart-card" id="{anchor}">
  <h3>{_esc(title)}</h3>
  <p class="chart-desc">{_esc(f.question[:120])}</p>
  <img src="data:image/png;base64,{b64}" alt="{_esc(title)}">
</div>""")
    return "\n".join(parts)


def _build_summary(findings: list[Finding], parameters: list[str]) -> str:
    impacts = ["High", "Medium", "Low"]
    counts = {}
    for f in findings:
        p = f.parameter_str
        i = f.impact
        counts[(p, i)] = counts.get((p, i), 0) + 1

    rows = []
    for param in parameters:
        cells = []
        for impact in impacts:
            c = counts.get((param, impact), 0)
            cls = ""
            if c > 0:
                cls = {"High": "impact-high", "Medium": "impact-med", "Low": "impact-low"}[impact]
            cells.append(f'<td class="{cls}">{c}</td>')
        total = sum(counts.get((param, i), 0) for i in impacts)
        rows.append(f"<tr><td>{_esc(param)}</td>{''.join(cells)}<td><strong>{total}</strong></td></tr>")

    total_row = "<tr class='total-row'><td><strong>Total</strong></td>"
    for impact in impacts:
        t = sum(counts.get((p, impact), 0) for p in parameters)
        total_row += f"<td><strong>{t}</strong></td>"
    total_row += f"<td><strong>{len(findings)}</strong></td></tr>"
    rows.append(total_row)

    return "\n".join(rows)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Quality Report — {project_name}</title>
<style>
  :root {{
    --high: #e74c3c; --high-bg: #fdecea;
    --med: #f39c12; --med-bg: #fef5e7;
    --low: #27ae60; --low-bg: #eafaf1;
    --border: #dee2e6; --bg: #f8f9fa;
    --text: #2c3e50; --card-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: var(--text); background: var(--bg); }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

  /* Header */
  .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 24px; margin-bottom: 6px; }}
  .header .meta {{ opacity: 0.85; font-size: 14px; }}

  /* KPI Cards */
  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
  .kpi {{ background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: var(--card-shadow); }}
  .kpi .number {{ font-size: 36px; font-weight: 700; }}
  .kpi .label {{ font-size: 13px; color: #7f8c8d; margin-top: 4px; }}
  .kpi.high .number {{ color: var(--high); }}
  .kpi.med .number {{ color: var(--med); }}
  .kpi.low .number {{ color: var(--low); }}
  .kpi.total .number {{ color: var(--text); }}

  /* Tabs */
  .tabs {{ display: flex; gap: 0; margin-bottom: 0; }}
  .tab {{ padding: 12px 24px; cursor: pointer; background: white; border: 1px solid var(--border); border-bottom: none;
          border-radius: 8px 8px 0 0; font-weight: 600; font-size: 14px; color: #7f8c8d; }}
  .tab.active {{ color: var(--text); background: white; border-bottom: 2px solid white; position: relative; z-index: 1; }}
  .tab-content {{ display: none; background: white; border: 1px solid var(--border); border-radius: 0 8px 8px 8px; padding: 20px; margin-bottom: 24px; }}
  .tab-content.active {{ display: block; }}

  /* Filters */
  .filters {{ display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }}
  .filters label {{ font-size: 13px; font-weight: 600; }}
  .filters select {{ padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; }}

  /* Table */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f1f3f5; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid var(--border); position: sticky; top: 0; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #f1f3f5; vertical-align: top; }}
  tr:hover {{ background: #f8f9fa; }}
  .question-cell {{ max-width: 500px; line-height: 1.5; }}
  .impact-high {{ background: var(--high-bg); color: var(--high); font-weight: 700; text-align: center; border-radius: 4px; }}
  .impact-med {{ background: var(--med-bg); color: var(--med); font-weight: 700; text-align: center; border-radius: 4px; }}
  .impact-low {{ background: var(--low-bg); color: var(--low); font-weight: 700; text-align: center; border-radius: 4px; }}
  .total-row td {{ border-top: 2px solid var(--text); }}
  a {{ color: #3498db; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* Summary Table */
  .summary-table {{ width: auto; }}
  .summary-table td, .summary-table th {{ text-align: center; min-width: 80px; }}
  .summary-table td:first-child, .summary-table th:first-child {{ text-align: left; min-width: 200px; }}

  /* Section rows */
  .section-row td {{ background: #2c3e50; color: white; font-weight: 700; font-size: 13px; padding: 8px 12px; letter-spacing: 0.5px; }}
  .chart-section-header {{ font-size: 18px; font-weight: 700; color: #2c3e50; margin: 28px 0 12px; padding-bottom: 6px; border-bottom: 2px solid #3498db; }}

  /* Charts */
  .charts-grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
  .chart-card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: var(--card-shadow); }}
  .chart-card h3 {{ font-size: 15px; margin-bottom: 6px; }}
  .chart-card .chart-desc {{ font-size: 12px; color: #7f8c8d; margin-bottom: 12px; }}
  .chart-card img {{ width: 100%; height: auto; border-radius: 6px; }}

  .hidden {{ display: none !important; }}

  @media print {{
    .filters, .tabs {{ display: none; }}
    .tab-content {{ display: block !important; border: none; page-break-inside: avoid; }}
    .chart-card {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Data Quality Report</h1>
    <div class="meta">{project_name} &nbsp;|&nbsp; Generated: {date}</div>
  </div>

  <div class="kpi-row">
    <div class="kpi total"><div class="number">{total}</div><div class="label">Total Findings</div></div>
    <div class="kpi high"><div class="number">{high}</div><div class="label">High Impact</div></div>
    <div class="kpi med"><div class="number">{med}</div><div class="label">Medium Impact</div></div>
    <div class="kpi low"><div class="number">{low}</div><div class="label">Low Impact</div></div>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('issues')">Issues</div>
    <div class="tab" onclick="switchTab('summary')">Summary</div>
    <div class="tab" onclick="switchTab('charts')">Charts</div>
  </div>

  <!-- Issues Tab -->
  <div class="tab-content active" id="tab-issues">
    <div class="filters">
      <label>Impact:</label>
      <select id="filter-impact" onchange="filterRows()">
        <option value="all">All</option>
        <option value="High">High</option>
        <option value="Medium">Medium</option>
        <option value="Low">Low</option>
      </select>
      <label>Parameter:</label>
      <select id="filter-param" onchange="filterRows()">
        <option value="all">All</option>
        {param_options}
      </select>
      <label>Check:</label>
      <select id="filter-check" onchange="filterRows()">
        <option value="all">All</option>
        {check_options}
      </select>
      <span id="row-count" style="font-size:13px;color:#7f8c8d;margin-left:auto;"></span>
    </div>
    <div style="overflow-x:auto;">
    <table>
      <thead>
        <tr><th>Impact</th><th>Check</th><th>Variable</th><th>Parameter</th><th>Question / Observation</th><th>Downstream</th><th>Chart</th></tr>
      </thead>
      <tbody id="findings-body">
        {rows}
      </tbody>
    </table>
    </div>
  </div>

  <!-- Summary Tab -->
  <div class="tab-content" id="tab-summary">
    <h2 style="margin-bottom:16px;font-size:18px;">Findings by Parameter &times; Impact</h2>
    <table class="summary-table">
      <thead><tr><th>Parameter</th><th>High</th><th>Medium</th><th>Low</th><th>Total</th></tr></thead>
      <tbody>{summary_table}</tbody>
    </table>
  </div>

  <!-- Charts Tab -->
  <div class="tab-content" id="tab-charts">
    <div class="charts-grid">
      {charts}
    </div>
  </div>

</div>

<script>
function switchTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}

function filterRows() {{
  const impact = document.getElementById('filter-impact').value;
  const param = document.getElementById('filter-param').value;
  const check = document.getElementById('filter-check').value;
  let visible = 0;
  document.querySelectorAll('.finding-row').forEach(row => {{
    const show = (impact === 'all' || row.dataset.impact === impact)
              && (param === 'all' || row.dataset.param === param)
              && (check === 'all' || row.dataset.check === check);
    row.classList.toggle('hidden', !show);
    if (show) visible++;
  }});
  document.getElementById('row-count').textContent = visible + ' / ' + document.querySelectorAll('.finding-row').length + ' findings';
}}
filterRows();
</script>
</body>
</html>"""
