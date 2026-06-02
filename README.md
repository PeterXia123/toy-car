# EDA Toolkit

Automated data quality checks, consistency analysis, and trend monitoring with HTML report and Excel issue log generation.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run with a project config
python run.py --project config/projects/mx_mg_test.yaml

# Run specific check categories only
python run.py --project config/projects/mx_mg_test.yaml --only data_quality,consistency

# List all available checks
python run.py --list-checks
```

## Options

| Flag | Description |
|------|-------------|
| `--project` | Path to project YAML config (required) |
| `--checks` | Path to checks YAML config (default: `config/checks.yaml`) |
| `--variables` | Path to variables YAML config (default: `config/variables.yaml`) |
| `--only` | Comma-separated categories: `data_quality`, `consistency`, `score_alignment`, `trends`, `account_tracking`, `term_checks`, `revolving_checks` |
| `--list-checks` | List all available checks and exit |

## Output

Reports are generated in the output directory specified in the project config:

- `Issue_Log_<project>.xlsx` — Excel issue log
- `Issue_Log_<project>.html` — Interactive HTML report with charts
- `charts/` — Generated chart images

## Project Config

See `config/projects/` for examples. Key settings:

```yaml
project:
  name: "Project Name"
  product: "Product"
  product_type: term  # or revolving

data:
  path: "path/to/data/"
  file: "data_file.parquet"
  format: parquet
```
