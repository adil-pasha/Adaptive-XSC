# Setup & Run

Python 3.12 or 3.13 confirmed working. **Avoid Python 3.14** — `shap`'s
native dependencies (`numba`, `llvmlite`) don't yet have stable prebuilt
wheels for it; this caused the original deployment failures (see
`docs/PROJECT_DOCUMENTATION.md` Section 2).

## 1. Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data

- `data/superstore.csv` is included in this repo.
- `data/DataCoSupplyChainDataset.csv` (91MB, 180,519 rows) is **not**
  included due to size. Download it:
  ```bash
  curl -L -o data/DataCoSupplyChainDataset.csv \
    https://raw.githubusercontent.com/ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS/main/DataCoSupplyChainDataset.csv
  ```
  For the paper, cite the original source, not this mirror:
  Constante, Silva & Pereira (2019), Mendeley Data, DOI: `10.17632/8gx2fvg2k6.5`.

## 3. Run

From the repo root:

```bash
cd src
python etl_demand.py          # -> ../data/weekly_panel.csv
python train_forecast.py      # -> ../results/forecast_results.json
python explain_shap.py        # -> ../results/shap_explanations.json
python train_risk_v2.py       # -> ../results/risk_results.json
python explain_risk_shap.py   # -> ../results/risk_shap_explanations.json
```

Note: the scripts currently read/write relative to the working directory
they're run from. If you keep everything in `src/`, either run from inside
`src/` and adjust the CSV read paths to `../data/...`, or copy the CSVs into
`src/` before running. (Left this simple deliberately — tidy up the paths
however fits your repo conventions.)

## 4. Expected results

| Model | Metric | Value |
|---|---|---|
| XGBoost forecaster | R² | ~0.57 |
| Naive baseline | R² | ~0.11 |
| Logistic regression risk | AUC | ~0.73 |
| XGBoost risk | AUC | ~0.74 |

Small variation is expected from library version drift.

## 5. View the demo

Open `app/index.html` directly in any browser — no server needed. To publish
it: push to GitHub, then Settings → Pages → Deploy from branch → `main`,
folder `/app`.
