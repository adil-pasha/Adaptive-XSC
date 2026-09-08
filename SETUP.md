# Setup & Run

Python 3.12 or 3.13 confirmed working. **Avoid Python 3.14** — `shap`'s
native dependencies (`numba`, `llvmlite`) don't yet have stable prebuilt
wheels for it; this caused the original Streamlit Cloud deployment failures
(see `docs/PROJECT_DOCUMENTATION.md` Section 2). Note: 3.14 has since been
confirmed to work in at least one local test — but 3.12/3.13 remain the
safer, tested choice.

## 1. Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data

All scripts and the app expect data files to sit alongside them in `src/`.

- `superstore.csv` is included in `src/`.
- `DataCoSupplyChainDataset.csv` (91MB, 180,519 rows) is **not** included
  due to size — needed only for the delivery-risk model. Download it into
  `src/`:
  ```bash
  cd src
  curl -L -o DataCoSupplyChainDataset.csv \
    https://raw.githubusercontent.com/ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS/main/DataCoSupplyChainDataset.csv
  ```
  (PowerShell without curl.exe available: use `Invoke-WebRequest -Uri <url> -OutFile DataCoSupplyChainDataset.csv`)

  For the paper, cite the original source, not this mirror:
  Constante, Silva & Pereira (2019), Mendeley Data, DOI: `10.17632/8gx2fvg2k6.5`.

## 3. Run the pipeline

All commands run from inside `src/`:

```bash
cd src
python etl_demand_v2.py       # -> weekly_panel_v2.csv (SKU x Warehouse panel)
python train_forecast_v2.py   # -> xgb_forecast_v2.json, label_encoders_v2.pkl
python train_risk_v2.py       # -> risk_models.pkl, risk_test_set.csv (needs DataCo CSV, step 2)
```

The older, coarser-grained scripts (`etl_demand.py`, `train_forecast.py`,
`explain_shap.py`, `explain_risk_shap.py`) still work and produce the
Sub-Category-only model (R²=0.568) referenced in `docs/PROJECT_DOCUMENTATION.md`
Section 5 — kept for comparison, not required to run the app.

## 4. Expected results

| Model | Metric | Value |
|---|---|---|
| XGBoost forecaster (SKU x Warehouse, v2 — used by the app) | R² | ~0.29 |
| XGBoost forecaster (Sub-Category only, v1) | R² | ~0.57 |
| Logistic regression risk | AUC | ~0.73 |
| XGBoost risk | AUC | ~0.74 |

The v2 model's lower R² is an expected, documented bias-variance tradeoff
from finer-grained grouping (68 SKU x Warehouse combinations vs. 17
Sub-Categories) — see `docs/PROJECT_DOCUMENTATION.md`. Small variation
across runs is expected from library version drift.

## 5. Run the app

```bash
cd src   # if not already there
streamlit run streamlit_app.py
```

Opens a single-page dashboard:
- **Observation Selection** — cascading Region (Warehouse) → Sub-Category
  (SKU) → Week selector over the real weekly panel.
- **Baseline Forecast** — real XGBoost prediction vs. actual, with a real
  SHAP bar chart explaining the top drivers.
- **What-If Scenario Analysis** — sliders perturb the model's actual inputs
  (recent demand), recomputing forecast + decision live.
- **Rule-based Decision Engine** — NORMAL / WATCH / LOW STOCK / CRITICAL
  STOCK, based on a transparent (s,S) inventory-policy formula applied to
  real demand statistics (not observed inventory data — Superstore has none;
  see in-app sidebar note). Forecast pressure escalates severity.
- **Human Decision** — Accept / Reject / Override (with reason), logged to
  `src/decision_log.csv` with an Evaluator ID — the trust-calibration
  dataset.
- **Evaluator Feedback** — logged to `src/feedback_log.csv`.
- **Data Collection Monitor** (sidebar) — live counts from the decision log.

Nothing in the app is precomputed or simulated — every forecast, SHAP value,
and decision-state calculation happens live against the real trained models.
