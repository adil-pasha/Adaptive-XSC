# AdaptiveXSC — End-to-End Project Documentation

This document covers the project from initialization to its current
production-ready state. It's written to double as source material for a
research paper's Methodology, Results, and Limitations sections.

---

## 1. Project Goal

AdaptiveXSC is a demand-forecasting and delivery-risk platform aimed at being
accessible to mid-market operations teams without a data-science budget. Its
core research contribution is **not** the forecasting model itself, but the
**override/audit log**: a mechanism that captures *why* a human manager
accepts, rejects, or overrides an AI recommendation, building a
trust-calibration dataset over time.

Three pillars, in order of how they support that goal:
1. A forecasting/risk model good enough to be worth trusting sometimes.
2. Explainability (SHAP), so the manager has grounds to evaluate the
   recommendation rather than accept/reject blindly.
3. The override log itself, which is the actual novel artifact.

---

## 2. History of this project's build (useful for a paper's "lessons learned")

1. **Initial deployment target: Streamlit Community Cloud.** The app repeatedly
   failed to start. Root cause: Streamlit Cloud was provisioning **Python
   3.14**, a version too new for `shap`'s native-compiled dependencies
   (`numba`, `llvmlite`) to have stable prebuilt wheels for. `runtime.txt`
   pins are only respected at first deploy, not on redeploy of an existing app.
2. **Moved to Render.** Build succeeded (Render's build environment handled
   the same dependencies fine), but the running service crashed with **exit
   code 139 (SIGSEGV)** — a native-level segfault, most likely triggered when
   a request invoked the numba JIT path inside `shap`, again a Python-version/
   native-library interaction issue.
3. **Decision: separate the research pipeline from the deployed demo.** Real
   ML work moved into a controlled sandbox running **Python 3.12** (later
   confirmed Python 3.13.7 also works, since `shap` ships a `cp312-abi3`
   wheel compatible with 3.12+). The deployed-facing demo became a static,
   dependency-free HTML/JS page that displays **precomputed real model
   outputs** rather than doing live inference in the browser or on a fragile
   server.

This history matters for the paper: it's a legitimate methodological point
that bleeding-edge language runtimes are a real, underdiscussed risk for
reproducible ML research infrastructure — you hit it, diagnosed it, and
engineered around it.

---

## 3. Data Sources

### 3.1 Demand forecasting: Sample Superstore
- 9,994 valid order-line transactions (806 rows with corrupted/missing order
  dates were dropped from an original 10,800).
- Standard, widely-used retail benchmark dataset (originally distributed via
  Tableau/Kaggle as "Sample - Superstore").
- Columns used: Order Date, Sub-Category, Quantity, Sales, Discount, Profit,
  Order ID.

### 3.2 Delivery/stockout risk: DataCo Smart Supply Chain
- 180,519 real logged orders.
- **Citation:** Constante, F., Silva, F., & Pereira, A. (2019). *DataCo Smart
  Supply Chain for Big Data Analysis*. Mendeley Data, V5.
  DOI: `10.17632/8gx2fvg2k6.5`. Cite this in the paper — not the GitHub
  mirror used to obtain the file during development.
- Target variable: `Late_delivery_risk` (binary), naturally balanced at
  54.8% late / 45.2% on-time.

### 3.3 Rejected dataset: "Supply Chain Analysis" (100-row synthetic)
- Initially used for the risk model. Produced near-chance performance
  (accuracy ~48%, AUC ~0.39) because its columns are independently randomly
  generated with no real causal relationships between features and the
  stockout label. **This is documented as a finding, not hidden** — it's
  legitimate evidence that off-the-shelf synthetic demo datasets can lack
  the structure needed for risk modeling research, and is a defensible point
  to make in a Limitations or Data section.

---

## 4. Methodology

### 4.1 Demand Forecasting Pipeline
**ETL (`src/etl_demand.py`):**
- Aggregate transactions to weekly demand per Sub-Category.
- Reindex to a complete weekly calendar per category (fill true zero-demand
  gaps rather than silently skipping them).
- Feature engineering: lags (1/2/4 weeks), 4-week rolling mean/std, calendar
  features (week-of-year, month, quarter).

**Model (`src/train_forecast.py`):**
- XGBoost regressor (300 trees, depth 4, learning rate 0.05, subsample 0.8,
  colsample 0.8).
- **Time-based train/test split** (not random) — trained on data up to the
  85th percentile week, tested on all later weeks. This is the methodologically
  correct approach for forecasting; a random split would leak future
  information into training and inflate apparent accuracy.
- Baselines: naive seasonal lag-4, and rolling 4-week mean.

**A real leakage bug found and fixed during development:** an initial feature
set included `n_orders` and `discount` for the *current* week being
predicted — both are only knowable after the week happens, so including them
let the model implicitly see the future. Removing them dropped R² from an
inflated ~0.89 to a real, defensible **0.568** — still clearly ahead of both
baselines. This is worth including in a paper's methodology section as
evidence of careful validation practice.

**Explainability (`src/explain_shap.py`):**
- `shap.TreeExplainer`, exact (not approximate) SHAP values for tree models.
- Global importance and five worked per-prediction examples saved to
  `results/shap_explanations.json`.

### 4.2 Delivery/Stockout Risk Pipeline
**Model (`src/train_risk_v2.py`):**
- Logistic Regression (standardized features) and XGBoost classifier (200
  trees, depth 4), 80/20 stratified split.
- **Leakage avoidance:** `Days for shipping (real)` and `Delivery Status` are
  excluded — both are only known after delivery completes and are near-direct
  proxies for the label. PII columns (names, emails, passwords, addresses)
  and high-cardinality ID/free-text columns are also excluded.

**Explainability (`src/explain_risk_shap.py`):**
- Real `TreeExplainer` SHAP values on the trained XGBoost classifier.
- Global importance led by `Days for shipment (scheduled)` and `Shipping
  Mode` — orders with short scheduled windows on slower shipping modes are
  most likely to be late. This matches domain intuition and independently
  published analyses of the same dataset (see Section 6).

---

## 5. Results

### 5.1 Demand Forecasting (held-out future weeks)

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| **XGBoost** | **8.10** | **12.06** | **0.568** | 68.9% |
| Naive (lag-4 seasonal) | 11.37 | 17.28 | 0.113 | 98.2% |
| Rolling-mean baseline | 9.13 | 13.21 | 0.482 | 79.9% |

MAPE is high in absolute terms because several Sub-Categories have low,
volatile weekly unit counts (MAPE is known to be unstable near low/zero
values); MAE/RMSE/R² are the more reliable metrics here and consistently
favor XGBoost.

### 5.2 Delivery/Stockout Risk (held-out test set, n=36,104)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.692 | 0.797 | 0.588 | 0.677 | 0.726 |
| **XGBoost** | **0.697** | **0.835** | 0.557 | 0.668 | **0.745** |

Both models perform well above chance (AUC 0.5), with no leakage, and are
consistent with independently published results using the same dataset and a
similar leakage-avoidance approach.

---

## 6. Reproducibility

Full setup and run instructions: see `SETUP.md`. In short:

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap
python src/etl_demand.py
python src/train_forecast.py
python src/explain_shap.py
python src/train_risk_v2.py
python src/explain_risk_shap.py
```

Python 3.12 and 3.13 both confirmed working. **Avoid Python 3.14** for this
stack until `numba`/`llvmlite`/`shap` publish stable wheels for it — this was
the direct cause of the original Streamlit Cloud deployment failures.

---

## 7. Deployment

- **Research pipeline**: runs locally or in any standard Python 3.12/3.13
  environment (see above). Not deployed as a live service — it's a batch
  training/evaluation pipeline, appropriately.
- **App** (`src/streamlit_app.py`): a real Streamlit web app that loads the
  actual trained model files and computes SHAP live on every interaction.
  An earlier static-HTML demo version (with precomputed placeholder outputs)
  was removed in favor of this real, running application.
- Earlier attempts to deploy the full `shap`/`xgboost` stack to Streamlit
  Community Cloud and Render ran into the native-library/Python-version
  issues in Section 2. Running the app locally (or on any host you control,
  pinned to Python 3.12/3.13) avoids those platform-specific failures
  entirely — the app has no external service dependency.

---

## 8. Limitations (for the paper's Limitations section)

- Superstore demand data is retail sales, aggregated to category level, not
  true per-SKU inventory-level demand — a reasonable proxy but not identical
  to real SKU-level replenishment data.
- MAPE is unstable for the forecasting model due to low-count weeks; readers
  should weight MAE/RMSE/R² more heavily.
- The risk classifier's recall (0.557–0.588) means a meaningful fraction of
  genuinely late orders are not flagged — precision is stronger (0.80–0.84).
  Worth discussing the operational cost of false negatives vs. false
  positives in a supply-chain context.
- The override/audit log's real-world validation (does it actually improve
  manager trust calibration over time?) has not yet been tested with real
  users — it's a mechanism, demonstrated to work, not yet an evaluated
  intervention. This is an honest, statable direction for future work.

---

## 9. File Map

```
repo/
├── src/
│   ├── streamlit_app.py        # The real app — run this
│   ├── etl_demand_v2.py        # SKU x Warehouse panel (used by the app)
│   ├── train_forecast_v2.py    # XGBoost forecaster, SKU x Warehouse (used by the app)
│   ├── weekly_panel_v2.csv     # Generated panel (committed — app needs this to run)
│   ├── xgb_forecast_v2.json    # Trained model (committed — app needs this to run)
│   ├── label_encoders_v2.pkl   # Encoders (committed — app needs this to run)
│   ├── superstore.csv          # Included (2.3MB)
│   ├── etl_demand.py           # Original Sub-Category-only pipeline, kept for comparison
│   ├── train_forecast.py       # Original coarser forecaster (R²=0.568)
│   ├── explain_shap.py         # Real SHAP for the original forecaster
│   ├── train_risk_v2.py        # Logistic regression + XGBoost delivery-risk classifier
│   │                            # (needs DataCoSupplyChainDataset.csv — see SETUP.md;
│   │                            #  not currently used by streamlit_app.py)
│   └── explain_risk_shap.py    # Real SHAP for the risk classifier
├── results/
│   ├── forecast_results.json
│   ├── risk_results.json
│   ├── shap_explanations.json
│   └── risk_shap_explanations.json
├── docs/
│   └── PROJECT_DOCUMENTATION.md  # This file
├── SETUP.md
├── requirements.txt
├── .gitignore
└── README.md
```
