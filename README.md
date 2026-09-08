# AdaptiveXSC

Demand forecasting and delivery-risk platform with an explainable,
override-logged decision layer — built for mid-market operations teams.

**Run the app:** see Quick Start below. This is a real Streamlit web app
backed by trained XGBoost models and live SHAP explanations — not a static
demo.

**Full write-up:** see [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md)
for complete methodology, results, and reproducibility instructions —
written to double as source material for the accompanying research paper.

## Quick facts

| | |
|---|---|
| Demand forecaster | XGBoost, SKU x Warehouse granularity, R²=0.294 (finer-grained model) / R²=0.568 (coarser Sub-Category-only model) |
| Delivery risk classifier | XGBoost, AUC=0.745; Logistic Regression, AUC=0.726 |
| Explainability | Real SHAP (`TreeExplainer`) on all models, computed live in the app |
| Datasets | Sample Superstore (9,994 txns, real Region column used as Warehouse) + DataCo Smart Supply Chain (180,519 orders) |
| Decision engine | Rule-based (s,S) inventory policy — explicitly not machine-learned, clearly labeled in-app |
| Python | 3.12 / 3.13 confirmed working. Avoid 3.14 (see docs) |

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd src
python etl_demand_v2.py
python train_forecast_v2.py
python train_risk_v2.py        # needs DataCoSupplyChainDataset.csv — see SETUP.md
cd ..

streamlit run src/streamlit_app.py
```

See [`SETUP.md`](SETUP.md) for full details, including the app's three
sections (observation selector, SHAP explanation, what-if simulator +
rule-based decision engine, human decision + audit log).

## Repo structure

```
src/      training/explainability scripts, trained model artifacts, the app itself
data/     superstore.csv (DataCo file too large — see SETUP.md)
results/  metrics + SHAP outputs, JSON
docs/     full project documentation
```

