# AdaptiveXSC

Demand forecasting and delivery-risk platform with an explainable,
override-logged decision layer — built for mid-market operations teams.

**Live demo:** open `app/index.html` directly, or deploy via GitHub Pages
(Settings → Pages → Deploy from branch → `main` → `/app`).

**Full write-up:** see [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md)
for complete methodology, results, and reproducibility instructions —
written to double as source material for the accompanying research paper.

## Quick facts

| | |
|---|---|
| Demand forecaster | XGBoost, R²=0.568 vs. 0.113 (naive) / 0.482 (rolling-mean) baselines |
| Delivery risk classifier | XGBoost, AUC=0.745; Logistic Regression, AUC=0.726 |
| Explainability | Real SHAP (`TreeExplainer`) on both models |
| Datasets | Sample Superstore (9,994 txns) + DataCo Smart Supply Chain (180,519 orders) |
| Python | 3.12 / 3.13 confirmed working. Avoid 3.14 (see docs) |

## Repo structure

```
src/      real training/explainability scripts
data/     superstore.csv (DataCo file too large — see SETUP.md)
results/  metrics + SHAP outputs, JSON
app/      static demo (no backend, no native deps)
docs/     full project documentation
```

## Setup

See [`SETUP.md`](SETUP.md).
