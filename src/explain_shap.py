"""
Phase II - Explainability (XAI)
Computes REAL SHAP values (TreeExplainer, exact for XGBoost) on the
trained forecaster's held-out predictions. No hand-picked coefficients.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
import pickle

model = xgb.XGBRegressor()
model.load_model("xgb_forecast.json")

test = pd.read_csv("test_set.csv", parse_dates=["week"])
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

features = [
    "subcat_enc",
    "units_lag1", "units_lag2", "units_lag4",
    "units_roll4_mean", "units_roll4_std",
    "week_of_year", "month", "quarter",
]
X_test = test[features]

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Global feature importance (mean |SHAP value|)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance = sorted(zip(features, mean_abs_shap), key=lambda x: -x[1])

print("--- Global feature importance (mean |SHAP|) ---")
for feat, val in importance:
    print(f"{feat:20s} {val:6.3f}")

# Per-row explanation for a handful of specific predictions (what the app UI will show)
sample_idx = test.sort_values("week", ascending=False).index[:5]
explanations = []
for i in sample_idx:
    row_pos = test.index.get_loc(i)
    row = test.loc[i]
    sv = shap_values.values[row_pos]
    base = float(shap_values.base_values[row_pos])
    contribs = sorted(zip(features, sv), key=lambda x: -abs(x[1]))
    explanations.append({
        "sub_category": row["Sub-Category"],
        "week": str(row["week"].date()),
        "actual_units": float(row["units"]),
        "predicted_units": float(base + sv.sum()),
        "base_value": base,
        "contributions": [{"feature": f, "shap_value": float(v)} for f, v in contribs],
    })

with open("shap_explanations.json", "w") as f:
    json.dump({
        "global_importance": [{"feature": f, "mean_abs_shap": float(v)} for f, v in importance],
        "sample_explanations": explanations,
    }, f, indent=2)

print("\nSaved shap_explanations.json with", len(explanations), "worked examples")
print("\n--- Example explanation ---")
print(json.dumps(explanations[0], indent=2))
