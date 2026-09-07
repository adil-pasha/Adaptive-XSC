"""
Phase II - Explainability (XAI) for the risk classifier
Real SHAP TreeExplainer values on the trained XGBoost late-delivery model.
"""
import pandas as pd
import numpy as np
import shap
import pickle
import json

with open("risk_models.pkl", "rb") as f:
    saved = pickle.load(f)

xgb_clf = saved["xgb"]
features = saved["features"]

test = pd.read_csv("risk_test_set.csv")
X_test = test[features]

explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer(X_test)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance = sorted(zip(features, mean_abs_shap), key=lambda x: -x[1])

print("--- Global feature importance (mean |SHAP|), risk model ---")
for feat, val in importance[:15]:
    print(f"{feat:30s} {val:6.4f}")

# Worked examples: a few high-confidence "at risk" and "not at risk" predictions
test_sorted = test.assign(row_pos=range(len(test)))
high_risk = test_sorted.sort_values("proba_xgb", ascending=False).head(3)
low_risk = test_sorted.sort_values("proba_xgb", ascending=True).head(3)

def build_example(row):
    pos = int(row["row_pos"])
    sv = shap_values.values[pos]
    base = float(shap_values.base_values[pos])
    contribs = sorted(zip(features, sv), key=lambda x: -abs(x[1]))[:8]
    return {
        "actual_late": int(row["y_true"]),
        "predicted_proba_late": float(row["proba_xgb"]),
        "base_value": base,
        "top_contributions": [{"feature": f, "shap_value": float(v)} for f, v in contribs],
    }

examples = [build_example(r) for _, r in pd.concat([high_risk, low_risk]).iterrows()]

with open("risk_shap_explanations.json", "w") as f:
    json.dump({
        "global_importance": [{"feature": f, "mean_abs_shap": float(v)} for f, v in importance],
        "sample_explanations": examples,
    }, f, indent=2)

print("\nSaved risk_shap_explanations.json")
print("\n--- Example: highest-confidence 'at risk' prediction ---")
print(json.dumps(examples[0], indent=2))
