"""
Phase II v2 - Retrain forecaster on the SKU x Warehouse (Sub-Category x Region) panel.
Same leakage-free feature discipline as the original: no same-week contemporaneous features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json
import pickle

panel = pd.read_csv("weekly_panel_v2.csv", parse_dates=["week"])

le_cat = LabelEncoder()
le_region = LabelEncoder()
panel["subcat_enc"] = le_cat.fit_transform(panel["Sub-Category"])
panel["region_enc"] = le_region.fit_transform(panel["Region"])

features = [
    "subcat_enc", "region_enc",
    "units_lag1", "units_lag2", "units_lag4",
    "units_roll4_mean", "units_roll4_std",
    "week_of_year", "month", "quarter",
]
target = "units"

panel = panel.sort_values("week")
cutoff = panel["week"].quantile(0.85)
train = panel[panel["week"] <= cutoff]
test = panel[panel["week"] > cutoff]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, reg_lambda=1.0,
)
model.fit(X_train, y_train)
pred = np.clip(model.predict(X_test), 0, None)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"Train: {len(train)} rows, Test: {len(test)} rows")
print(f"XGBoost (v2, SKU x Warehouse): MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")

with open("forecast_results_v2.json", "w") as f:
    json.dump({"mae": mae, "rmse": rmse, "r2": r2, "n_train": len(train), "n_test": len(test)}, f, indent=2)

model.save_model("xgb_forecast_v2.json")
with open("label_encoders_v2.pkl", "wb") as f:
    pickle.dump({"subcat": le_cat, "region": le_region}, f)

print("Saved xgb_forecast_v2.json, label_encoders_v2.pkl, forecast_results_v2.json")
