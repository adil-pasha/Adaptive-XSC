"""
Phase II - Models & Explainability (XAI): Forecasting
Trains XGBoost regressor to predict next-week units demand per Sub-Category.
Compares against a naive seasonal-lag baseline (Prophet-style comparator,
simplified since Prophet needs its own heavy install).
Uses a TIME-BASED split (not random) since this is forecasting, not iid data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json

panel = pd.read_csv("weekly_panel.csv", parse_dates=["week"])

le = LabelEncoder()
panel["subcat_enc"] = le.fit_transform(panel["Sub-Category"])

# IMPORTANT: only lagged/calendar features are used. `n_orders` and `discount`
# for the CURRENT week are contemporaneous with the target and would leak
# information a real forecast wouldn't have in advance (you don't know this
# week's order count before the week happens). Excluding them.
features = [
    "subcat_enc",
    "units_lag1", "units_lag2", "units_lag4",
    "units_roll4_mean", "units_roll4_std",
    "week_of_year", "month", "quarter",
]
target = "units"

# Time-based split: last 15% of weeks per sub-category = test set
panel = panel.sort_values("week")
cutoff = panel["week"].quantile(0.85)
train = panel[panel["week"] <= cutoff]
test = panel[panel["week"] > cutoff]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

print(f"Train: {len(train)} rows (up to {cutoff.date()})")
print(f"Test:  {len(test)} rows (after {cutoff.date()})")

# --- XGBoost model ---
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    reg_lambda=1.0,
)
model.fit(X_train, y_train)
pred_xgb = np.clip(model.predict(X_test), 0, None)

# --- Naive baseline: predict this week = same week last cycle (units_lag4) ---
pred_naive = X_test["units_lag4"].values

# --- Rolling-mean baseline (a simple, honest "classical" comparator) ---
pred_rollmean = X_test["units_roll4_mean"].values

def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE, guarding against zero actuals
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() else float("nan")
    print(f"{name:22s}  MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:6.3f}  MAPE={mape:6.1f}%")
    return dict(model=name, mae=mae, rmse=rmse, r2=r2, mape=mape)

print("\n--- Forecast accuracy (held-out future weeks) ---")
results = [
    report("XGBoost", y_test.values, pred_xgb),
    report("Naive (lag-4 seasonal)", y_test.values, pred_naive),
    report("Rolling-mean baseline", y_test.values, pred_rollmean),
]

with open("forecast_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save model + encoder + test set for the SHAP step
model.save_model("xgb_forecast.json")
test.to_csv("test_set.csv", index=False)
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nSaved: xgb_forecast.json, test_set.csv, label_encoder.pkl, forecast_results.json")
