"""
Phase I v2 - ETL with SKU x Warehouse granularity
Extends the original weekly panel to group by (Sub-Category, Region) instead
of Sub-Category alone, so the app can offer a genuine SKU x Warehouse
observation selector like the original design intended.

Region is a REAL column in Superstore (South/West/Central/East) — used here
as the Warehouse-equivalent dimension. Unit_Price and Unit_Cost are REAL
derived quantities (Sales/Quantity, (Sales-Profit)/Quantity) — not invented.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("superstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["Order Date", "Region"]).reset_index(drop=True)

df["week"] = df["Order Date"].dt.to_period("W").apply(lambda p: p.start_time)

# Real derived per-unit price/cost (guard against zero quantity)
df["line_price"] = np.where(df["Quantity"] > 0, df["Sales"] / df["Quantity"], np.nan)
df["line_cost"] = np.where(df["Quantity"] > 0, (df["Sales"] - df["Profit"]) / df["Quantity"], np.nan)

weekly = (
    df.groupby(["Sub-Category", "Region", "week"])
    .agg(units=("Quantity", "sum"),
         sales=("Sales", "sum"),
         discount=("Discount", "mean"),
         profit=("Profit", "sum"),
         n_orders=("Order ID", "nunique"),
         unit_price=("line_price", "mean"),
         unit_cost=("line_cost", "mean"))
    .reset_index()
)

full_frames = []
for (cat, region), g in weekly.groupby(["Sub-Category", "Region"]):
    g = g.set_index("week").sort_index()
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON")
    g = g.reindex(full_idx)
    g["Sub-Category"] = cat
    g["Region"] = region
    g[["units", "sales", "profit", "n_orders"]] = g[["units", "sales", "profit", "n_orders"]].fillna(0)
    g["discount"] = g["discount"].fillna(0)
    g[["unit_price", "unit_cost"]] = g[["unit_price", "unit_cost"]].ffill().bfill()
    full_frames.append(g)

panel = pd.concat(full_frames).reset_index().rename(columns={"index": "week"})
panel = panel.sort_values(["Sub-Category", "Region", "week"])

for lag in [1, 2, 4]:
    panel[f"units_lag{lag}"] = panel.groupby(["Sub-Category", "Region"])["units"].shift(lag)
panel["units_roll4_mean"] = panel.groupby(["Sub-Category", "Region"])["units"].transform(
    lambda s: s.shift(1).rolling(4).mean())
panel["units_roll4_std"] = panel.groupby(["Sub-Category", "Region"])["units"].transform(
    lambda s: s.shift(1).rolling(4).std())
panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
panel["month"] = panel["week"].dt.month
panel["quarter"] = panel["week"].dt.quarter

# --- Rule-based inventory policy layer (explicitly NOT machine-learned) ---
# Deterministic base-stock formula from REAL observed demand statistics:
#   Reorder_Point = lead_time * rolling_mean + safety_stock
#   safety_stock  = z * rolling_std * sqrt(lead_time)
# This is standard inventory-theory (s,S)-policy math, not a prediction.
# Superstore has no real inventory records, so this layer is clearly labeled
# as an illustrative policy simulation in the app, not observed inventory data.
LEAD_TIME_WEEKS = 2
Z_SAFETY = 1.65  # ~95% service level
panel["reorder_point"] = (
    LEAD_TIME_WEEKS * panel["units_roll4_mean"].fillna(0)
    + Z_SAFETY * panel["units_roll4_std"].fillna(0) * np.sqrt(LEAD_TIME_WEEKS)
).round(1)
# Illustrative on-hand inventory: target coverage of 3x the rolling mean,
# net of an assumed order placed one lead-time ago (deterministic, no randomness)
COVERAGE_WEEKS = 3
panel["inventory_level"] = (COVERAGE_WEEKS * panel["units_roll4_mean"].fillna(0)).round(1)
panel["order_quantity"] = 0.0  # snapshot assumption: no open order at observation time

panel = panel.dropna(subset=["units_lag1", "units_lag2", "units_lag4", "units_roll4_mean", "units_roll4_std"]).reset_index(drop=True)

panel.to_csv("weekly_panel_v2.csv", index=False)
print("Panel shape:", panel.shape)
print("Sub-Category x Region combos:", panel.groupby(["Sub-Category","Region"]).ngroups)
print(panel.head(5).to_string())
