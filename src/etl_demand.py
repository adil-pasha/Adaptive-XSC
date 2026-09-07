"""
Phase I - ETL & Data Foundation
Ingests Superstore transactional data, aggregates to weekly demand
per Sub-Category (proxy for SKU-family demand), and builds engineered
features suitable for a regression forecaster.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("superstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y", errors="coerce")
n_before = len(df)
df = df.dropna(subset=["Order Date"]).reset_index(drop=True)
print(f"Dropped {n_before - len(df)} rows with missing Order Date ({len(df)} remain)")

# Aggregate to weekly demand per Sub-Category (this becomes our "SKU family")
df["week"] = df["Order Date"].dt.to_period("W").apply(lambda p: p.start_time)

weekly = (
    df.groupby(["Sub-Category", "week"])
    .agg(units=("Quantity", "sum"),
         sales=("Sales", "sum"),
         discount=("Discount", "mean"),
         profit=("Profit", "sum"),
         n_orders=("Order ID", "nunique"))
    .reset_index()
)

# Fill missing weeks per sub-category with 0 demand (real gaps happen in retail data)
full_frames = []
for cat, g in weekly.groupby("Sub-Category"):
    g = g.set_index("week").sort_index()
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON")
    g = g.reindex(full_idx)
    g["Sub-Category"] = cat
    g[["units", "sales", "profit", "n_orders"]] = g[["units", "sales", "profit", "n_orders"]].fillna(0)
    g["discount"] = g["discount"].fillna(0)
    full_frames.append(g)

panel = pd.concat(full_frames).reset_index().rename(columns={"index": "week"})

# Feature engineering: lags, rolling stats, calendar features
panel = panel.sort_values(["Sub-Category", "week"])
for lag in [1, 2, 4]:
    panel[f"units_lag{lag}"] = panel.groupby("Sub-Category")["units"].shift(lag)
panel["units_roll4_mean"] = panel.groupby("Sub-Category")["units"].transform(lambda s: s.shift(1).rolling(4).mean())
panel["units_roll4_std"] = panel.groupby("Sub-Category")["units"].transform(lambda s: s.shift(1).rolling(4).std())
panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
panel["month"] = panel["week"].dt.month
panel["quarter"] = panel["week"].dt.quarter

panel = panel.dropna().reset_index(drop=True)

panel.to_csv("weekly_panel.csv", index=False)
print("Panel shape:", panel.shape)
print("Sub-categories:", panel["Sub-Category"].nunique())
print(panel.head(8).to_string())
