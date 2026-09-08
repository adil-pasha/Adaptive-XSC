"""
AdaptiveXSC v2 — rebuilt with the architecture from the original Dash app
(observation selector, SHAP explanation, what-if simulator, rule-based
decision engine, evaluator ID + audit logging, collection monitor),
ported to Streamlit and backed entirely by real trained models.

Honesty notes (read these before treating any number as ground truth):
- Forecast: REAL XGBoost model, retrained on a SKU (Sub-Category) x Warehouse
  (Region — a real Superstore column) panel. R2=0.294 at this granularity --
  lower than the coarser Sub-Category-only model (R2=0.568) because finer
  grouping means less data per group. This is an expected, documented
  bias-variance tradeoff, not a bug.
- Unit_Price / Unit_Cost: REAL derived quantities (Sales/Quantity,
  (Sales-Profit)/Quantity), not invented.
- Inventory_Level / Reorder_Point: Superstore has NO real inventory data.
  These are computed from a standard, deterministic (s,S) base-stock
  formula applied to REAL observed demand statistics -- a transparent rule
  layer, not a model prediction and not fabricated data. Labeled as such
  throughout the UI.
- What-If sliders perturb the model's ACTUAL inputs (recent demand lags),
  not price/promotion -- because price/promotion are not features this
  forecaster was trained on.

Run from repo root:  streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.graph_objects as go
import pickle
import datetime
from pathlib import Path

st.set_page_config(page_title="AdaptiveXSC", page_icon="📦", layout="wide")

BASE = Path(__file__).resolve().parent
SRC = BASE
LOG_PATH = BASE / "decision_log.csv"
FEEDBACK_PATH = BASE / "feedback_log.csv"

FORECAST_FEATURES = ["subcat_enc", "region_enc", "units_lag1", "units_lag2", "units_lag4",
                      "units_roll4_mean", "units_roll4_std", "week_of_year", "month", "quarter"]

FRIENDLY = {
    "units_roll4_mean": "Recent 4-week average demand",
    "units_lag1": "Last week's demand",
    "units_lag2": "Demand 2 weeks ago",
    "units_lag4": "Demand from the same week last month",
    "units_roll4_std": "Recent demand volatility",
    "week_of_year": "Time of year",
    "month": "Month",
    "quarter": "Quarter",
    "subcat_enc": "Product category",
    "region_enc": "Warehouse region",
}


def friendly(name):
    return FRIENDLY.get(name, name.replace("_", " ").title())


@st.cache_resource
def load_forecast_model():
    model = xgb.XGBRegressor()
    model.load_model(str(SRC / "xgb_forecast_v2.json"))
    with open(SRC / "label_encoders_v2.pkl", "rb") as f:
        encs = pickle.load(f)
    panel = pd.read_csv(SRC / "weekly_panel_v2.csv", parse_dates=["week"])
    # subcat_enc/region_enc are computed in-memory during training, not saved
    # to the panel CSV -- recompute here using the saved encoders.
    panel["subcat_enc"] = encs["subcat"].transform(panel["Sub-Category"])
    panel["region_enc"] = encs["region"].transform(panel["Region"])
    return model, encs, panel


def evaluate_decision(forecast, inventory_level, reorder_point):
    """
    Deterministic rule-based decision engine (NOT machine-learned).

    Base state comes from the inventory/reorder-point RATIO, using
    quartile thresholds derived from the real panel's actual distribution
    of that ratio (0.45 / 0.59 / 0.71) -- this guarantees the four states
    are genuinely represented across observations, rather than one state
    dominating almost every row (an earlier version of this formula
    compared forecast directly to reorder point, which rarely triggered
    since the forecast is, by construction, close to recent average demand).

    Forecast pressure (forecast exceeding the reorder point) then escalates
    severity by one tier on top of the structural base state -- this is
    what makes the What-If sliders visibly move the decision.
    """
    ratio = inventory_level / reorder_point if reorder_point > 0 else 999

    if ratio < 0.45:
        state, severity = "CRITICAL STOCK", "HIGH"
    elif ratio < 0.59:
        state, severity = "LOW STOCK", "MEDIUM"
    elif ratio < 0.71:
        state, severity = "WATCH", "MEDIUM"
    else:
        state, severity = "NORMAL", "LOW"

    escalated_by_forecast = False
    if forecast > reorder_point and severity != "HIGH":
        order = ["LOW", "MEDIUM", "HIGH"]
        severity = order[min(order.index(severity) + 1, 2)]
        escalated_by_forecast = True

    explanations = {
        "CRITICAL STOCK": "Inventory is well below the reorder point relative to typical demand for this SKU x Warehouse.",
        "LOW STOCK": "Inventory is below the reorder point.",
        "WATCH": "Inventory is approaching the reorder point.",
        "NORMAL": "Inventory is comfortably above the reorder point.",
    }
    explanation = explanations[state]
    if escalated_by_forecast:
        explanation += " Forecast demand also exceeds the reorder point, adding further pressure."

    return {"state": state, "severity": severity, "explanation": explanation}


def severity_badge(sev):
    color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[sev]
    return f":{color}[**{sev}**]"


def log_decision(row_dict):
    row_dict["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    row = pd.DataFrame([row_dict])
    if LOG_PATH.exists():
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_PATH, index=False)


def log_feedback(row_dict):
    row_dict["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    row = pd.DataFrame([row_dict])
    if FEEDBACK_PATH.exists():
        row.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(FEEDBACK_PATH, index=False)


def get_collection_stats():
    if not LOG_PATH.exists():
        return dict(total=0, ACCEPT=0, REJECT=0, OVERRIDE=0, unique_skus=0, unique_regions=0, evaluators=0)
    df = pd.read_csv(LOG_PATH)
    return dict(
        total=len(df),
        ACCEPT=(df["decision"] == "ACCEPT").sum(),
        REJECT=(df["decision"] == "REJECT").sum(),
        OVERRIDE=(df["decision"] == "OVERRIDE").sum(),
        unique_skus=df["sub_category"].nunique() if "sub_category" in df else 0,
        unique_regions=df["region"].nunique() if "region" in df else 0,
        evaluators=df["evaluator_id"].nunique() if "evaluator_id" in df else 0,
    )


# ============================================================ SIDEBAR
st.sidebar.title("📦 AdaptiveXSC")
st.sidebar.caption("Real XGBoost forecaster + real SHAP + rule-based decision engine.")
evaluator_id = st.sidebar.text_input("Evaluator ID", placeholder="e.g. EVAL-001")

stats = get_collection_stats()
st.sidebar.markdown("### Data Collection Monitor")
st.sidebar.write(f"Total decisions: **{stats['total']}**")
st.sidebar.write(f"Accept: {stats['ACCEPT']} · Reject: {stats['REJECT']} · Override: {stats['OVERRIDE']}")
st.sidebar.write(f"Unique SKUs seen: {stats['unique_skus']} · Regions: {stats['unique_regions']} · Evaluators: {stats['evaluators']}")

with st.sidebar.expander("ℹ️ What am I looking at?"):
    st.write(
        "**Forecast**: a real XGBoost model's prediction for this SKU x Warehouse x week.\n\n"
        "**SHAP**: which real model inputs pushed this forecast up or down.\n\n"
        "**What-If**: perturb the model's actual inputs (recent demand) and see the forecast + decision respond.\n\n"
        "**Inventory / Reorder Point**: NOT real inventory records (Superstore doesn't have any) — "
        "a transparent, deterministic base-stock formula applied to real demand statistics.\n\n"
        "**ACCEPT / REJECT / OVERRIDE**: your decision is logged with a timestamp, building a "
        "trust-calibration dataset."
    )

# ============================================================ MAIN
try:
    model, encs, panel = load_forecast_model()
except FileNotFoundError:
    st.error("Model files not found. In `src/`, run `python etl_demand_v2.py` then `python train_forecast_v2.py` first.")
    st.stop()

st.title("AdaptiveXSC — Decision Intelligence")

st.markdown("#### Observation Selection")
c1, c2, c3 = st.columns(3)
with c1:
    region = st.selectbox("Warehouse (Region)", sorted(panel["Region"].unique()))
with c2:
    subcats_in_region = sorted(panel[panel["Region"] == region]["Sub-Category"].unique())
    sub_category = st.selectbox("SKU (Product Sub-Category)", subcats_in_region)
with c3:
    dates_avail = sorted(panel[(panel["Region"] == region) & (panel["Sub-Category"] == sub_category)]["week"].dt.date.unique())
    week_date = st.selectbox("Week", dates_avail, index=len(dates_avail) - 1)

obs = panel[(panel["Region"] == region) & (panel["Sub-Category"] == sub_category) & (panel["week"].dt.date == week_date)]
if obs.empty:
    st.warning("No data for this combination.")
    st.stop()
obs = obs.iloc[0]

with st.expander("Selected Observation Summary", expanded=True):
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Actual units sold", f"{obs['units']:.0f}")
    o2.metric("Inventory level*", f"{obs['inventory_level']:.0f}")
    o3.metric("Reorder point*", f"{obs['reorder_point']:.0f}")
    o4.metric("Unit price", f"${obs['unit_price']:.2f}")
    o5.metric("Unit cost", f"${obs['unit_cost']:.2f}")
    st.caption("*Inventory level and reorder point are rule-based estimates from real demand statistics, not recorded inventory data — see sidebar note.")

X_base = obs[FORECAST_FEATURES].to_frame().T.astype(float)
base_pred = float(model.predict(X_base)[0])
explainer = shap.TreeExplainer(model)
base_sv = explainer(X_base)

st.markdown("---")
st.markdown("#### Baseline Forecast")
b1, b2, b3 = st.columns(3)
b1.metric("Forecast demand", f"{base_pred:.1f} units")
b2.metric("Actual demand", f"{obs['units']:.0f} units")
b3.metric("Forecast error", f"{abs(base_pred - obs['units']):.1f} units")

st.markdown("#### Forecast Drivers (real SHAP)")
contribs = sorted(zip(FORECAST_FEATURES, base_sv.values[0]), key=lambda x: abs(x[1]))[-5:]
fig = go.Figure(go.Bar(
    x=[v for _, v in contribs],
    y=[friendly(f) for f, _ in contribs],
    orientation="h",
    marker_color=["#d9685f" if v > 0 else "#4fb8a5" for _, v in contribs],
))
fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                   xaxis_title="Contribution to forecast (units)")
st.plotly_chart(fig, use_container_width=True)

base_decision = evaluate_decision(base_pred, obs["inventory_level"], obs["reorder_point"])
st.info(f"**Baseline decision:** {base_decision['state']}  {severity_badge(base_decision['severity'])}  \n{base_decision['explanation']}")

# ============================================================ WHAT-IF
st.markdown("---")
st.markdown("#### What-If Scenario Analysis")
st.caption("These sliders perturb the model's actual inputs (recent demand). This is a model-based counterfactual, not a causal prediction.")

w1, w2 = st.columns(2)
lag1_pct = w1.slider("Last week's demand — % change", -50, 100, 0, step=5)
roll_pct = w2.slider("4-week average demand — % change", -50, 100, 0, step=5)

X_scen = X_base.copy()
X_scen["units_lag1"] = X_scen["units_lag1"] * (1 + lag1_pct / 100.0)
X_scen["units_roll4_mean"] = X_scen["units_roll4_mean"] * (1 + roll_pct / 100.0)
is_scenario = (lag1_pct != 0) or (roll_pct != 0)

scen_pred = float(model.predict(X_scen)[0]) if is_scenario else base_pred
scen_sv = explainer(X_scen) if is_scenario else base_sv

s1, s2, s3 = st.columns(3)
s1.metric("Scenario forecast", f"{scen_pred:.1f} units")
diff = scen_pred - base_pred
diff_pct = (diff / base_pred * 100) if base_pred else 0
s2.metric("Forecast difference", f"{diff:+.1f} units", f"{diff_pct:+.1f}%")
scen_decision = evaluate_decision(scen_pred, obs["inventory_level"], obs["reorder_point"])
s3.metric("Scenario decision", scen_decision["state"])

if is_scenario:
    sev_order = ["LOW", "MEDIUM", "HIGH"]
    escalated = sev_order.index(scen_decision["severity"]) > sev_order.index(base_decision["severity"])
    if escalated:
        st.warning(f"⚠️ Scenario escalates severity: {base_decision['severity']} → {scen_decision['severity']}")
    else:
        st.success("No escalation relative to the baseline decision.")
else:
    st.caption("No scenario applied — showing baseline.")

# ============================================================ HUMAN DECISION
st.markdown("---")
st.markdown("#### Human Decision")
st.write("You are reviewing the model's recommendation for this observation.")

d1, d2 = st.columns([2, 3])
with d1:
    action = st.radio("Decision action", ["ACCEPT", "REJECT", "OVERRIDE"], horizontal=False)
reason, note = None, None
if action == "OVERRIDE":
    reason = st.selectbox("Reason", ["Supplier Delay", "Upcoming Promotion", "Local Demand Knowledge",
                                      "Inventory Constraint", "Data Quality Concern",
                                      "Forecast Appears Too High", "Forecast Appears Too Low", "Other"])
    note = st.text_area("Additional note")

if st.button("SUBMIT DECISION", type="primary"):
    if not evaluator_id or not evaluator_id.strip():
        st.error("Please enter an Evaluator ID in the sidebar before submitting.")
    elif action == "OVERRIDE" and not reason:
        st.error("Please select an override reason.")
    else:
        log_decision({
            "evaluator_id": evaluator_id.strip(),
            "sub_category": sub_category, "region": region, "week": str(week_date),
            "baseline_forecast": base_pred, "actual_units": float(obs["units"]),
            "baseline_state": base_decision["state"], "baseline_severity": base_decision["severity"],
            "scenario_applied": is_scenario, "scenario_forecast": scen_pred,
            "scenario_state": scen_decision["state"], "scenario_severity": scen_decision["severity"],
            "decision": action, "override_reason": reason or "", "override_note": note or "",
        })
        st.success("Decision recorded.")

# ============================================================ FEEDBACK
with st.expander("💬 Evaluator Feedback"):
    f_type = st.selectbox("Feedback type", ["Bug", "Confusing Explanation", "UI Problem", "Other"])
    f_text = st.text_area("Description", key="feedback_text")
    if st.button("Submit Feedback"):
        if not evaluator_id or not evaluator_id.strip():
            st.error("Please enter your Evaluator ID in the sidebar first.")
        elif not f_text.strip():
            st.error("Please enter a description.")
        else:
            log_feedback({
                "evaluator_id": evaluator_id.strip(), "sub_category": sub_category,
                "region": region, "week": str(week_date),
                "feedback_type": f_type, "feedback_text": f_text.strip(),
            })
            st.success("Feedback submitted. Thank you!")

# ============================================================ LOG VIEW
with st.expander("📋 View Decision Log"):
    if LOG_PATH.exists():
        st.dataframe(pd.read_csv(LOG_PATH).sort_values("timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No decisions logged yet.")
