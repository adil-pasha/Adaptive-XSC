"""
Phase II - Models: Stockout/Delay Risk (real dataset)
DataCo Smart Supply Chain for Big Data Analysis (Constante, Silva & Pereira, 2019).
180,519 real orders. Target: Late_delivery_risk.

Leakage avoidance (critical for research validity):
- 'Days for shipping (real)' and 'Delivery Status' are only known AFTER
  delivery -- they are direct proxies for the label itself and must be
  excluded, or the model would be predicting the past, not forecasting risk.
- PII (names, emails, passwords, street addresses) dropped.
- High-cardinality free-text/ID columns dropped to avoid memorization.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
import json

df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")

y = df["Late_delivery_risk"]

leakage_cols = ["Days for shipping (real)", "Delivery Status", "Late_delivery_risk"]
pii_cols = ["Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
            "Customer Street", "Customer City", "Customer State", "Customer Zipcode",
            "Customer Id", "Order Customer Id"]
id_or_highcard = ["Order Id", "Order Item Id", "Order Item Cardprod Id", "Product Card Id",
                   "Product Category Id", "Product Description", "Product Image",
                   "Product Name", "Order City", "Order State", "Order Zipcode",
                   "Order Status", "order date (DateOrders)", "shipping date (DateOrders)",
                   "Latitude", "Longitude"]

drop_cols = leakage_cols + pii_cols + id_or_highcard
X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])

cat_cols = [c for c in X_raw.columns if pd.api.types.is_string_dtype(X_raw[c]) or X_raw[c].dtype == object]
num_cols = [c for c in X_raw.columns if c not in cat_cols]

print("Categorical features:", cat_cols)
print("Numeric features:", num_cols)

X = X_raw.copy()
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    encoders[c] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_s, y_train)
pred_lr = logreg.predict(X_test_s)
proba_lr = logreg.predict_proba(X_test_s)[:, 1]

xgb_clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_clf.fit(X_train, y_train)
pred_xgb = xgb_clf.predict(X_test)
proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

def report(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"{name:20s} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} auc={auc:.3f}")
    print(f"  confusion matrix [[TN,FP],[FN,TP]]: {cm}")
    return dict(model=name, accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc, confusion_matrix=cm)

print(f"\nn={len(df)}, train={len(X_train)}, test={len(X_test)}")
print("\n--- Late delivery / stockout risk classification (held-out test set) ---")
results = [
    report("Logistic Regression", y_test.values, pred_lr, proba_lr),
    report("XGBoost Classifier", y_test.values, pred_xgb, proba_xgb),
]

with open("risk_results.json", "w") as f:
    json.dump(results, f, indent=2)

import pickle
with open("risk_models.pkl", "wb") as f:
    pickle.dump({"logreg": logreg, "xgb": xgb_clf, "scaler": scaler, "encoders": encoders,
                 "num_cols": num_cols, "cat_cols": cat_cols, "features": list(X.columns)}, f)
X_test.assign(y_true=y_test.values, pred_lr=pred_lr, proba_lr=proba_lr,
              pred_xgb=pred_xgb, proba_xgb=proba_xgb).to_csv("risk_test_set.csv", index=False)

print("\nSaved risk_results.json, risk_models.pkl, risk_test_set.csv")
