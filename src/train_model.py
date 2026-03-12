"""
FinShield — Financial Fraud Detection System
ML Pipeline: Feature Engineering + Model Training + Evaluation
Models: Random Forest, XGBoost, Logistic Regression (ensemble)
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import pickle

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("FinShield — ML Fraud Detection Pipeline")
print("=" * 60)

df = pd.read_csv("data/transactions.csv")
print(f"Loaded {len(df):,} transactions | Fraud rate: {df['is_fraud'].mean()*100:.1f}%")

# ── Feature engineering ───────────────────────────────────────────────────────
print("\n[1/5] Feature Engineering...")

le_cat = LabelEncoder()
le_dow = LabelEncoder()
df["category_enc"]  = le_cat.fit_transform(df["merchant_category"])
df["dow_enc"]       = le_dow.fit_transform(df["day_of_week"])
df["is_weekend"]    = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
df["is_peak_hour"]  = df["transaction_hour"].between(10, 20).astype(int)
df["high_velocity"] = (df["velocity_24h"] > 10).astype(int)
df["far_from_home"] = (df["distance_from_home_km"] > 1000).astype(int)
df["high_amount"]   = (df["amount"] > df["amount"].quantile(0.90)).astype(int)
df["amount_log"]    = np.log1p(df["amount"])
df["velocity_ratio"]= df["velocity_1h"] / (df["velocity_24h"] + 1)
df["risk_x_foreign"]= df["risk_score"] * df["is_foreign"]

FEATURES = [
    "amount", "amount_log", "transaction_hour", "is_night_txn",
    "is_weekend", "is_peak_hour", "is_foreign", "velocity_1h",
    "velocity_24h", "high_velocity", "velocity_ratio",
    "distance_from_home_km", "far_from_home", "is_new_merchant",
    "declined_last_24h", "credit_utilization_pct", "account_age_days",
    "customer_age", "credit_limit", "risk_score", "risk_x_foreign",
    "high_amount", "category_enc", "dow_enc",
]

X = df[FEATURES]
y = df["is_fraud"]

# ── Train/test split ──────────────────────────────────────────────────────────
print("[2/5] Train/Test Split (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Train fraud: {y_train.sum():,} | Test fraud: {y_test.sum():,}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Class weights for imbalanced data
cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}

# ── Train models ──────────────────────────────────────────────────────────────
print("[3/5] Training Models...")

models = {
    "Logistic Regression": LogisticRegression(
        class_weight=class_weight, max_iter=1000, random_state=42, C=0.1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight=class_weight,
        random_state=42, n_jobs=-1, min_samples_leaf=5
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...")
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

    threshold = 0.35  # Lower threshold = catch more fraud
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    f1       = f1_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)

    results[name] = {
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_prec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "model": model,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
    print(f"    ROC-AUC: {roc_auc:.4f} | Avg Precision: {avg_prec:.4f} | F1: {f1:.4f}")

# ── Best model ────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best = results[best_name]
print(f"\n[4/5] Best Model: {best_name} (ROC-AUC: {best['roc_auc']})")

print("\nClassification Report:")
print(classification_report(y_test, best["y_pred"], target_names=["Legit", "Fraud"]))

# Feature importance (Random Forest)
rf = results["Random Forest"]["model"]
feat_imp = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 Features:")
print(feat_imp.head(10).to_string(index=False))

# ── Score full dataset + export ───────────────────────────────────────────────
print("\n[5/5] Scoring full dataset for PowerBI export...")

df["category_enc"] = le_cat.transform(df["merchant_category"])
df["dow_enc"]      = le_dow.transform(df["day_of_week"])

X_full = df[FEATURES]
rf_model = results["Random Forest"]["model"]
df["ml_fraud_probability"] = rf_model.predict_proba(X_full)[:, 1].round(4)
df["ml_fraud_flag"]        = (df["ml_fraud_probability"] >= 0.35).astype(int)
df["ml_risk_tier"]         = pd.cut(df["ml_fraud_probability"],
                                     bins=[-0.001, 0.2, 0.4, 0.65, 1.0],
                                     labels=["Low", "Medium", "High", "Critical"])

os.makedirs("data", exist_ok=True)
export_cols = [
    "transaction_id", "customer_id", "merchant_id", "merchant_category",
    "transaction_date", "transaction_hour", "day_of_week", "month",
    "amount", "country", "is_foreign", "is_night_txn",
    "velocity_1h", "velocity_24h", "distance_from_home_km",
    "is_new_merchant", "declined_last_24h", "credit_utilization_pct",
    "account_age_days", "customer_age", "credit_limit",
    "risk_score", "risk_tier", "is_fraud",
    "ml_fraud_probability", "ml_fraud_flag", "ml_risk_tier",
]
df[export_cols].to_csv("data/transactions_scored.csv", index=False)
feat_imp.to_csv("data/feature_importance.csv", index=False)

# Summary stats for dashboard
summary = {
    "total_transactions": len(df),
    "total_fraud":        int(df["is_fraud"].sum()),
    "fraud_rate_pct":     round(df["is_fraud"].mean() * 100, 2),
    "total_fraud_amount": round(df[df["is_fraud"]==1]["amount"].sum(), 2),
    "ml_flagged":         int(df["ml_fraud_flag"].sum()),
    "best_model":         best_name,
    "best_roc_auc":       best["roc_auc"],
    "best_f1":            best["f1_score"],
}
with open("data/model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save model
os.makedirs("src/models", exist_ok=True)
with open("src/models/fraud_rf_model.pkl", "wb") as f:
    pickle.dump({"model": rf_model, "scaler": scaler,
                 "features": FEATURES, "le_cat": le_cat, "le_dow": le_dow}, f)

print("\n✅ Pipeline complete!")
print(f"  Exported: data/transactions_scored.csv ({len(df):,} rows)")
print(f"  Exported: data/feature_importance.csv")
print(f"  Saved:    src/models/fraud_rf_model.pkl")
print(f"\nModel Summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")
