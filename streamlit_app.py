"""
FinShield — Financial Fraud Detection System
Streamlit Demo App
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import time
from datetime import datetime

st.set_page_config(
    page_title="FinShield — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; }
.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.critical { color: #ff4444; font-weight: 700; }
.high     { color: #ff8800; font-weight: 700; }
.medium   { color: #ffcc00; font-weight: 700; }
.low      { color: #00cc66; font-weight: 700; }
.alert-box {
    background: #1a1d27;
    border-left: 4px solid #ff4444;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "retail", "electronics",
    "travel", "online_shopping", "atm_withdrawal", "luxury_goods",
    "crypto_exchange", "wire_transfer", "gaming",
]
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
COUNTRIES = ["IN","US","GB","DE","SG","AE","NG","RO","CN","BR"]

# ── Feature computation ────────────────────────────────────────────────────────
def compute_risk_score(is_foreign, hour, vel_1h, dist, is_new, declined):
    is_night = 1 if (hour >= 23 or hour <= 5) else 0
    return min(100, (
        is_foreign * 25 +
        is_night * 15 +
        (1 if vel_1h > 3 else 0) * 20 +
        (1 if dist > 1000 else 0) * 20 +
        is_new * 10 +
        declined * 5
    ))

def get_alert_reasons(amount, is_foreign, hour, vel_1h, dist, declined, prob):
    reasons = []
    if is_foreign and amount > 500:
        reasons.append(f"Foreign high-value transaction of ₹{amount:,.0f}")
    if vel_1h > 4:
        reasons.append(f"High velocity: {vel_1h} transactions in last 1 hour")
    if dist > 1000:
        reasons.append(f"Transaction {dist:.0f} km from home location")
    if declined >= 3:
        reasons.append(f"{declined} declined transactions in last 24 hours")
    if hour >= 23 or hour <= 5:
        reasons.append(f"Suspicious hour: {hour}:00 (late night activity)")
    if prob >= 0.65:
        reasons.append(f"ML model high confidence: {prob*100:.1f}% fraud probability")
    return reasons or ["Flagged by ML ensemble model"]

def score_transaction_rules(amount, merchant_category, hour, is_foreign,
                             vel_1h, vel_24h, dist, is_new, declined,
                             credit_util, account_age, cust_age, credit_limit):
    """Rule-based + simple scoring when model not available"""
    BASE_SCORES = {
        "wire_transfer": 0.55, "crypto_exchange": 0.50, "luxury_goods": 0.45,
        "atm_withdrawal": 0.40, "online_shopping": 0.38, "electronics": 0.35,
        "travel": 0.28, "gaming": 0.18, "retail": 0.10, "gas_station": 0.08,
        "restaurant": 0.07, "grocery": 0.05,
    }
    base = BASE_SCORES.get(merchant_category, 0.2)
    risk = compute_risk_score(is_foreign, hour, vel_1h, dist, is_new, declined)

    prob = base
    if is_foreign: prob += 0.15
    if hour >= 23 or hour <= 5: prob += 0.10
    if vel_1h > 4: prob += 0.12
    if dist > 1000: prob += 0.12
    if declined >= 3: prob += 0.10
    if amount > 5000: prob += 0.08
    if credit_util > 80: prob += 0.05
    if account_age < 90: prob += 0.06
    return min(0.99, round(prob, 4)), risk

# ── Load model if available ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "src/models/fraud_rf_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    if os.path.exists("data/transactions_scored.csv"):
        return pd.read_csv("data/transactions_scored.csv")
    return None

bundle = load_model()
df     = load_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ FinShield")
st.markdown("##### Financial Fraud Detection System · ML-Powered · Real-Time Risk Scoring")
st.divider()

# ── Sidebar nav ───────────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🔍 Score a Transaction",
    "📁 Dataset Explorer",
    "🤖 Model Performance",
])
st.sidebar.divider()
st.sidebar.caption("FinShield v1.0 · Random Forest + Gradient Boosting")
st.sidebar.caption("github.com/tusharg007/finshield")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    if df is not None:
        fraud_df  = df[df["is_fraud"] == 1]
        total     = len(df)
        n_fraud   = int(df["is_fraud"].sum())
        rate      = round(df["is_fraud"].mean() * 100, 2)
        exposure  = round(fraud_df["amount"].sum(), 0)
        avg_fraud = round(fraud_df["amount"].mean(), 0)

        st.markdown("### 📈 Executive Summary")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Fraud Cases",        f"{n_fraud:,}")
        c3.metric("Fraud Rate",         f"{rate}%")
        c4.metric("Total Exposure",     f"₹{exposure/1e5:.1f}L")
        c5.metric("Avg Fraud Amount",   f"₹{avg_fraud:,}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Fraud by Merchant Category")
            cat_stats = df.groupby("merchant_category").agg(
                fraud_count=("is_fraud","sum"),
                fraud_rate=("is_fraud","mean"),
                total=("is_fraud","count")
            ).reset_index().sort_values("fraud_rate", ascending=False)
            cat_stats["fraud_rate_pct"] = (cat_stats["fraud_rate"]*100).round(2)
            st.bar_chart(cat_stats.set_index("merchant_category")["fraud_rate_pct"])

        with col2:
            st.markdown("#### Fraud by Hour of Day")
            hour_stats = df.groupby("transaction_hour")["is_fraud"].mean().reset_index()
            hour_stats["fraud_rate_pct"] = (hour_stats["is_fraud"]*100).round(2)
            st.line_chart(hour_stats.set_index("transaction_hour")["fraud_rate_pct"])

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Risk Tier Distribution")
            tier_counts = df["ml_risk_tier"].value_counts()
            st.bar_chart(tier_counts)

        with col4:
            st.markdown("#### Fraud: Foreign vs Domestic")
            geo = df.groupby("is_foreign").agg(
                total=("is_fraud","count"),
                fraud=("is_fraud","sum")
            ).reset_index()
            geo["label"] = geo["is_foreign"].map({0:"Domestic", 1:"Foreign"})
            geo["fraud_rate_pct"] = (geo["fraud"]/geo["total"]*100).round(2)
            st.bar_chart(geo.set_index("label")["fraud_rate_pct"])

        st.divider()
        st.markdown("#### 🔴 Top 10 Highest-Risk Transactions")
        top_risk = df.nlargest(10, "ml_fraud_probability")[
            ["transaction_id","amount","merchant_category","country",
             "ml_fraud_probability","ml_risk_tier","is_fraud"]
        ].rename(columns={"ml_fraud_probability":"fraud_prob","ml_risk_tier":"risk_tier"})
        st.dataframe(top_risk, use_container_width=True)

    else:
        st.warning("Dataset not found. Run `python src/generate_data.py` and `python src/train_model.py` first.")
        st.info("In the meantime, use the **Score a Transaction** page to test the rule-based engine.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SCORE A TRANSACTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Score a Transaction":
    st.markdown("### 🔍 Real-Time Transaction Fraud Scorer")
    st.caption("Fill in transaction details below and click **Score Transaction**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Transaction Details**")
        txn_id   = st.text_input("Transaction ID", value=f"TXN{np.random.randint(1000,9999)}")
        amount   = st.number_input("Amount (₹)", min_value=1.0, max_value=500000.0, value=4500.0, step=100.0)
        category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES, index=10)
        country  = st.selectbox("Country", COUNTRIES)
        is_foreign = 1 if country != "IN" else 0

    with col2:
        st.markdown("**Temporal & Behavioral**")
        hour     = st.slider("Transaction Hour", 0, 23, 2)
        dow      = st.selectbox("Day of Week", DAYS, index=5)
        vel_1h   = st.slider("Transactions in Last 1 Hour", 0, 20, 6)
        vel_24h  = st.slider("Transactions in Last 24 Hours", 0, 50, 14)
        declined = st.slider("Declines in Last 24 Hours", 0, 10, 3)

    with col3:
        st.markdown("**Customer Profile**")
        dist        = st.number_input("Distance from Home (km)", 0.0, 20000.0, 8000.0, 100.0)
        is_new_merch= st.selectbox("New Merchant?", [("Yes",1),("No",0)], format_func=lambda x: x[0])[1]
        credit_util = st.slider("Credit Utilization (%)", 0, 100, 87)
        account_age = st.number_input("Account Age (days)", 1, 3650, 45)
        cust_age    = st.number_input("Customer Age", 18, 80, 28)
        credit_limit= st.number_input("Credit Limit (₹)", 5000, 1000000, 50000, 5000)

    st.divider()

    # Quick presets
    st.markdown("**Quick Presets:**")
    p1, p2, p3, p4 = st.columns(4)
    preset = None
    if p1.button("🔴 High Risk Wire Transfer"): preset = "high"
    if p2.button("🟠 Suspicious Foreign Txn"): preset = "medium"
    if p3.button("🟢 Normal Grocery Purchase"): preset = "low"
    if p4.button("🟡 Borderline Online Shop"): preset = "border"

    if preset == "high":
        st.info("Preset loaded: High-risk wire transfer at 2AM from overseas")
        amount=45000; category="wire_transfer"; country="NG"; is_foreign=1
        hour=2; vel_1h=7; vel_24h=18; declined=4; dist=12000; is_new_merch=1; credit_util=92
    elif preset == "medium":
        amount=8500; category="luxury_goods"; country="AE"; is_foreign=1
        hour=14; vel_1h=2; vel_24h=8; declined=1; dist=5000; is_new_merch=1; credit_util=65
    elif preset == "low":
        amount=350; category="grocery"; country="IN"; is_foreign=0
        hour=11; vel_1h=1; vel_24h=3; declined=0; dist=5; is_new_merch=0; credit_util=15
    elif preset == "border":
        amount=3200; category="online_shopping"; country="IN"; is_foreign=0
        hour=22; vel_1h=3; vel_24h=9; declined=2; dist=300; is_new_merch=1; credit_util=55

    if st.button("⚡ SCORE TRANSACTION", type="primary", use_container_width=True):
        with st.spinner("Scoring..."):
            time.sleep(0.4)
            prob, risk_score = score_transaction_rules(
                amount, category, hour, is_foreign,
                vel_1h, vel_24h, dist, is_new_merch,
                declined, credit_util, account_age, cust_age, credit_limit
            )

            # If model available, use it
            if bundle:
                try:
                    le_cat = bundle["le_cat"]; le_dow = bundle["le_dow"]
                    FEATURES = bundle["features"]; model = bundle["model"]
                    is_night = 1 if (hour >= 23 or hour <= 5) else 0
                    is_weekend = 1 if dow in ["Saturday","Sunday"] else 0
                    try: cat_enc = int(le_cat.transform([category])[0])
                    except: cat_enc = 0
                    try: dow_enc = int(le_dow.transform([dow])[0])
                    except: dow_enc = 0
                    row = {
                        "amount": amount, "amount_log": np.log1p(amount),
                        "transaction_hour": hour, "is_night_txn": is_night,
                        "is_weekend": is_weekend, "is_peak_hour": 1 if 10<=hour<=20 else 0,
                        "is_foreign": is_foreign, "velocity_1h": vel_1h,
                        "velocity_24h": vel_24h, "high_velocity": 1 if vel_24h>10 else 0,
                        "velocity_ratio": vel_1h/(vel_24h+1),
                        "distance_from_home_km": dist, "far_from_home": 1 if dist>1000 else 0,
                        "is_new_merchant": is_new_merch, "declined_last_24h": declined,
                        "credit_utilization_pct": credit_util, "account_age_days": account_age,
                        "customer_age": cust_age, "credit_limit": credit_limit,
                        "risk_score": risk_score, "risk_x_foreign": risk_score*is_foreign,
                        "high_amount": 1 if amount>5000 else 0,
                        "category_enc": cat_enc, "dow_enc": dow_enc,
                    }
                    X = pd.DataFrame([row])[FEATURES]
                    prob = round(float(model.predict_proba(X)[0][1]), 4)
                except Exception as e:
                    pass  # fall back to rule-based

            tier = ("Critical" if prob>=0.65 else "High" if prob>=0.40
                    else "Medium" if prob>=0.20 else "Low")
            tier_color = {"Critical":"critical","High":"high","Medium":"medium","Low":"low"}[tier]
            reasons = get_alert_reasons(amount, is_foreign, hour, vel_1h, dist, declined, prob)

        st.success("✅ Scoring complete")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fraud Probability", f"{prob*100:.1f}%")
        r2.metric("Risk Tier", tier)
        r3.metric("Rule Risk Score", f"{risk_score}/100")
        r4.metric("Fraud Flag", "🚨 YES" if prob>=0.35 else "✅ NO")

        st.markdown(f"#### Decision: <span class='{tier_color}'>{tier.upper()} RISK</span>",
                    unsafe_allow_html=True)

        st.markdown("**Alert Reasons:**")
        for r in reasons:
            color = "#ff4444" if tier=="Critical" else "#ff8800" if tier=="High" else "#ffcc00"
            st.markdown(f'<div class="alert-box" style="border-left-color:{color}">⚠️ {r}</div>',
                        unsafe_allow_html=True)

        st.markdown("**Probability Gauge:**")
        st.progress(min(int(prob*100), 100))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Dataset Explorer":
    st.markdown("### 📁 Transaction Dataset Explorer")

    if df is not None:
        st.markdown(f"**{len(df):,} transactions** · {df['is_fraud'].sum():,} fraud · {df['is_fraud'].mean()*100:.1f}% fraud rate")
        st.divider()

        col1, col2, col3 = st.columns(3)
        filt_tier    = col1.multiselect("Risk Tier", ["Low","Medium","High","Critical"],
                                         default=["High","Critical"])
        filt_cat     = col2.multiselect("Category", MERCHANT_CATEGORIES)
        filt_foreign = col3.selectbox("Transaction Type", ["All","Foreign","Domestic"])

        filtered = df.copy()
        if filt_tier:
            filtered = filtered[filtered["ml_risk_tier"].isin(filt_tier)]
        if filt_cat:
            filtered = filtered[filtered["merchant_category"].isin(filt_cat)]
        if filt_foreign == "Foreign":
            filtered = filtered[filtered["is_foreign"] == 1]
        elif filt_foreign == "Domestic":
            filtered = filtered[filtered["is_foreign"] == 0]

        st.markdown(f"Showing **{len(filtered):,}** transactions")
        cols_show = ["transaction_id","transaction_date","amount","merchant_category",
                     "country","is_fraud","ml_fraud_probability","ml_risk_tier","risk_score"]
        st.dataframe(filtered[cols_show].head(500), use_container_width=True)

        csv = filtered[cols_show].to_csv(index=False)
        st.download_button("⬇️ Download Filtered CSV", csv,
                           "finshield_filtered.csv", "text/csv")
    else:
        st.warning("Dataset not found. Run the ML pipeline first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("### 🤖 ML Model Performance")

    if os.path.exists("data/model_summary.json"):
        with open("data/model_summary.json") as f:
            summary = json.load(f)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Best Model",  summary["best_model"])
        c2.metric("ROC-AUC",     summary["best_roc_auc"])
        c3.metric("F1 Score",    summary["best_f1"])
        c4.metric("ML Flagged",  f"{summary['ml_flagged']:,}")

    if os.path.exists("data/feature_importance.csv"):
        st.divider()
        fi = pd.read_csv("data/feature_importance.csv").head(15)
        st.markdown("#### Top 15 Features by Importance")
        st.bar_chart(fi.set_index("feature")["importance"])
        st.dataframe(fi, use_container_width=True)

    st.divider()
    st.markdown("#### Models Trained")
    st.markdown("""
    | Model | Approach | Handles Imbalance |
    |-------|----------|-------------------|
    | Logistic Regression | Linear baseline | class_weight=balanced |
    | Random Forest | Bagging ensemble, 200 trees | class_weight=balanced |
    | Gradient Boosting | Sequential boosting, 200 trees | subsample=0.8 |

    **Threshold:** 0.35 (tuned for high recall — catch more fraud at cost of some precision)

    **Features engineered:** velocity_ratio, amount_log, risk_x_foreign, far_from_home, high_velocity, is_weekend, is_peak_hour, high_amount
    """)
    if df is not None:
        st.divider()
        st.markdown("#### Fraud Probability Distribution")
        st.bar_chart(pd.cut(df["ml_fraud_probability"], bins=20).value_counts().sort_index())
