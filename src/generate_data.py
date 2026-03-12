"""
FinShield — Financial Fraud Detection System
Data Generator: Creates realistic synthetic transaction dataset
Generates ~50,000 transactions with labeled fraud patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N_TRANSACTIONS = 50000
FRAUD_RATE = 0.034  # 3.4% fraud rate (realistic industry figure)
N_CUSTOMERS = 5000
N_MERCHANTS = 800

# ── Merchant categories ────────────────────────────────────────────────────────
MERCHANT_CATEGORIES = {
    "grocery":        (35, 0.008),   # (avg_amount, fraud_weight)
    "gas_station":    (55, 0.012),
    "restaurant":     (45, 0.010),
    "retail":         (120, 0.020),
    "electronics":    (450, 0.080),
    "travel":         (800, 0.060),
    "online_shopping":(180, 0.090),
    "atm_withdrawal": (200, 0.100),
    "luxury_goods":   (1200, 0.120),
    "crypto_exchange":(600, 0.150),
    "wire_transfer":  (2500, 0.180),
    "gaming":         (80, 0.040),
}

COUNTRIES = ["IN", "US", "GB", "DE", "SG", "AE", "NG", "RO", "CN", "BR"]
HOME_COUNTRY = "IN"

def generate_customers():
    customers = []
    for i in range(N_CUSTOMERS):
        customers.append({
            "customer_id": f"CUST{i+1:05d}",
            "age": np.random.randint(22, 72),
            "credit_limit": np.random.choice([25000, 50000, 100000, 200000, 500000],
                                              p=[0.35, 0.30, 0.20, 0.10, 0.05]),
            "account_age_days": np.random.randint(30, 3650),
            "home_country": HOME_COUNTRY,
        })
    return pd.DataFrame(customers)

def generate_transactions(customers_df):
    records = []
    start_date = datetime(2023, 1, 1)
    cat_list = list(MERCHANT_CATEGORIES.keys())
    fraud_weights = [MERCHANT_CATEGORIES[c][1] for c in cat_list]
    total_fw = sum(fraud_weights)
    fraud_cat_probs = [w / total_fw for w in fraud_weights]

    for i in range(N_TRANSACTIONS):
        cust = customers_df.sample(1).iloc[0]
        txn_date = start_date + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )
        hour = txn_date.hour
        is_night = 1 if (hour >= 23 or hour <= 5) else 0

        # Determine if fraud
        is_fraud = np.random.random() < FRAUD_RATE

        if is_fraud:
            category = np.random.choice(cat_list, p=fraud_cat_probs)
            avg_amt, _ = MERCHANT_CATEGORIES[category]
            amount = round(avg_amt * np.random.uniform(1.5, 8.0), 2)
            country = np.random.choice([c for c in COUNTRIES if c != HOME_COUNTRY])
            is_night = np.random.choice([0, 1], p=[0.3, 0.7])
            velocity_1h = np.random.randint(3, 15)
            velocity_24h = np.random.randint(8, 30)
            distance_from_home = np.random.uniform(500, 15000)
            is_new_merchant = np.random.choice([0, 1], p=[0.2, 0.8])
            declined_last_24h = np.random.randint(1, 6)
        else:
            category = np.random.choice(cat_list,
                                         p=[1/len(cat_list)]*len(cat_list))
            avg_amt, _ = MERCHANT_CATEGORIES[category]
            amount = round(avg_amt * np.random.uniform(0.3, 2.5), 2)
            country = HOME_COUNTRY if np.random.random() < 0.82 else np.random.choice([c for c in COUNTRIES if c != HOME_COUNTRY])
            velocity_1h = np.random.randint(0, 4)
            velocity_24h = np.random.randint(1, 12)
            distance_from_home = np.random.uniform(0, 800)
            is_new_merchant = np.random.choice([0, 1], p=[0.7, 0.3])
            declined_last_24h = np.random.randint(0, 2)

        merchant_id = f"MERCH{np.random.randint(1, N_MERCHANTS+1):04d}"
        utilization = round(min(amount / cust["credit_limit"] * 100, 100), 2)

        records.append({
            "transaction_id":      f"TXN{i+1:07d}",
            "customer_id":         cust["customer_id"],
            "merchant_id":         merchant_id,
            "merchant_category":   category,
            "transaction_date":    txn_date.strftime("%Y-%m-%d"),
            "transaction_time":    txn_date.strftime("%H:%M:%S"),
            "transaction_hour":    hour,
            "day_of_week":         txn_date.strftime("%A"),
            "month":               txn_date.month,
            "amount":              amount,
            "country":             country,
            "is_foreign":          1 if country != HOME_COUNTRY else 0,
            "is_night_txn":        is_night,
            "velocity_1h":         velocity_1h,
            "velocity_24h":        velocity_24h,
            "distance_from_home_km": round(distance_from_home, 2),
            "is_new_merchant":     is_new_merchant,
            "declined_last_24h":   declined_last_24h,
            "credit_utilization_pct": utilization,
            "account_age_days":    cust["account_age_days"],
            "customer_age":        cust["age"],
            "credit_limit":        cust["credit_limit"],
            "is_fraud":            int(is_fraud),
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    print("Generating customers...")
    customers = generate_customers()

    print("Generating transactions...")
    df = generate_transactions(customers)

    # Add risk score column (rule-based pre-score for dashboard)
    df["risk_score"] = (
        df["is_foreign"] * 25 +
        df["is_night_txn"] * 15 +
        (df["velocity_1h"] > 3).astype(int) * 20 +
        (df["distance_from_home_km"] > 1000).astype(int) * 20 +
        df["is_new_merchant"] * 10 +
        df["declined_last_24h"] * 5
    ).clip(0, 100)

    df["risk_tier"] = pd.cut(df["risk_score"],
                              bins=[-1, 20, 45, 70, 100],
                              labels=["Low", "Medium", "High", "Critical"])

    out = "data/transactions.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nDataset saved: {out}")
    print(f"Total transactions : {len(df):,}")
    print(f"Fraud transactions : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"Fraud amount       : ₹{df[df['is_fraud']==1]['amount'].sum():,.0f}")
    print(f"Date range         : {df['transaction_date'].min()} → {df['transaction_date'].max()}")
    print("\nFraud by category:")
    print(df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False).round(3))
