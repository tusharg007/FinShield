"""
FinShield — Financial Fraud Detection System
REST API: Real-time fraud scoring endpoint
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = FastAPI(
    title="FinShield Fraud Detection API",
    description="Real-time financial transaction fraud scoring",
    version="1.0.0",
)

# Load model
try:
    with open("src/models/fraud_rf_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    MODEL   = bundle["model"]
    SCALER  = bundle["scaler"]
    FEATURES= bundle["features"]
    LE_CAT  = bundle["le_cat"]
    LE_DOW  = bundle["le_dow"]
    print("Model loaded successfully")
except FileNotFoundError:
    MODEL = None
    print("WARNING: Model not found. Run train_model.py first.")


class TransactionRequest(BaseModel):
    transaction_id:         str
    amount:                 float = Field(..., gt=0)
    merchant_category:      str
    transaction_hour:       int   = Field(..., ge=0, le=23)
    day_of_week:            str
    is_foreign:             int   = Field(..., ge=0, le=1)
    velocity_1h:            int   = Field(..., ge=0)
    velocity_24h:           int   = Field(..., ge=0)
    distance_from_home_km:  float = Field(..., ge=0)
    is_new_merchant:        int   = Field(..., ge=0, le=1)
    declined_last_24h:      int   = Field(..., ge=0)
    credit_utilization_pct: float = Field(..., ge=0, le=100)
    account_age_days:       int   = Field(..., gt=0)
    customer_age:           int   = Field(..., ge=18)
    credit_limit:           float = Field(..., gt=0)
    risk_score:             Optional[int] = 0


class FraudResponse(BaseModel):
    transaction_id:     str
    fraud_probability:  float
    fraud_flag:         bool
    risk_tier:          str
    alert_reasons:      list
    scored_at:          str


def compute_features(req: TransactionRequest) -> pd.DataFrame:
    is_night   = 1 if (req.transaction_hour >= 23 or req.transaction_hour <= 5) else 0
    is_weekend = 1 if req.day_of_week in ["Saturday", "Sunday"] else 0
    is_peak    = 1 if 10 <= req.transaction_hour <= 20 else 0

    try:
        cat_enc = int(LE_CAT.transform([req.merchant_category])[0])
    except ValueError:
        cat_enc = 0
    try:
        dow_enc = int(LE_DOW.transform([req.day_of_week])[0])
    except ValueError:
        dow_enc = 0

    risk_score = req.risk_score or (
        req.is_foreign * 25 +
        is_night * 15 +
        (1 if req.velocity_1h > 3 else 0) * 20 +
        (1 if req.distance_from_home_km > 1000 else 0) * 20 +
        req.is_new_merchant * 10 +
        req.declined_last_24h * 5
    )

    row = {
        "amount":                 req.amount,
        "amount_log":             np.log1p(req.amount),
        "transaction_hour":       req.transaction_hour,
        "is_night_txn":           is_night,
        "is_weekend":             is_weekend,
        "is_peak_hour":           is_peak,
        "is_foreign":             req.is_foreign,
        "velocity_1h":            req.velocity_1h,
        "velocity_24h":           req.velocity_24h,
        "high_velocity":          1 if req.velocity_24h > 10 else 0,
        "velocity_ratio":         req.velocity_1h / (req.velocity_24h + 1),
        "distance_from_home_km":  req.distance_from_home_km,
        "far_from_home":          1 if req.distance_from_home_km > 1000 else 0,
        "is_new_merchant":        req.is_new_merchant,
        "declined_last_24h":      req.declined_last_24h,
        "credit_utilization_pct": req.credit_utilization_pct,
        "account_age_days":       req.account_age_days,
        "customer_age":           req.customer_age,
        "credit_limit":           req.credit_limit,
        "risk_score":             risk_score,
        "risk_x_foreign":         risk_score * req.is_foreign,
        "high_amount":            1 if req.amount > 5000 else 0,
        "category_enc":           cat_enc,
        "dow_enc":                dow_enc,
    }
    return pd.DataFrame([row])[FEATURES]


def get_alert_reasons(req: TransactionRequest, prob: float) -> list:
    reasons = []
    if req.is_foreign and req.amount > 500:
        reasons.append(f"Foreign transaction of ₹{req.amount:,.0f}")
    if req.velocity_1h > 4:
        reasons.append(f"High velocity: {req.velocity_1h} transactions in 1 hour")
    if req.distance_from_home_km > 1000:
        reasons.append(f"Transaction {req.distance_from_home_km:.0f}km from home location")
    if req.declined_last_24h >= 3:
        reasons.append(f"{req.declined_last_24h} declined transactions in last 24 hours")
    if req.transaction_hour >= 23 or req.transaction_hour <= 5:
        reasons.append(f"Unusual hour: {req.transaction_hour}:00")
    if prob > 0.65:
        reasons.append(f"ML model high confidence: {prob*100:.1f}% fraud probability")
    return reasons if reasons else ["Flagged by ML model ensemble"]


@app.post("/score", response_model=FraudResponse)
async def score_transaction(req: TransactionRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded. Run train_model.py first.")
    X = compute_features(req)
    prob = float(MODEL.predict_proba(X)[0][1])
    flag = prob >= 0.35
    tier = (
        "Critical" if prob >= 0.65 else
        "High"     if prob >= 0.40 else
        "Medium"   if prob >= 0.20 else "Low"
    )
    return FraudResponse(
        transaction_id=req.transaction_id,
        fraud_probability=round(prob, 4),
        fraud_flag=flag,
        risk_tier=tier,
        alert_reasons=get_alert_reasons(req, prob),
        scored_at=datetime.utcnow().isoformat(),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "version": "1.0.0"}


@app.get("/stats")
async def stats():
    import json
    try:
        with open("data/model_summary.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, "Run train_model.py to generate stats")
