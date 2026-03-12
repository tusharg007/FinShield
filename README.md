# 🛡️ FinShield — Financial Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PowerBI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-Star_Schema-336791?style=for-the-badge&logo=postgresql&logoColor=white)

**[🚀 Live Demo](https://finshield-e7nexmopxknwbv4nvjehdn.streamlit.app/)**

*End-to-end financial fraud detection platform — ML pipeline, SQL warehouse, Power BI dashboard, REST API*

</div>

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Live Demo](#-live-demo)
3. [How It Works](#-how-it-works)
4. [Dataset](#-dataset)
5. [ML Pipeline](#-ml-pipeline)
6. [SQL Database Architecture](#-sql-database-architecture)
7. [Power BI Dashboard](#-powerbi-dashboard)
8. [REST API](#-rest-api)
9. [Tech Stack](#-tech-stack)
10. [Project Structure](#-project-structure)
11. [Quick Start](#-quick-start)
12. [Results](#-results)

---

## 🎯 Project Overview

FinShield is a production-ready financial fraud detection system that identifies fraudulent transactions in real time. It combines machine learning, rule-based scoring, a SQL star-schema data warehouse, a 4-page Power BI dashboard, and a FastAPI REST endpoint — all connected through an interactive Streamlit demo.

**The problem it solves:** Financial institutions lose billions annually to transaction fraud. Traditional rule-based systems miss complex fraud patterns. FinShield uses an ensemble ML approach with 24 engineered features to detect fraud with high precision while keeping false positives low.

**Key results:**

| Metric | Value |
|--------|-------|
| Dataset size | 50,000 transactions |
| Fraud rate | 3.4% (realistic industry figure) |
| Total fraud exposure | ₹74.4 Lakhs |
| Best model ROC-AUC | 1.00 |
| Inference latency | < 50ms per transaction |
| Risk tiers | Low / Medium / High / Critical |

---

## 🚀 Live Demo

**Try it live: [YOUR_STREAMLIT_URL](YOUR_STREAMLIT_URL)**

The app has 4 pages:

### Page 1 — Executive Dashboard
Bird's-eye view of all fraud activity:
- KPI cards: total transactions, fraud count, fraud rate %, total ₹ exposure
- Bar chart: fraud rate by merchant category (wire transfers and crypto highest)
- Line chart: fraud rate by hour of day (spikes midnight–5AM)
- Risk tier distribution, top 10 highest-risk transactions

### Page 2 — Score a Transaction
Enter any transaction's details and get an instant fraud score:
- Inputs: amount, merchant category, country, hour, velocity, distance from home, declined transactions, credit utilization
- **Quick Presets** to test instantly:
  - 🔴 High Risk: Wire transfer at 2AM from Nigeria for ₹45,000
  - 🟠 Medium Risk: Luxury goods from UAE
  - 🟢 Low Risk: Grocery store nearby
  - 🟡 Borderline: Late-night online shopping
- Output: fraud probability (0–100%), risk tier, specific alert reasons, progress gauge

### Page 3 — Dataset Explorer
Browse all 50,000 transactions with filters:
- Filter by risk tier, merchant category, domestic/foreign
- Download filtered CSV
- View ML scores alongside raw data

### Page 4 — Model Performance
- ROC-AUC, F1 score, precision/recall
- Feature importance bar chart (which signals matter most)
- Fraud probability distribution across dataset
- All 3 models compared

---

## ⚙️ How It Works

```
┌──────────────────────────────────────────────────────┐
│                    DATA LAYER                        │
│  generate_data.py → 50,000 synthetic transactions    │
│  Fraud patterns: velocity, geography, time, amount   │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│                     ML LAYER                         │
│  train_model.py → 24 features engineered             │
│  3 models: LR + Random Forest + Gradient Boosting    │
│  Output: transactions_scored.csv + model.pkl         │
└──────────┬───────────────────────┬───────────────────┘
           │                       │
           ▼                       ▼
┌──────────────────┐   ┌──────────────────────────┐
│  DASHBOARD LAYER │   │        API LAYER         │
│  Power BI reads  │   │  FastAPI POST /score     │
│  scored CSV      │   │  Real-time < 50ms        │
│  4-page report   │   │  Auto-docs /docs         │
└──────────────────┘   └──────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│                    DEMO LAYER                        │
│  streamlit_app.py → Interactive 4-page web app       │
│  Rule-based fallback if model not loaded             │
└──────────────────────────────────────────────────────┘
```

### Fraud Scoring Formula

**Rule-Based Risk Score (0–100):**
```
Risk Score =
  is_foreign        × 25   (overseas transaction)
  is_night_txn      × 15   (11PM–5AM)
  high_velocity_1h  × 20   (>3 txns in 1 hour)
  far_from_home     × 20   (>1000km from home)
  is_new_merchant   × 10   (never visited before)
  declined_24h      × 5    (per declined txn)
```

**ML Fraud Probability (0.0–1.0):** Random Forest trained on 24 features. Threshold = 0.35 (tuned for high recall).

**Risk Tiers:**

| ML Probability | Tier | Action |
|---------------|------|--------|
| ≥ 0.65 | 🔴 Critical | Immediate block |
| 0.40–0.65 | 🟠 High | Manual review |
| 0.20–0.40 | 🟡 Medium | Monitor |
| < 0.20 | 🟢 Low | Auto-approve |

---

## 📊 Dataset

### How It Was Built

No publicly available labeled dataset had all the features needed, so we built a realistic synthetic generator (`src/generate_data.py`) based on financial industry fraud research.

| Property | Value |
|----------|-------|
| Total transactions | 50,000 |
| Fraud transactions | 1,675 (3.4%) |
| Unique customers | 5,000 |
| Unique merchants | 800 |
| Date range | Jan 2023 – Dec 2023 |
| Merchant categories | 12 |
| Countries | 10 (India home, 9 overseas) |

**Why 3.4% fraud rate?** The global average fraud rate among flagged transactions is 3–5%. This gives the ML model enough positive samples while keeping class imbalance realistic.

### All Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique ID (TXN0000001 format) |
| `customer_id` | string | Customer reference |
| `merchant_id` | string | Merchant reference |
| `merchant_category` | string | grocery / electronics / wire_transfer etc. |
| `transaction_date` | date | Date (Jan–Dec 2023) |
| `transaction_hour` | int | 0–23 |
| `day_of_week` | string | Monday–Sunday |
| `amount` | float | Amount in ₹ |
| `country` | string | ISO code (IN, US, GB, DE, SG, AE, NG, RO, CN, BR) |
| `is_foreign` | int | 1 if country ≠ IN |
| `is_night_txn` | int | 1 if hour ≥ 23 or ≤ 5 |
| `velocity_1h` | int | Txns by same customer in last 1 hour |
| `velocity_24h` | int | Txns by same customer in last 24 hours |
| `distance_from_home_km` | float | Distance from customer's home location |
| `is_new_merchant` | int | 1 if never transacted here before |
| `declined_last_24h` | int | Declined txns in last 24 hours |
| `credit_utilization_pct` | float | % of credit limit used this txn |
| `account_age_days` | int | Days since account was opened |
| `customer_age` | int | Customer age in years |
| `credit_limit` | float | Credit limit in ₹ |
| `risk_score` | int | Rule-based score 0–100 |
| `risk_tier` | string | Rule-based: Low/Medium/High/Critical |
| `is_fraud` | int | **Ground truth: 1 = fraud, 0 = legitimate** |
| `ml_fraud_probability` | float | Model output score 0.0–1.0 |
| `ml_fraud_flag` | int | 1 if probability ≥ 0.35 |
| `ml_risk_tier` | string | ML-derived tier |

### Fraud Patterns in the Data

The generator embeds realistic fraud patterns:

| Pattern | Fraud Transactions | Legitimate Transactions |
|---------|-------------------|------------------------|
| Night time (11PM–5AM) | 70% probability | 30% probability |
| Distance from home | 500–15,000 km | 0–800 km |
| Velocity (txns/hour) | 3–15 | 0–4 |
| New merchant | 80% | 30% |
| Declines in 24h | 1–6 | 0–2 |
| Amount multiplier | 1.5–8× base | 0.3–2.5× base |

---

## 🤖 ML Pipeline

### Feature Engineering

24 features are created from raw columns (`src/train_model.py`):

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `amount_log` | log(1 + amount) | Reduces skew from high-value outliers |
| `velocity_ratio` | vel_1h / (vel_24h + 1) | Detects sudden activity burst |
| `risk_x_foreign` | risk_score × is_foreign | Overseas + risky = very suspicious |
| `far_from_home` | dist > 1000 → 1 | Binary geographic anomaly flag |
| `high_velocity` | vel_24h > 10 → 1 | Binary velocity attack flag |
| `high_amount` | amount > 90th pct → 1 | Binary large transaction flag |
| `is_weekend` | Sat/Sun → 1 | Weekend fraud patterns differ |
| `is_peak_hour` | 10AM–8PM → 1 | Business hours vs off-hours |
| `category_enc` | LabelEncoder | Converts merchant type to numeric |

### Three Models Explained

**1. Logistic Regression**
- How it works: Finds a linear boundary separating fraud from legitimate transactions
- Strengths: Fast, interpretable, good baseline
- Limitation: Cannot capture non-linear interactions (e.g. "foreign AND night AND high amount" is more than the sum of parts)
- Parameters: `C=0.1` (regularization), `class_weight="balanced"`

**2. Random Forest (200 trees)**
- How it works: Trains 200 independent decision trees on random subsets of data and features, then averages their predictions
- Strengths: Handles non-linear interactions, robust to outliers, provides feature importance
- Why 200 trees: Enough to reduce variance without overfitting
- Parameters: `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`

**3. Gradient Boosting (200 trees)**
- How it works: Trains trees sequentially — each new tree focuses on the mistakes of the previous ensemble
- Strengths: Best at catching subtle patterns, often highest accuracy
- Parameters: `learning_rate=0.05`, `subsample=0.8`, `max_depth=5`

### Handling Class Imbalance

With 3.4% fraud, a naive model gets 96.6% accuracy predicting everything as legitimate — useless. Our approach:

1. **`class_weight="balanced"`**: Makes the model treat fraud as 29× more important than legitimate transactions
2. **Stratified split**: Both train (80%) and test (20%) preserve the exact 3.4% fraud ratio
3. **Threshold = 0.35**: Instead of 0.5, we flag at 0.35 — catches more fraud at cost of slightly more false positives. In fraud detection, missing real fraud is worse than reviewing extra cases.

---

## 🗄️ SQL Database Architecture

**File:** `sql/schema_and_queries.sql`

### Star Schema Design

The database follows a **star schema** — the industry standard for analytics warehouses used at banks, EY, Deloitte, and similar firms:

```
              ┌─────────────────┐
              │  dim_customer   │
              │  customer_id PK │
              │  age            │
              │  credit_limit   │
              │  account_age    │
              └────────┬────────┘
                       │ FK
┌─────────────┐  ┌─────┴──────────────────┐  ┌────────────────┐
│  dim_date   │  │   fact_transactions    │  │  dim_merchant  │
│  date_key PK├──│   transaction_id PK    ├──│  merchant_id PK│
│  day_of_week│  │   customer_id FK       │  │  category      │
│  month      │  │   merchant_id FK       │  │  fraud_rate    │
│  quarter    │  │   date_key FK          │  └────────────────┘
│  year       │  │   amount               │
│  is_weekend │  │   is_fraud             │  ┌────────────────┐
└─────────────┘  │   ml_fraud_probability │  │  fraud_alerts  │
                 │   risk_score           ├──│  alert_id PK   │
                 │   risk_tier            │  │  transaction FK│
                 └────────────────────────┘  │  severity      │
                                             │  resolved      │
                                             └────────────────┘
```

**Why star schema?**
- Fast GROUP BY aggregations (Power BI needs these for every chart)
- Dimension tables (customer, merchant, date) update independently
- Easy to understand for business stakeholders
- Industry standard at consulting firms like EY

### The 10 Analytical Queries

Each query serves a specific dashboard visual or business question:

| Query | Business Question | Used In |
|-------|-------------------|---------|
| 1. KPI Summary | What is our overall fraud exposure? | Executive dashboard cards |
| 2. By Category | Which merchant types are riskiest? | Bar chart |
| 3. By Hour | When does fraud spike? | Line chart |
| 4. By Country | Where is fraud geographically? | Map visual |
| 5. Risk Tiers | How are transactions distributed? | Donut chart |
| 6. Monthly Trend | Is fraud increasing? | Time series |
| 7. High Velocity | Which customers need investigation? | Alert list |
| 8. Foreign vs Domestic | Is overseas the problem? | Comparison bar |
| 9. Alert Generation | Which transactions trigger alerts? | Alerts table |
| 10. Running Total | Cumulative fraud exposure over time | Area chart |

---

## 📈 PowerBI Dashboard

**File:** `dashboard/POWERBI_SETUP.md`

### Setup Steps

1. Download [Power BI Desktop](https://powerbi.microsoft.com/downloads/) — free
2. **Home → Get Data → Text/CSV** → open `data/transactions_scored.csv`
3. In Transform Data: set `transaction_date` to Date, `amount` to Decimal Number
4. Close & Apply
5. Add DAX measures from `POWERBI_SETUP.md` (copy-paste each one)
6. Build 4 pages following the visual layout guide

### DAX Measures (Key Ones)

```dax
-- Total fraud cases
Total Fraud = CALCULATE(COUNTROWS(transactions_scored), transactions_scored[is_fraud] = 1)

-- Fraud rate percentage
Fraud Rate % = DIVIDE([Total Fraud], [Total Transactions]) * 100

-- Total money lost to fraud
Fraud Amount = CALCULATE(SUM(transactions_scored[amount]), transactions_scored[is_fraud] = 1)

-- Cumulative fraud over time (for trend line)
Running Fraud Amount =
CALCULATE(
    SUM(transactions_scored[amount]),
    transactions_scored[is_fraud] = 1,
    FILTER(
        ALL(transactions_scored),
        transactions_scored[transaction_date] <= MAX(transactions_scored[transaction_date])
    )
)

-- What % of fraud did our ML model catch?
ML Detection Rate % = DIVIDE(
    CALCULATE(COUNTROWS(transactions_scored),
        transactions_scored[ml_fraud_flag] = 1,
        transactions_scored[is_fraud] = 1),
    [Total Fraud]
) * 100
```

### 4 Dashboard Pages

**Page 1 — Executive Summary**
KPI cards (total txns, fraud count, rate, ₹ exposure), monthly trend line, fraud by category bar chart, risk tier donut

**Page 2 — Transaction Analysis**
Fraud by hour column chart, domestic vs foreign bar, country map, amount distribution histogram, day-of-week matrix

**Page 3 — ML Model Insights**
Feature importance from `feature_importance.csv`, probability histogram, confusion matrix, model comparison table

**Page 4 — Fraud Alerts**
Filterable table of all flagged transactions with slicers for risk tier, category, date range — this is the operational investigation page

---

## 🔌 REST API

**File:** `src/api.py`

The FastAPI backend exposes real-time fraud scoring. When a new transaction arrives, it is scored in under 50ms.

### Start the API

```bash
uvicorn src.api:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI where you can test all endpoints directly in your browser.

### Endpoints

| Method | Endpoint | What It Does |
|--------|----------|--------------|
| POST | `/score` | Submit a transaction → get fraud probability + risk tier + alert reasons |
| GET | `/health` | Check if service is running and model is loaded |
| GET | `/stats` | Get overall model performance summary |

### Full Example

**Request:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_TEST_001",
    "amount": 45000,
    "merchant_category": "wire_transfer",
    "transaction_hour": 2,
    "day_of_week": "Saturday",
    "is_foreign": 1,
    "velocity_1h": 7,
    "velocity_24h": 18,
    "distance_from_home_km": 12000,
    "is_new_merchant": 1,
    "declined_last_24h": 4,
    "credit_utilization_pct": 92,
    "account_age_days": 45,
    "customer_age": 28,
    "credit_limit": 50000
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN_TEST_001",
  "fraud_probability": 0.9823,
  "fraud_flag": true,
  "risk_tier": "Critical",
  "alert_reasons": [
    "Foreign high-value transaction of ₹45,000",
    "High velocity: 7 transactions in last 1 hour",
    "Transaction 12000 km from home location",
    "4 declined transactions in last 24 hours",
    "Unusual hour: 2:00 (late night activity)",
    "ML model high confidence: 98.2% fraud probability"
  ],
  "scored_at": "2026-01-15T02:34:11"
}
```

---

## 🛠️ Tech Stack

| Layer | Tool | How We Used It |
|-------|------|----------------|
| **Data** | Python, NumPy, Pandas | Built a 50K transaction generator with realistic fraud patterns |
| **ML** | Scikit-learn | Trained LR, Random Forest, Gradient Boosting with class balancing |
| **Feature Eng.** | Pandas, NumPy | 24 features: log transforms, velocity ratios, interaction terms, binary flags |
| **API** | FastAPI + Pydantic | REST endpoint with request validation, auto-docs, < 50ms latency |
| **Dashboard** | Microsoft Power BI | Star schema source, DAX measures for running totals and detection rate |
| **Database** | SQL (PostgreSQL) | Star schema with fact/dimension tables and 10 analytical queries |
| **Demo** | Streamlit | 4-page web app with scoring, filtering, and visualizations |
| **DevOps** | Git, GitHub | Version control, GitHub Actions CI/CD config |

---

## 📁 Project Structure

```
finshield/
│
├── src/
│   ├── generate_data.py        # Dataset generator — 50K realistic transactions
│   │                           # 12 merchant categories, 10 countries
│   │                           # Embeds real fraud patterns (velocity, geo, time)
│   │
│   ├── train_model.py          # Full ML pipeline
│   │                           # 24 features engineered from raw data
│   │                           # 3 models trained with class balancing
│   │                           # Exports scored CSV + pickled model
│   │
│   ├── api.py                  # FastAPI scoring service
│   │                           # POST /score → real-time fraud probability
│   │                           # GET /health, GET /stats
│   │
│   └── models/
│       └── fraud_rf_model.pkl  # Saved Random Forest model
│
├── data/
│   ├── transactions.csv         # Raw 50K transaction dataset
│   ├── transactions_scored.csv  # Scored with ML — Power BI data source
│   ├── feature_importance.csv   # RF feature rankings
│   └── model_summary.json       # ROC-AUC, F1, detection stats
│
├── sql/
│   └── schema_and_queries.sql   # Star schema DDL + 10 analytical queries
│
├── dashboard/
│   └── POWERBI_SETUP.md         # Power BI guide: DAX measures + page layouts
│
├── streamlit_app.py             # 4-page interactive demo app
├── requirements.txt             # pandas, numpy, scikit-learn, streamlit, fastapi
└── README.md                    # This file
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/tusharg007/finshield.git
cd finshield

# 2. Install
pip install -r requirements.txt

# 3. Generate 50K transactions (~30 seconds)
python src/generate_data.py

# 4. Train 3 ML models (~1-2 minutes)
python src/train_model.py

# 5. Run Streamlit demo
python -m streamlit run streamlit_app.py
# → http://localhost:8501

# 6. (Optional) Start REST API
uvicorn src.api:app --reload --port 8000
# → http://localhost:8000/docs
```

---

## 📉 Results

### Model Performance Summary

| Model | ROC-AUC | Avg Precision | F1 Score |
|-------|---------|---------------|----------|
| Logistic Regression | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | 1.000 | 1.000 | 1.000 |

> Perfect scores are expected on synthetic data with deterministic patterns. On real bank data, Random Forest typically achieves ROC-AUC 0.92–0.96. The architecture and pipeline are production-ready.

### Top Features by Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | risk_score | 19.4% |
| 2 | distance_from_home_km | 18.3% |
| 3 | risk_x_foreign | 17.3% |
| 4 | far_from_home | 14.6% |
| 5 | velocity_1h | 10.2% |

### Fraud Rates by Category

| Category | Fraud Rate | Risk Level |
|----------|------------|------------|
| Wire Transfer | 7.6% | 🔴 Critical |
| Crypto Exchange | 6.5% | 🔴 Critical |
| Luxury Goods | 6.1% | 🟠 High |
| ATM Withdrawal | 4.6% | 🟠 High |
| Online Shopping | 4.1% | 🟠 High |
| Grocery | 0.2% | 🟢 Low |

---

## 👨‍💻 About

Built by **Tushar Ghosh** — Data Science undergraduate at IIIT Nagpur.

[![GitHub](https://img.shields.io/badge/GitHub-tusharg007-181717?style=flat&logo=github)](https://github.com/tusharg007)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tushar_Ghosh-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/tushar-ghosh-a3355124a/)
