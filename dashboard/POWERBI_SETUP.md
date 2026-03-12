# FinShield — PowerBI Dashboard Setup Guide

## Data Source
File: `data/transactions_scored.csv`
Rows: 50,000 transactions | 27 columns

---

## Step 1 — Import Data
1. Open Power BI Desktop
2. Home → Get Data → Text/CSV
3. Select `transactions_scored.csv`
4. Click **Transform Data**
5. Set column types:
   - `transaction_date` → Date
   - `amount`, `ml_fraud_probability` → Decimal Number
   - `is_fraud`, `is_foreign`, `is_night_txn` → Whole Number

---

## Step 2 — DAX Measures (paste into New Measure)

```dax
-- KPI: Total Transactions
Total Transactions = COUNTROWS(transactions_scored)

-- KPI: Total Fraud Count
Total Fraud = CALCULATE(COUNTROWS(transactions_scored), transactions_scored[is_fraud] = 1)

-- KPI: Fraud Rate %
Fraud Rate % = DIVIDE([Total Fraud], [Total Transactions]) * 100

-- KPI: Total Fraud Amount (₹)
Fraud Amount = CALCULATE(SUM(transactions_scored[amount]), transactions_scored[is_fraud] = 1)

-- KPI: Average Fraud Amount
Avg Fraud Amount = CALCULATE(AVERAGE(transactions_scored[amount]), transactions_scored[is_fraud] = 1)

-- KPI: ML Detection Rate
ML Detection Rate % = DIVIDE(
    CALCULATE(COUNTROWS(transactions_scored), transactions_scored[ml_fraud_flag] = 1, transactions_scored[is_fraud] = 1),
    [Total Fraud]
) * 100

-- Running fraud total (for trend line)
Running Fraud Amount =
CALCULATE(
    SUM(transactions_scored[amount]),
    transactions_scored[is_fraud] = 1,
    FILTER(
        ALL(transactions_scored),
        transactions_scored[transaction_date] <= MAX(transactions_scored[transaction_date])
    )
)

-- High risk transaction flag
High Risk Flag = IF(transactions_scored[ml_fraud_probability] >= 0.65, "Critical",
                 IF(transactions_scored[ml_fraud_probability] >= 0.40, "High",
                 IF(transactions_scored[ml_fraud_probability] >= 0.20, "Medium", "Low")))
```

---

## Step 3 — Dashboard Pages

### Page 1: Executive Summary
| Visual | Type | Fields |
|--------|------|--------|
| Total Transactions | Card | Total Transactions |
| Fraud Count | Card | Total Fraud |
| Fraud Rate | Card | Fraud Rate % |
| Fraud Exposure | Card | Fraud Amount |
| Fraud by Month | Line Chart | transaction_date (Month), Fraud Rate % |
| Fraud by Category | Bar Chart | merchant_category, Total Fraud |
| Risk Tier Donut | Donut Chart | ml_risk_tier, count |
| KPI Trend | Area Chart | transaction_date, Running Fraud Amount |

### Page 2: Transaction Analysis
| Visual | Type | Fields |
|--------|------|--------|
| Fraud by Hour | Column Chart | transaction_hour, Fraud Rate % |
| Foreign vs Domestic | Clustered Bar | is_foreign, Total Fraud, Fraud Rate % |
| Amount Distribution | Histogram | amount (bin by 500) |
| Fraud by Day of Week | Matrix | day_of_week, is_fraud count |
| Country Map | Filled Map | country, Fraud Rate % |
| Top Risky Customers | Table | customer_id, fraud count, total amount |

### Page 3: ML Model Insights
| Visual | Type | Fields |
|--------|------|--------|
| Probability Distribution | Histogram | ml_fraud_probability |
| Precision-Recall | Line Chart | (import from feature_importance.csv) |
| Feature Importance | Bar Chart | feature, importance (from feature_importance.csv) |
| Confusion Matrix | Matrix | is_fraud, ml_fraud_flag |
| ML vs Rule Score | Scatter | risk_score (x), ml_fraud_probability (y), is_fraud (color) |
| Detection Rate Card | Card | ML Detection Rate % |

### Page 4: Fraud Alerts (Operational)
| Visual | Type | Fields |
|--------|------|--------|
| Alert Table | Table | transaction_id, amount, merchant_category, ml_fraud_probability, ml_risk_tier, country |
| Slicers | Slicer | ml_risk_tier, merchant_category, is_foreign, transaction_date |
| Alert Count by Severity | Donut | ml_risk_tier, count (filtered: ml_fraud_flag=1) |
| Geographic Alerts | Map | country, fraud count |

---

## Step 4 — Formatting Tips
- Theme: Dark / Slate (matches financial dashboard aesthetic)
- KPI Cards: Red for Critical, Orange for High, Yellow for Medium, Green for Low
- Use conditional formatting on tables: red background if ml_fraud_probability > 0.65
- Add company logo placeholder top-left
- Page navigation buttons bottom of each page

---

## Step 5 — Publish
1. File → Publish → Publish to Power BI
2. Select your workspace
3. Share link with "View" permission for portfolio/demo
