-- ============================================================
-- FinShield — Financial Fraud Detection System
-- Database Architecture & Analytical SQL Queries
-- Compatible with: PostgreSQL / SQL Server / SQLite
-- ============================================================

-- ── SCHEMA DESIGN ────────────────────────────────────────────────────────────

CREATE TABLE dim_customer (
    customer_id       VARCHAR(10) PRIMARY KEY,
    age               INT,
    credit_limit      DECIMAL(12,2),
    account_age_days  INT,
    home_country      CHAR(2),
    risk_profile      VARCHAR(10)  -- LOW / MEDIUM / HIGH
);

CREATE TABLE dim_merchant (
    merchant_id        VARCHAR(10) PRIMARY KEY,
    merchant_category  VARCHAR(30),
    avg_txn_amount     DECIMAL(10,2),
    fraud_rate_hist    DECIMAL(5,4)
);

CREATE TABLE dim_date (
    date_key     DATE PRIMARY KEY,
    day_of_week  VARCHAR(10),
    month        INT,
    quarter      INT,
    year         INT,
    is_weekend   BIT
);

CREATE TABLE fact_transactions (
    transaction_id          VARCHAR(12) PRIMARY KEY,
    customer_id             VARCHAR(10) REFERENCES dim_customer(customer_id),
    merchant_id             VARCHAR(10) REFERENCES dim_merchant(merchant_id),
    date_key                DATE        REFERENCES dim_date(date_key),
    transaction_time        TIME,
    transaction_hour        INT,
    amount                  DECIMAL(12,2),
    country                 CHAR(2),
    is_foreign              BIT,
    is_night_txn            BIT,
    velocity_1h             INT,
    velocity_24h            INT,
    distance_from_home_km   DECIMAL(10,2),
    is_new_merchant         BIT,
    declined_last_24h       INT,
    credit_utilization_pct  DECIMAL(5,2),
    risk_score              INT,
    risk_tier               VARCHAR(10),
    is_fraud                BIT,
    ml_fraud_probability    DECIMAL(5,4),   -- populated after ML scoring
    ml_fraud_flag           BIT             -- populated after ML scoring
);

CREATE TABLE fraud_alerts (
    alert_id         SERIAL PRIMARY KEY,
    transaction_id   VARCHAR(12) REFERENCES fact_transactions(transaction_id),
    alert_timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_reason     VARCHAR(200),
    alert_severity   VARCHAR(10),  -- LOW / MEDIUM / HIGH / CRITICAL
    resolved         BIT DEFAULT 0,
    resolved_by      VARCHAR(50),
    resolved_at      TIMESTAMP
);

-- ── ANALYTICAL QUERIES ───────────────────────────────────────────────────────

-- 1. Overall fraud KPI summary
SELECT
    COUNT(*)                                          AS total_transactions,
    SUM(is_fraud)                                     AS total_fraud,
    ROUND(AVG(CAST(is_fraud AS FLOAT)) * 100, 2)     AS fraud_rate_pct,
    SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS total_fraud_amount,
    AVG(CASE WHEN is_fraud = 1 THEN amount END)       AS avg_fraud_amount,
    AVG(CASE WHEN is_fraud = 0 THEN amount END)       AS avg_legit_amount
FROM fact_transactions;

-- 2. Fraud rate by merchant category (PowerBI bar chart source)
SELECT
    d.merchant_category,
    COUNT(*)                                               AS total_txns,
    SUM(f.is_fraud)                                        AS fraud_count,
    ROUND(AVG(CAST(f.is_fraud AS FLOAT)) * 100, 2)        AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN f.is_fraud=1 THEN f.amount END), 0) AS fraud_amount
FROM fact_transactions f
JOIN dim_merchant d ON f.merchant_id = d.merchant_id
GROUP BY d.merchant_category
ORDER BY fraud_rate_pct DESC;

-- 3. Hourly fraud pattern (PowerBI line chart — identifies night fraud spike)
SELECT
    transaction_hour,
    COUNT(*)                                          AS total_txns,
    SUM(is_fraud)                                     AS fraud_count,
    ROUND(AVG(CAST(is_fraud AS FLOAT)) * 100, 2)     AS fraud_rate_pct
FROM fact_transactions
GROUP BY transaction_hour
ORDER BY transaction_hour;

-- 4. Geographic fraud heatmap data
SELECT
    country,
    COUNT(*)                                          AS total_txns,
    SUM(is_fraud)                                     AS fraud_count,
    ROUND(AVG(CAST(is_fraud AS FLOAT)) * 100, 2)     AS fraud_rate_pct,
    SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END)  AS fraud_exposure
FROM fact_transactions
GROUP BY country
ORDER BY fraud_rate_pct DESC;

-- 5. Risk tier distribution
SELECT
    risk_tier,
    COUNT(*)                                          AS transaction_count,
    SUM(is_fraud)                                     AS actual_fraud,
    ROUND(AVG(CAST(is_fraud AS FLOAT)) * 100, 2)     AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                             AS avg_amount
FROM fact_transactions
GROUP BY risk_tier
ORDER BY CASE risk_tier
    WHEN 'Critical' THEN 1 WHEN 'High' THEN 2
    WHEN 'Medium' THEN 3 ELSE 4 END;

-- 6. Monthly fraud trend (PowerBI time series)
SELECT
    d.year,
    d.month,
    COUNT(*)                                          AS total_txns,
    SUM(f.is_fraud)                                   AS fraud_count,
    ROUND(AVG(CAST(f.is_fraud AS FLOAT)) * 100, 2)   AS fraud_rate_pct,
    SUM(CASE WHEN f.is_fraud=1 THEN f.amount ELSE 0 END) AS fraud_amount
FROM fact_transactions f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.month
ORDER BY d.year, d.month;

-- 7. High-velocity customers (fraud investigation list)
SELECT
    customer_id,
    COUNT(*)                                          AS txn_count,
    SUM(is_fraud)                                     AS fraud_count,
    MAX(velocity_24h)                                 AS max_velocity_24h,
    SUM(amount)                                       AS total_amount,
    AVG(risk_score)                                   AS avg_risk_score
FROM fact_transactions
WHERE velocity_24h > 10
GROUP BY customer_id
HAVING SUM(is_fraud) > 0
ORDER BY fraud_count DESC
LIMIT 50;

-- 8. DAX-equivalent: Running fraud total by date (use in PowerBI calculated column)
-- In PowerBI DAX:
-- Running Fraud Amount =
-- CALCULATE(
--     SUM(transactions[amount]),
--     transactions[is_fraud] = 1,
--     FILTER(ALL(dim_date), dim_date[date_key] <= MAX(dim_date[date_key]))
-- )

-- 9. Foreign transaction fraud comparison
SELECT
    CASE WHEN is_foreign = 1 THEN 'Foreign' ELSE 'Domestic' END AS txn_type,
    COUNT(*)                                          AS total_txns,
    SUM(is_fraud)                                     AS fraud_count,
    ROUND(AVG(CAST(is_fraud AS FLOAT)) * 100, 2)     AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                             AS avg_amount
FROM fact_transactions
GROUP BY is_foreign;

-- 10. Alert generation — insert rule-based fraud alerts
INSERT INTO fraud_alerts (transaction_id, alert_reason, alert_severity)
SELECT
    transaction_id,
    CASE
        WHEN is_foreign = 1 AND amount > 500  AND is_night_txn = 1
            THEN 'Foreign high-value night transaction'
        WHEN velocity_1h > 5
            THEN 'High velocity: ' + CAST(velocity_1h AS VARCHAR) + ' txns in 1 hour'
        WHEN distance_from_home_km > 5000
            THEN 'Transaction ' + CAST(ROUND(distance_from_home_km,0) AS VARCHAR) + 'km from home'
        WHEN declined_last_24h >= 3
            THEN 'Multiple declines: ' + CAST(declined_last_24h AS VARCHAR) + ' in 24h'
        ELSE 'ML model high fraud probability'
    END AS alert_reason,
    CASE
        WHEN risk_score >= 70 THEN 'CRITICAL'
        WHEN risk_score >= 45 THEN 'HIGH'
        WHEN risk_score >= 20 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS alert_severity
FROM fact_transactions
WHERE is_fraud = 1 OR ml_fraud_flag = 1;
