-- Bureau history aggregated to one row per applicant (SK_ID_CURR).
-- Every bureau field is recorded as of the application date, so these
-- aggregates use only information available at decision time (no leakage).
DROP TABLE IF EXISTS bureau_features;
CREATE TABLE bureau_features AS
SELECT
    SK_ID_CURR,
    COUNT(*)                                                        AS bur_n_loans,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)       AS bur_n_active,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END)       AS bur_n_closed,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Bad debt' THEN 1 ELSE 0 END)     AS bur_n_bad_debt,
    AVG(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1.0 ELSE 0.0 END)   AS bur_active_share,
    SUM(CASE WHEN CREDIT_TYPE = 'Credit card' THEN 1 ELSE 0 END)    AS bur_n_credit_cards,
    SUM(CASE WHEN CREDIT_TYPE = 'Microloan' THEN 1 ELSE 0 END)      AS bur_n_microloans,
    -- recency and tenure of credit history (DAYS_* are negative = days before application)
    -MAX(DAYS_CREDIT)                                               AS bur_days_since_latest,
    -MIN(DAYS_CREDIT)                                               AS bur_history_length_days,
    -- delinquency signals
    MAX(CREDIT_DAY_OVERDUE)                                         AS bur_max_days_overdue_now,
    SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END)         AS bur_n_overdue_now,
    MAX(AMT_CREDIT_MAX_OVERDUE)                                     AS bur_max_amt_overdue_ever,
    SUM(COALESCE(AMT_CREDIT_SUM_OVERDUE, 0))                        AS bur_sum_amt_overdue_now,
    SUM(CNT_CREDIT_PROLONG)                                         AS bur_n_prolongations,
    -- exposure on active credit
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active'
             THEN COALESCE(AMT_CREDIT_SUM_DEBT, 0) ELSE 0 END)      AS bur_active_debt,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active'
             THEN COALESCE(AMT_CREDIT_SUM, 0) ELSE 0 END)           AS bur_active_credit,
    -- loans opened in the last year: a "credit hungry" signal
    SUM(CASE WHEN DAYS_CREDIT >= -365 THEN 1 ELSE 0 END)            AS bur_n_opened_last_year
FROM bureau
GROUP BY SK_ID_CURR;

CREATE INDEX IF NOT EXISTS ix_bureau_features ON bureau_features (SK_ID_CURR);
