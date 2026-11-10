-- Portfolio views used for the hypothesis tests and the dashboard.
-- Each query returns: segment, applications, defaults, default_rate.

-- name: by_bureau_history
SELECT CASE has_bureau_history WHEN 1 THEN 'Has bureau history' ELSE 'Thin file (no bureau record)' END AS segment,
       COUNT(*) AS applications, SUM(TARGET) AS defaults, AVG(TARGET) AS default_rate
FROM model_features GROUP BY has_bureau_history ORDER BY default_rate DESC;

-- name: by_income_type
SELECT NAME_INCOME_TYPE AS segment,
       COUNT(*) AS applications, SUM(TARGET) AS defaults, AVG(TARGET) AS default_rate
FROM model_features GROUP BY NAME_INCOME_TYPE HAVING COUNT(*) >= 1000 ORDER BY default_rate DESC;

-- name: by_education
SELECT NAME_EDUCATION_TYPE AS segment,
       COUNT(*) AS applications, SUM(TARGET) AS defaults, AVG(TARGET) AS default_rate
FROM model_features GROUP BY NAME_EDUCATION_TYPE HAVING COUNT(*) >= 1000 ORDER BY default_rate DESC;

-- name: by_contract_type
SELECT NAME_CONTRACT_TYPE AS segment,
       COUNT(*) AS applications, SUM(TARGET) AS defaults, AVG(TARGET) AS default_rate
FROM model_features GROUP BY NAME_CONTRACT_TYPE ORDER BY default_rate DESC;

-- name: by_credit_to_income_band
SELECT CASE WHEN credit_to_income < 2 THEN '1: <2x income'
            WHEN credit_to_income < 4 THEN '2: 2-4x income'
            WHEN credit_to_income < 6 THEN '3: 4-6x income'
            ELSE '4: 6x+ income' END AS segment,
       COUNT(*) AS applications, SUM(TARGET) AS defaults, AVG(TARGET) AS default_rate
FROM model_features GROUP BY segment ORDER BY segment;
