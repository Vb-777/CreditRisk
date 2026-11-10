-- One modelling row per application: cleaned application fields,
-- affordability ratios, and the bureau aggregates from 01_bureau_features.sql.
-- CODE_GENDER is deliberately excluded: gender is a protected attribute and
-- is not used for credit decisions.
DROP TABLE IF EXISTS model_features;
CREATE TABLE model_features AS
SELECT
    a.SK_ID_CURR,
    a.TARGET,

    -- product and applicant profile
    a.NAME_CONTRACT_TYPE,
    a.NAME_INCOME_TYPE,
    a.NAME_EDUCATION_TYPE,
    a.NAME_FAMILY_STATUS,
    a.NAME_HOUSING_TYPE,
    a.OCCUPATION_TYPE,
    a.ORGANIZATION_TYPE,
    a.FLAG_OWN_CAR,
    a.FLAG_OWN_REALTY,
    a.CNT_CHILDREN,
    a.CNT_FAM_MEMBERS,
    a.REGION_RATING_CLIENT_W_CITY,
    a.REGION_POPULATION_RELATIVE,

    -- amounts
    a.AMT_INCOME_TOTAL,
    a.AMT_CREDIT,
    a.AMT_ANNUITY,
    a.AMT_GOODS_PRICE,

    -- affordability ratios (NULLIF guards against division by zero)
    1.0 * a.AMT_CREDIT  / NULLIF(a.AMT_INCOME_TOTAL, 0)   AS credit_to_income,
    1.0 * a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0)   AS annuity_to_income,
    1.0 * a.AMT_ANNUITY / NULLIF(a.AMT_CREDIT, 0)         AS annuity_to_credit,
    1.0 * a.AMT_CREDIT  / NULLIF(a.AMT_GOODS_PRICE, 0)    AS credit_to_goods,

    -- time fields converted to positive years; DAYS_EMPLOYED = 365243 is a
    -- placeholder for "not employed" (pensioners), so it becomes NULL + a flag
    -a.DAYS_BIRTH / 365.25                           AS age_years,
    CASE WHEN a.DAYS_EMPLOYED = 365243 THEN NULL
         ELSE -a.DAYS_EMPLOYED / 365.25 END          AS employed_years,
    CASE WHEN a.DAYS_EMPLOYED = 365243 THEN 1 ELSE 0 END AS flag_not_employed,
    -a.DAYS_ID_PUBLISH / 365.25                      AS id_age_years,
    -a.DAYS_REGISTRATION / 365.25                    AS registration_age_years,
    -a.DAYS_LAST_PHONE_CHANGE / 365.25               AS phone_change_years,
    a.OWN_CAR_AGE,

    -- external scores (third-party / alternate data)
    a.EXT_SOURCE_1,
    a.EXT_SOURCE_2,
    a.EXT_SOURCE_3,

    -- contactability and social-circle delinquency
    a.FLAG_WORK_PHONE,
    a.FLAG_PHONE,
    a.DEF_30_CNT_SOCIAL_CIRCLE,
    a.AMT_REQ_CREDIT_BUREAU_YEAR,

    -- bureau history; NULL bureau row means a thin-file applicant
    CASE WHEN b.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS has_bureau_history,
    b.bur_n_loans,
    b.bur_n_active,
    b.bur_active_share,
    b.bur_n_credit_cards,
    b.bur_n_microloans,
    b.bur_days_since_latest,
    b.bur_history_length_days,
    b.bur_max_days_overdue_now,
    b.bur_n_overdue_now,
    b.bur_max_amt_overdue_ever,
    b.bur_sum_amt_overdue_now,
    b.bur_n_prolongations,
    b.bur_n_bad_debt,
    b.bur_n_opened_last_year,
    1.0 * b.bur_active_debt / NULLIF(b.bur_active_credit, 0)       AS bur_utilisation,
    1.0 * b.bur_active_debt / NULLIF(a.AMT_INCOME_TOTAL, 0)        AS bur_debt_to_income
FROM application a
LEFT JOIN bureau_features b ON b.SK_ID_CURR = a.SK_ID_CURR;
