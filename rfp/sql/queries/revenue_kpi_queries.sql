-- ================================================================
-- revenue_kpi_queries.sql
-- Core SQL queries for Revenue Forecasting & Variance Analysis
-- Run against the warehouse after the pipeline completes
-- ================================================================


-- ---------------------------------------------------------------
-- 1. Monthly revenue summary — actual vs budget vs prior year
-- ---------------------------------------------------------------
SELECT
    period,
    department,
    SUM(actual_revenue)                                      AS total_actual,
    SUM(budget_revenue)                                      AS total_budget,
    SUM(prior_year_revenue)                                  AS total_prior_year,
    SUM(actual_revenue - budget_revenue)                     AS budget_variance,
    ROUND(100.0 * SUM(actual_revenue - budget_revenue)
          / NULLIF(SUM(budget_revenue), 0), 2)               AS budget_variance_pct,
    ROUND(100.0 * (SUM(actual_revenue) - SUM(prior_year_revenue))
          / NULLIF(SUM(prior_year_revenue), 0), 2)           AS yoy_growth_pct
FROM revenue_actuals
GROUP BY period, department
ORDER BY period, department;


-- ---------------------------------------------------------------
-- 2. Executive summary — total revenue performance by department
-- ---------------------------------------------------------------
SELECT
    department,
    ROUND(SUM(actual_revenue),  2)                           AS total_actual,
    ROUND(SUM(budget_revenue),  2)                           AS total_budget,
    ROUND(SUM(actual_revenue - budget_revenue), 2)           AS total_variance,
    ROUND(AVG(gross_margin_pct), 1)                          AS avg_margin_pct,
    ROUND(100.0 * SUM(actual_revenue - budget_revenue)
          / NULLIF(SUM(budget_revenue), 0), 2)               AS budget_attainment_pct
FROM revenue_actuals
GROUP BY department
ORDER BY total_actual DESC;


-- ---------------------------------------------------------------
-- 3. Revenue by region — performance ranking
-- ---------------------------------------------------------------
SELECT
    region,
    SUM(actual_revenue)                                      AS total_actual,
    SUM(budget_revenue)                                      AS total_budget,
    ROUND(100.0 * SUM(actual_revenue)
          / SUM(SUM(actual_revenue)) OVER (), 1)             AS pct_of_total_revenue,
    ROUND(100.0 * SUM(actual_revenue - budget_revenue)
          / NULLIF(SUM(budget_revenue), 0), 2)               AS budget_variance_pct,
    RANK() OVER (ORDER BY SUM(actual_revenue) DESC)          AS revenue_rank
FROM revenue_actuals
GROUP BY region
ORDER BY total_actual DESC;


-- ---------------------------------------------------------------
-- 4. Departments over / under budget (current year)
-- ---------------------------------------------------------------
SELECT
    department,
    period,
    ROUND(SUM(actual_revenue),  2)                           AS actual,
    ROUND(SUM(budget_revenue),  2)                           AS budget,
    ROUND(SUM(actual_revenue - budget_revenue), 2)           AS variance,
    CASE
        WHEN SUM(actual_revenue) > SUM(budget_revenue) * 1.05 THEN 'OVER BUDGET'
        WHEN SUM(actual_revenue) < SUM(budget_revenue) * 0.95 THEN 'UNDER BUDGET'
        ELSE 'ON TRACK'
    END                                                      AS budget_status
FROM revenue_actuals
WHERE year = YEAR(CURRENT_DATE)
GROUP BY department, period
ORDER BY period, department;


-- ---------------------------------------------------------------
-- 5. Rolling 12-month revenue trend per department
-- ---------------------------------------------------------------
SELECT
    period,
    department,
    SUM(actual_revenue)                                      AS monthly_revenue,
    ROUND(AVG(SUM(actual_revenue))
          OVER (PARTITION BY department
                ORDER BY period
                ROWS BETWEEN 11 PRECEDING AND CURRENT ROW), 2)
                                                             AS rolling_12m_avg,
    ROUND(AVG(SUM(actual_revenue))
          OVER (PARTITION BY department
                ORDER BY period
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
                                                             AS rolling_3m_avg
FROM revenue_actuals
GROUP BY period, department
ORDER BY department, period;


-- ---------------------------------------------------------------
-- 6. Forecast vs actuals comparison (where periods overlap)
-- ---------------------------------------------------------------
SELECT
    a.period,
    a.department,
    a.region,
    ROUND(SUM(a.actual_revenue), 2)                         AS actual,
    ROUND(AVG(f.forecast), 2)                               AS forecast,
    ROUND(SUM(a.actual_revenue) - AVG(f.forecast), 2)       AS forecast_error,
    ROUND(100.0 * (SUM(a.actual_revenue) - AVG(f.forecast))
          / NULLIF(AVG(f.forecast), 0), 2)                  AS forecast_error_pct
FROM revenue_actuals a
JOIN revenue_forecasts f
  ON a.period    = f.forecast_period
 AND a.department = f.department
 AND a.region     = f.region
GROUP BY a.period, a.department, a.region
ORDER BY ABS(SUM(a.actual_revenue) - AVG(f.forecast)) DESC;


-- ---------------------------------------------------------------
-- 7. Revenue anomalies — periods with Z-score > 2.5
-- ---------------------------------------------------------------
WITH stats AS (
    SELECT
        department,
        AVG(actual_revenue)    AS mean_rev,
        STDDEV(actual_revenue) AS std_rev
    FROM revenue_actuals
    GROUP BY department
)
SELECT
    r.period,
    r.department,
    r.region,
    ROUND(r.actual_revenue, 2)                              AS actual_revenue,
    ROUND((r.actual_revenue - s.mean_rev) / NULLIF(s.std_rev, 0), 3)
                                                            AS z_score,
    CASE
        WHEN ABS((r.actual_revenue - s.mean_rev) / NULLIF(s.std_rev, 0)) > 2.5
        THEN 'ANOMALY' ELSE 'NORMAL'
    END                                                     AS anomaly_flag
FROM revenue_actuals r
JOIN stats s ON r.department = s.department
ORDER BY ABS((r.actual_revenue - s.mean_rev) / NULLIF(s.std_rev, 0)) DESC
LIMIT 20;
