"""
variance_analysis.py
Computes detailed variance analysis between:
  - Actual vs Budget
  - Actual vs Prior Year
  - Actual vs Forecast
Flags anomalies and generates an executive-ready variance report.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANOMALY_THRESHOLD_PCT = 10.0  # flag if variance > ±10% of budget


def variance_vs_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of actual vs budget variance by department and period."""
    required = ["period", "department", "actual_revenue", "budget_revenue"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"Missing columns for variance analysis: {required}")

    summary = (
        df.groupby(["period", "department"])
        .agg(
            actual_revenue =("actual_revenue",  "sum"),
            budget_revenue =("budget_revenue",  "sum"),
        )
        .reset_index()
    )
    summary["variance_amt"] = (summary["actual_revenue"] - summary["budget_revenue"]).round(2)
    summary["variance_pct"] = np.where(
        summary["budget_revenue"] != 0,
        (summary["variance_amt"] / summary["budget_revenue"] * 100).round(2),
        0
    )
    summary["flag"] = summary["variance_pct"].apply(
        lambda x: "OVER"  if x >  ANOMALY_THRESHOLD_PCT else
                  "UNDER" if x < -ANOMALY_THRESHOLD_PCT else "OK"
    )
    logger.info(f"Budget variance: {(summary['flag'] == 'OVER').sum()} over, "
                f"{(summary['flag'] == 'UNDER').sum()} under budget")
    return summary.sort_values(["period", "variance_pct"])


def variance_vs_prior_year(df: pd.DataFrame) -> pd.DataFrame:
    """YoY revenue growth by department."""
    if "prior_year_revenue" not in df.columns:
        logger.warning("prior_year_revenue not found — skipping YoY analysis")
        return pd.DataFrame()

    summary = (
        df.groupby(["period", "department"])
        .agg(
            actual_revenue      =("actual_revenue",       "sum"),
            prior_year_revenue  =("prior_year_revenue",   "sum"),
        )
        .reset_index()
    )
    summary["yoy_variance_amt"] = (summary["actual_revenue"] - summary["prior_year_revenue"]).round(2)
    summary["yoy_growth_pct"]   = np.where(
        summary["prior_year_revenue"] != 0,
        (summary["yoy_variance_amt"] / summary["prior_year_revenue"] * 100).round(2),
        0
    )
    return summary.sort_values(["period", "department"])


def variance_vs_forecast(actuals: pd.DataFrame,
                         forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compare actuals to model forecasts where periods overlap."""
    actuals_agg = (
        actuals.groupby(["period", "department", "region"])["actual_revenue"]
        .sum().reset_index()
    )
    actuals_agg["period"] = actuals_agg["period"].astype(str).str[:7]

    merged = actuals_agg.merge(
        forecasts.rename(columns={"forecast_period": "period"}),
        on=["period", "department", "region"],
        how="inner"
    )
    if merged.empty:
        logger.warning("No overlapping periods between actuals and forecasts")
        return pd.DataFrame()

    merged["forecast_error"]     = (merged["actual_revenue"] - merged["forecast"]).round(2)
    merged["forecast_error_pct"] = np.where(
        merged["forecast"] != 0,
        (merged["forecast_error"] / merged["forecast"] * 100).round(2),
        0
    )
    merged["accuracy_flag"] = merged["forecast_error_pct"].abs().apply(
        lambda x: "ACCURATE" if x <= 5 else ("ACCEPTABLE" if x <= 15 else "INACCURATE")
    )
    logger.info(f"Forecast accuracy: "
                f"{(merged['accuracy_flag'] == 'ACCURATE').sum()} accurate, "
                f"{(merged['accuracy_flag'] == 'INACCURATE').sum()} inaccurate")
    return merged


def detect_anomalies(df: pd.DataFrame,
                     z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag revenue anomalies using Z-score per department.
    A Z-score > threshold indicates an unusual revenue period.
    """
    df = df.copy()
    df["z_score"] = (
        df.groupby("department")["actual_revenue"]
        .transform(lambda x: (x - x.mean()) / x.std())
        .round(3)
    )
    df["is_anomaly"] = df["z_score"].abs() > z_threshold
    anomalies = df[df["is_anomaly"]].copy()
    logger.info(f"Detected {len(anomalies)} anomalies (Z > {z_threshold})")
    return anomalies[["period", "department", "region",
                       "actual_revenue", "z_score", "is_anomaly"]]


def executive_variance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-level executive summary: total revenue, budget, variance,
    and YoY growth per department — suitable for a boardroom report.
    """
    agg = df.groupby("department").agg(
        total_actual    =("actual_revenue",       "sum"),
        total_budget    =("budget_revenue",        "sum"),
        total_prior_year=("prior_year_revenue",    "sum"),
        avg_margin_pct  =("gross_margin_pct",      "mean"),
    ).reset_index()

    agg["budget_variance_amt"] = (agg["total_actual"] - agg["total_budget"]).round(2)
    agg["budget_variance_pct"] = (agg["budget_variance_amt"] / agg["total_budget"] * 100).round(2)
    agg["yoy_growth_pct"]      = (
        (agg["total_actual"] - agg["total_prior_year"]) / agg["total_prior_year"] * 100
    ).round(2)

    for col in ["total_actual", "total_budget", "total_prior_year", "budget_variance_amt"]:
        agg[col] = agg[col].round(2)

    return agg.sort_values("total_actual", ascending=False)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from pipelines.extract.extract_revenue import extract_from_csv
    from pipelines.transform.transform_revenue import transform_revenue

    df = transform_revenue(extract_from_csv())

    print("\n--- Budget Variance (sample) ---")
    bv = variance_vs_budget(df)
    print(bv[bv["flag"] != "OK"].head(10).to_string())

    print("\n--- YoY Variance (sample) ---")
    yoy = variance_vs_prior_year(df)
    print(yoy.head(10).to_string())

    print("\n--- Executive Summary ---")
    print(executive_variance_summary(df).to_string())

    print("\n--- Anomalies ---")
    anomalies = detect_anomalies(df)
    print(anomalies.head(10).to_string())
