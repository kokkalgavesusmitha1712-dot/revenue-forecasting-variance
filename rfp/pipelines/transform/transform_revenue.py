"""
transform_revenue.py
Cleans, validates, and feature-engineers revenue data
in preparation for forecasting models and variance analysis.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED = ["period", "department", "region", "actual_revenue"]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    logger.info("Schema validation passed")


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    money = ["actual_revenue", "budget_revenue", "prior_year_revenue",
             "cost", "gross_margin", "variance_vs_budget", "variance_vs_prior_year"]
    for col in money:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m")
    df["year"]   = df["period"].dt.year
    df["month"]  = df["period"].dt.month
    logger.info("Type casting complete")
    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["actual_revenue"]  = df["actual_revenue"].fillna(0)
    df["budget_revenue"]  = df["budget_revenue"].fillna(df["actual_revenue"])
    df["cost"]            = df["cost"].fillna(df["actual_revenue"] * 0.65)
    null_count = df[REQUIRED].isnull().sum().sum()
    if null_count:
        logger.warning(f"{null_count} nulls remain in required columns")
    return df


def add_variance_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute variance vs budget and prior year."""
    df = df.copy()
    if "budget_revenue" in df.columns:
        df["variance_vs_budget"]  = df["actual_revenue"] - df["budget_revenue"]
        df["variance_pct_budget"] = np.where(
            df["budget_revenue"] != 0,
            (df["variance_vs_budget"] / df["budget_revenue"] * 100).round(2),
            0
        )
        df["budget_flag"] = df["variance_pct_budget"].apply(
            lambda x: "over_budget" if x > 5 else ("under_budget" if x < -5 else "on_track")
        )
    if "prior_year_revenue" in df.columns:
        df["variance_vs_prior_year"]  = df["actual_revenue"] - df["prior_year_revenue"]
        df["yoy_growth_pct"] = np.where(
            df["prior_year_revenue"] != 0,
            (df["variance_vs_prior_year"] / df["prior_year_revenue"] * 100).round(2),
            0
        )
    return df


def add_margin_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gross margin and margin %."""
    df = df.copy()
    if "cost" in df.columns:
        df["gross_margin"]     = df["actual_revenue"] - df["cost"]
        df["gross_margin_pct"] = np.where(
            df["actual_revenue"] != 0,
            (df["gross_margin"] / df["actual_revenue"] * 100).round(2),
            0
        )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for ML models."""
    df = df.copy()
    df["quarter"]       = df["period"].dt.quarter
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["is_q4"]         = (df["quarter"] == 4).astype(int)
    df["period_index"]  = (
        (df["period"].dt.year - df["period"].dt.year.min()) * 12
        + df["period"].dt.month
    )
    logger.info("Time features added")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages per department+region for trend features."""
    df = df.sort_values(["department", "region", "period"]).copy()
    grp = df.groupby(["department", "region"])["actual_revenue"]
    df["rolling_3m_avg"]  = grp.transform(lambda x: x.rolling(3,  min_periods=1).mean()).round(2)
    df["rolling_12m_avg"] = grp.transform(lambda x: x.rolling(12, min_periods=1).mean()).round(2)
    df["lag_1m"]          = grp.transform(lambda x: x.shift(1))
    df["lag_12m"]         = grp.transform(lambda x: x.shift(12))
    logger.info("Rolling features added")
    return df


def transform_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Master transform — runs full pipeline."""
    logger.info(f"Starting transform — input: {len(df):,} rows")
    validate_schema(df)
    df = cast_types(df)
    df = handle_nulls(df)
    df = add_variance_fields(df)
    df = add_margin_fields(df)
    df = add_time_features(df)
    df = add_rolling_features(df)
    logger.info(f"Transform complete — output: {len(df):,} rows, {len(df.columns)} cols")
    return df


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from pipelines.extract.extract_revenue import extract_from_csv
    df  = extract_from_csv()
    out = transform_revenue(df)
    print(out.head())
    print(out.columns.tolist())
