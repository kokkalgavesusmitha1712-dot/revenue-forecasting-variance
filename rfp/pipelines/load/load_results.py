"""
load_results.py
Exports all pipeline outputs — cleaned data, forecasts, variance reports —
to CSV files and optionally to a SQL warehouse.
"""

import pandas as pd
import logging
from pathlib import Path
from sqlalchemy import create_engine
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Exported {len(df):,} rows → {path}")


def export_all(cleaned: pd.DataFrame,
               forecasts: pd.DataFrame,
               budget_variance: pd.DataFrame,
               yoy_variance: pd.DataFrame,
               exec_summary: pd.DataFrame,
               anomalies: pd.DataFrame,
               output_dir: str = "data/processed") -> None:
    """Export all pipeline outputs to CSV."""
    export_csv(cleaned,          f"{output_dir}/revenue_cleaned.csv")
    export_csv(forecasts,        f"{output_dir}/revenue_forecasts.csv")
    export_csv(budget_variance,  f"{output_dir}/variance_vs_budget.csv")
    export_csv(yoy_variance,     f"{output_dir}/variance_yoy.csv")
    export_csv(exec_summary,     f"{output_dir}/executive_summary.csv")
    export_csv(anomalies,        f"{output_dir}/anomalies.csv")
    logger.info(f"All outputs exported to {output_dir}/")


def load_to_warehouse(df: pd.DataFrame, table: str,
                      config_path: str = "config/config.yaml",
                      if_exists: str = "replace") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    db  = cfg["warehouse"]
    url = f"{db['dialect']}://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"
    engine = create_engine(url)
    df.to_sql(table, engine, if_exists=if_exists, index=False, chunksize=1000)
    logger.info(f"Loaded {len(df):,} rows → {table}")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from pipelines.extract.extract_revenue       import extract_from_csv
    from pipelines.transform.transform_revenue   import transform_revenue
    from pipelines.transform.variance_analysis   import (
        variance_vs_budget, variance_vs_prior_year,
        detect_anomalies, executive_variance_summary
    )
    from models.regression.linear_regression     import train_linear_model, forecast_linear
    from models.xgboost.xgboost_forecast         import train_xgb_model, forecast_xgb

    print("Running full pipeline...")
    df      = transform_revenue(extract_from_csv())
    lr_m, _ = train_linear_model(df)
    xgb_m,_ = train_xgb_model(df)

    lr_fc   = forecast_linear(df, lr_m,  n_months=12)
    xgb_fc  = forecast_xgb(df,   xgb_m, n_months=12)
    all_fc  = pd.concat([lr_fc, xgb_fc], ignore_index=True)

    export_all(
        cleaned         = df,
        forecasts       = all_fc,
        budget_variance = variance_vs_budget(df),
        yoy_variance    = variance_vs_prior_year(df),
        exec_summary    = executive_variance_summary(df),
        anomalies       = detect_anomalies(df),
    )
    print("Pipeline complete — check data/processed/")
