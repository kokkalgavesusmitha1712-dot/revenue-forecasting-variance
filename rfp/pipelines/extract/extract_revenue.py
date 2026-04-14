"""
extract_revenue.py
Extracts historical revenue, budget, and prior-year data from
a SQL / Snowflake data warehouse using SQLAlchemy.
"""

import pandas as pd
import logging
import yaml
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_engine(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    db  = cfg["sources"]["sql_db"]
    url = f"{db['dialect']}://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"
    logger.info(f"Connecting to {db['host']} / {db['database']}")
    return create_engine(url)


def extract_revenue_actuals(years: int = 4,
                            config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Pull actual revenue by period, department, region, and product line."""
    engine = get_engine(config_path)
    query  = text(f"""
        SELECT
            period,
            year,
            month,
            department,
            region,
            product_line,
            actual_revenue,
            cost,
            transactions
        FROM revenue_actuals
        WHERE year >= YEAR(CURRENT_DATE) - {years}
        ORDER BY period, department, region
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    logger.info(f"Extracted {len(df):,} actuals rows")
    return df


def extract_budget(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Pull approved budget figures."""
    engine = get_engine(config_path)
    query  = text("""
        SELECT period, department, region, product_line, budget_revenue
        FROM revenue_budget
        ORDER BY period, department
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    logger.info(f"Extracted {len(df):,} budget rows")
    return df


def extract_from_csv(filepath: str = "data/sample/revenue_sample.csv") -> pd.DataFrame:
    """Load revenue data from a CSV file (for local dev / demo)."""
    df = pd.read_csv(filepath, parse_dates=False)
    logger.info(f"Loaded {len(df):,} rows from {filepath}")
    return df


if __name__ == "__main__":
    df = extract_from_csv()
    print(df.shape)
    print(df.head())
