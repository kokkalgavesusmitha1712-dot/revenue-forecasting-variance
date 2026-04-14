"""
generate_sample_data.py
Generates realistic monthly revenue data across departments, regions,
and product lines for testing the forecasting pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

MONTHS        = pd.date_range("2020-01-01", "2024-12-01", freq="MS")
DEPARTMENTS   = ["Inpatient", "Outpatient", "Pharmacy", "Diagnostics", "Behavioral Health"]
REGIONS       = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
PRODUCT_LINES = ["PPO", "HMO", "EPO", "HDHP", "Medicaid"]

base_revenue = {
    "Inpatient":        500_000,
    "Outpatient":       300_000,
    "Pharmacy":         200_000,
    "Diagnostics":      150_000,
    "Behavioral Health": 80_000,
}

rows = []
for month in MONTHS:
    for dept in DEPARTMENTS:
        for region in REGIONS:
            trend      = 1 + 0.003 * ((month.year - 2020) * 12 + month.month)
            seasonality = 1 + 0.07 * np.sin(2 * np.pi * month.month / 12)
            noise       = np.random.normal(1.0, 0.04)
            actual      = round(base_revenue[dept] * trend * seasonality * noise / len(REGIONS), 2)
            budget      = round(base_revenue[dept] * trend / len(REGIONS) * 1.02, 2)
            prior_year  = round(base_revenue[dept] * (trend - 0.036) * seasonality / len(REGIONS), 2)

            rows.append({
                "period":         month.strftime("%Y-%m"),
                "year":           month.year,
                "month":          month.month,
                "department":     dept,
                "region":         region,
                "product_line":   np.random.choice(PRODUCT_LINES),
                "actual_revenue": actual,
                "budget_revenue": budget,
                "prior_year_revenue": prior_year,
                "cost":           round(actual * np.random.uniform(0.60, 0.75), 2),
                "transactions":   int(np.random.randint(200, 1200)),
            })

df = pd.DataFrame(rows)
df["gross_margin"]     = round(df["actual_revenue"] - df["cost"], 2)
df["variance_vs_budget"]     = round(df["actual_revenue"] - df["budget_revenue"], 2)
df["variance_vs_prior_year"] = round(df["actual_revenue"] - df["prior_year_revenue"], 2)
df["variance_pct_budget"]    = round(df["variance_vs_budget"] / df["budget_revenue"] * 100, 2)

out = Path("data/sample/revenue_sample.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Generated {len(df):,} rows → {out}")
print(df.head())
print(f"\nDate range: {df['period'].min()} to {df['period'].max()}")
print(f"Total actual revenue: ${df['actual_revenue'].sum():,.0f}")
