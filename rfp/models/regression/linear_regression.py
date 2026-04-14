"""
linear_regression.py
Trains a Linear Regression model on historical revenue data
and forecasts the next N months per department + region.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = ["period_index", "month_sin", "month_cos", "is_q4",
            "rolling_3m_avg", "rolling_12m_avg", "lag_1m", "lag_12m"]
TARGET   = "actual_revenue"


def train_linear_model(df: pd.DataFrame, group_col: list = None):
    """
    Train one Linear Regression model per department+region group.
    Returns a dict of {group_key: (model, scaler, metrics)}.
    """
    if group_col is None:
        group_col = ["department", "region"]

    models  = {}
    metrics = []

    for keys, grp in df.groupby(group_col):
        grp = grp.dropna(subset=FEATURES + [TARGET]).sort_values("period")
        if len(grp) < 12:
            logger.warning(f"Skipping {keys} — insufficient data ({len(grp)} rows)")
            continue

        X = grp[FEATURES].values
        y = grp[TARGET].values

        split    = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model   = LinearRegression()
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        mae     = round(mean_absolute_error(y_test, y_pred), 2)
        rmse    = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        r2      = round(r2_score(y_test, y_pred), 4)

        group_key = keys if isinstance(keys, tuple) else (keys,)
        models[group_key] = (model, scaler)
        metrics.append({**dict(zip(group_col, group_key)),
                        "model": "linear_regression",
                        "mae": mae, "rmse": rmse, "r2": r2,
                        "train_rows": split, "test_rows": len(X_test)})

        logger.info(f"{keys} | MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2}")

    return models, pd.DataFrame(metrics)


def forecast_linear(df: pd.DataFrame, models: dict,
                    n_months: int = 12,
                    group_col: list = None) -> pd.DataFrame:
    """
    Generate N-month forecasts for every group using trained models.
    Returns a long-format DataFrame of forecasts with confidence intervals.
    """
    if group_col is None:
        group_col = ["department", "region"]

    forecasts = []

    for keys, (model, scaler) in models.items():
        grp = df[np.logical_and.reduce(
            [df[col] == val for col, val in zip(group_col, keys)]
        )].sort_values("period")

        last_period_idx = int(grp["period_index"].max())
        last_period     = grp["period"].max()

        for i in range(1, n_months + 1):
            future_period = last_period + pd.DateOffset(months=i)
            pidx          = last_period_idx + i
            month         = future_period.month

            feat = np.array([[
                pidx,
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                int(future_period.quarter == 4),
                float(grp["rolling_3m_avg"].iloc[-1]),
                float(grp["rolling_12m_avg"].iloc[-1]),
                float(grp["actual_revenue"].iloc[-1]),
                float(grp[grp["month"] == month]["actual_revenue"].mean()
                      if len(grp[grp["month"] == month]) else grp["actual_revenue"].mean()),
            ]])

            feat_scaled = scaler.transform(feat)
            pred        = model.predict(feat_scaled)[0]
            std         = grp["actual_revenue"].std()

            row = {**dict(zip(group_col, keys)),
                   "forecast_period": future_period.strftime("%Y-%m"),
                   "model":           "linear_regression",
                   "forecast":        round(pred, 2),
                   "lower_bound":     round(pred - 1.96 * std, 2),
                   "upper_bound":     round(pred + 1.96 * std, 2)}
            forecasts.append(row)

    out = pd.DataFrame(forecasts)
    logger.info(f"Generated {len(out):,} linear regression forecast rows")
    return out


def save_models(models: dict, output_dir: str = "models/regression") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for keys, (model, scaler) in models.items():
        fname = "_".join(str(k).replace(" ", "_") for k in keys)
        with open(f"{output_dir}/{fname}_lr.pkl", "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
    logger.info(f"Saved {len(models)} models to {output_dir}/")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from pipelines.extract.extract_revenue import extract_from_csv
    from pipelines.transform.transform_revenue import transform_revenue

    df        = transform_revenue(extract_from_csv())
    models, metrics = train_linear_model(df)
    forecasts = forecast_linear(df, models, n_months=12)

    print("\n--- Model Metrics ---")
    print(metrics.sort_values("r2", ascending=False).head(10))
    print("\n--- Sample Forecasts ---")
    print(forecasts.head(10))
