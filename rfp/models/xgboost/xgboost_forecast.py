"""
xgboost_forecast.py
Trains an XGBoost regression model for revenue forecasting.
XGBoost captures non-linear patterns and interaction effects
that simple regression misses — ideal for multi-dimensional revenue data.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = ["period_index", "month_sin", "month_cos", "is_q4",
            "rolling_3m_avg", "rolling_12m_avg", "lag_1m", "lag_12m"]
TARGET   = "actual_revenue"

XGB_PARAMS = {
    "n_estimators":    200,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "random_state":    42,
    "verbosity":       0,
}


def train_xgb_model(df: pd.DataFrame,
                    group_col: list = None) -> tuple[dict, pd.DataFrame]:
    """Train one XGBoost model per department+region group."""
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

        split           = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred = model.predict(X_test)
        mae    = round(mean_absolute_error(y_test, y_pred), 2)
        rmse   = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        r2     = round(r2_score(y_test, y_pred), 4)

        group_key         = keys if isinstance(keys, tuple) else (keys,)
        models[group_key] = model
        metrics.append({**dict(zip(group_col, group_key)),
                        "model": "xgboost",
                        "mae": mae, "rmse": rmse, "r2": r2,
                        "train_rows": split, "test_rows": len(X_test)})

        logger.info(f"{keys} | MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2}")

    return models, pd.DataFrame(metrics)


def get_feature_importance(models: dict,
                           group_col: list = None) -> pd.DataFrame:
    """Return averaged feature importances across all trained models."""
    if group_col is None:
        group_col = ["department", "region"]

    rows = []
    for keys, model in models.items():
        imp = dict(zip(FEATURES, model.feature_importances_))
        rows.append({**dict(zip(group_col, keys)), **imp})

    imp_df = pd.DataFrame(rows)
    avg    = imp_df[FEATURES].mean().sort_values(ascending=False).reset_index()
    avg.columns = ["feature", "avg_importance"]
    return avg


def forecast_xgb(df: pd.DataFrame, models: dict,
                 n_months: int = 12,
                 group_col: list = None) -> pd.DataFrame:
    """Generate N-month forecasts using trained XGBoost models."""
    if group_col is None:
        group_col = ["department", "region"]

    forecasts = []

    for keys, model in models.items():
        grp = df[np.logical_and.reduce(
            [df[col] == val for col, val in zip(group_col, keys)]
        )].sort_values("period")

        last_idx    = int(grp["period_index"].max())
        last_period = grp["period"].max()
        std         = grp["actual_revenue"].std()

        for i in range(1, n_months + 1):
            fp    = last_period + pd.DateOffset(months=i)
            month = fp.month
            feat  = np.array([[
                last_idx + i,
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                int(fp.quarter == 4),
                float(grp["rolling_3m_avg"].iloc[-1]),
                float(grp["rolling_12m_avg"].iloc[-1]),
                float(grp["actual_revenue"].iloc[-1]),
                float(grp[grp["month"] == month]["actual_revenue"].mean()
                      if len(grp[grp["month"] == month])
                      else grp["actual_revenue"].mean()),
            ]])

            pred = model.predict(feat)[0]
            forecasts.append({
                **dict(zip(group_col, keys)),
                "forecast_period": fp.strftime("%Y-%m"),
                "model":           "xgboost",
                "forecast":        round(pred, 2),
                "lower_bound":     round(pred - 1.96 * std, 2),
                "upper_bound":     round(pred + 1.96 * std, 2),
            })

    out = pd.DataFrame(forecasts)
    logger.info(f"Generated {len(out):,} XGBoost forecast rows")
    return out


def save_xgb_models(models: dict,
                    output_dir: str = "models/xgboost") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for keys, model in models.items():
        fname = "_".join(str(k).replace(" ", "_") for k in keys)
        model.save_model(f"{output_dir}/{fname}_xgb.json")
    logger.info(f"Saved {len(models)} XGBoost models to {output_dir}/")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from pipelines.extract.extract_revenue import extract_from_csv
    from pipelines.transform.transform_revenue import transform_revenue

    df              = transform_revenue(extract_from_csv())
    models, metrics = train_xgb_model(df)
    forecasts       = forecast_xgb(df, models, n_months=12)
    importances     = get_feature_importance(models)

    print("\n--- XGBoost Metrics (top 10 by R²) ---")
    print(metrics.sort_values("r2", ascending=False).head(10).to_string())
    print("\n--- Feature Importances ---")
    print(importances.to_string())
    print("\n--- Sample Forecasts ---")
    print(forecasts.head(10).to_string())
