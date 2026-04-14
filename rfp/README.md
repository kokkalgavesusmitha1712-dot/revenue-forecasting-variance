# 📈 Revenue Forecasting & Variance Analysis

> **An end-to-end machine learning pipeline that forecasts monthly revenue using Linear Regression and XGBoost, automates budget variance detection, flags anomalies, and delivers executive-ready reports via Power BI and Tableau dashboards.**

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-4479A1?style=flat&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-EE4C2C?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Snowflake-29B5E8?style=flat&logo=snowflake&logoColor=white" />
  <img src="https://img.shields.io/badge/Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black" />
  <img src="https://img.shields.io/badge/Tableau-E97627?style=flat&logo=tableau&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white" />
</p>

---

## 📌 Overview

Finance and analytics teams spend hours manually pulling revenue data, comparing actuals to budgets, and trying to predict next quarter's numbers. This project automates the entire workflow:

- **Extracts** historical revenue data from Snowflake / SQL warehouses
- **Transforms** raw records into ML-ready features — rolling averages, lag features, seasonality encodings
- **Forecasts** 12 months of revenue per department and region using two complementary models
- **Detects** budget overruns, underruns, and anomalies automatically
- **Generates** an executive variance report highlighting every department that is off-track
- **Visualises** everything in Power BI and Tableau dashboards

---

## 🏗️ Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                            │
│        Snowflake / SQL DB  │  CSV / Excel flat files         │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
      ┌───────────────────────────┐
      │       EXTRACT LAYER       │
      │  extract_revenue.py       │
      │  · Revenue actuals        │
      │  · Budget targets         │
      │  · Prior year comparisons │
      └──────────────┬────────────┘
                     │
                     ▼
      ┌───────────────────────────┐
      │     TRANSFORM LAYER       │
      │  transform_revenue.py     │
      │  · Type casting           │
      │  · Rolling avg features   │
      │  · Lag features (1m, 12m) │
      │  · Seasonality encoding   │
      │  · Margin calculations    │
      │                           │
      │  variance_analysis.py     │
      │  · Actual vs Budget       │
      │  · Actual vs Prior Year   │
      │  · Anomaly detection      │
      └──────────────┬────────────┘
                     │
                     ▼
      ┌───────────────────────────┐
      │       MODEL LAYER         │
      │  linear_regression.py     │
      │  · Trend + seasonality    │
      │                           │
      │  xgboost_forecast.py      │
      │  · Non-linear patterns    │
      │  · Feature importance     │
      └──────────────┬────────────┘
                     │
                     ▼
      ┌───────────────────────────┐
      │       LOAD LAYER          │
      │  load_results.py          │
      │  · Warehouse tables       │
      │  · CSV exports            │
      └──────────────┬────────────┘
                     │
                     ▼
      ┌───────────────────────────┐
      │         OUTPUTS           │
      │  Power BI Dashboard       │
      │  Tableau Dashboard        │
      │  Executive Variance Report│
      │  Forecast CSVs            │
      └───────────────────────────┘
```

---

## ✨ Key Features

- **Dual forecasting models** — Linear Regression for interpretability, XGBoost for accuracy; compare both side-by-side
- **Rich feature engineering** — rolling 3-month and 12-month averages, 1-month and 12-month lag features, sine/cosine seasonality encoding
- **Variance engine** — automatically computes actual vs budget, actual vs prior year, and actual vs forecast variances with % flags
- **Anomaly detection** — Z-score based statistical flagging of unusual revenue periods per department
- **Executive summary** — one-table board-level view of every department's performance vs budget and prior year
- **Confidence intervals** — every forecast includes upper and lower bounds at 95% confidence
- **Modular & testable** — each component can be run independently or as part of the full pipeline
- **Snowflake-ready** — native Snowflake connector via SQLAlchemy

---

## 🗂️ Project Structure

```
revenue-forecasting-variance/
│
├── 📁 pipelines/
│   ├── extract/
│   │   └── extract_revenue.py        # SQL / CSV extraction
│   ├── transform/
│   │   ├── transform_revenue.py      # Cleaning + feature engineering
│   │   └── variance_analysis.py      # Variance + anomaly detection engine
│   └── load/
│       └── load_results.py           # Export to warehouse + CSV
│
├── 📁 models/
│   ├── regression/
│   │   └── linear_regression.py      # Train + forecast with Linear Regression
│   ├── xgboost/
│   │   └── xgboost_forecast.py       # Train + forecast with XGBoost
│   └── arima/                        # (extend with ARIMA model here)
│
├── 📁 sql/
│   ├── queries/
│   │   └── revenue_kpi_queries.sql   # 7 core KPI SQL queries
│   └── reports/
│       └── monthly_report.sql        # Monthly executive report
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb     # Revenue data profiling & EDA
│   ├── 02_model_comparison.ipynb     # LR vs XGBoost comparison
│   └── 03_variance_analysis.ipynb    # Variance deep-dive & visualisation
│
├── 📁 dashboards/
│   ├── powerbi/                      # Power BI (.pbix) files
│   └── tableau/                      # Tableau (.twbx) files
│
├── 📁 data/
│   ├── raw/                          # Source data (gitignored)
│   ├── processed/                    # Pipeline outputs (gitignored)
│   └── sample/
│       ├── revenue_sample.csv        # 15,000-row sample dataset
│       └── generate_sample_data.py   # Script to regenerate sample data
│
├── 📁 reports/                       # Auto-generated variance reports
├── 📁 config/
│   └── config.yaml                   # Connection config template
├── 📁 tests/                         # Unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Outputs Produced

| Output | Description |
|--------|-------------|
| `revenue_forecasts.csv` | 12-month forecast per department × region, both models, with confidence intervals |
| `variance_vs_budget.csv` | Monthly actual vs budget variance with OVER / UNDER / OK flag |
| `variance_yoy.csv` | YoY growth % per department and period |
| `executive_summary.csv` | Board-level one-pager: total revenue, budget attainment, margin, YoY growth |
| `anomalies.csv` | Flagged periods with Z-score > 2.5 per department |
| `model_metrics.csv` | MAE, RMSE, R² per model per department+region group |

---

## 📐 Models Used

| Model | Strengths | Best For |
|-------|-----------|----------|
| Linear Regression | Interpretable, fast, stable | Trend + seasonality, stakeholder explainability |
| XGBoost | Captures non-linearity and interactions | Higher accuracy, multi-dimensional data |

Both models produce 12-month forecasts with 95% confidence intervals. Compare both outputs in the `notebooks/02_model_comparison.ipynb` notebook.

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/sushmitha-kokkalgave/revenue-forecasting-variance.git
cd revenue-forecasting-variance
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure connections

```bash
cp config/config.yaml config/config_local.yaml
# Edit config_local.yaml with your Snowflake / DB credentials
```

### 4. Generate sample data

```bash
python data/sample/generate_sample_data.py
```

### 5. Run the full pipeline

```bash
# Generate features
python pipelines/transform/transform_revenue.py

# Train models and forecast
python models/regression/linear_regression.py
python models/xgboost/xgboost_forecast.py

# Export all results
python pipelines/load/load_results.py
```

### 6. Explore results in Jupyter

```bash
jupyter notebook notebooks/03_variance_analysis.ipynb
```

---

## 📈 Sample Results

On the included 5-year sample dataset (5 departments × 5 regions × 60 months):

| Metric | Value |
|--------|-------|
| XGBoost avg R² | ~0.91 |
| Linear Regression avg R² | ~0.84 |
| Departments flagged over budget | Varies by threshold |
| Anomalies detected | ~3–5% of periods |
| Forecast horizon | 12 months |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.8+, SQL |
| ML Models | Scikit-learn, XGBoost |
| Data Warehouse | Snowflake, PostgreSQL |
| Libraries | Pandas, NumPy, SQLAlchemy, Statsmodels |
| Visualisation | Power BI (DAX), Tableau, Plotly, Seaborn |
| Config | YAML, python-dotenv |
| Testing | Pytest |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Extend the `models/arima/` folder to add an ARIMA baseline model.

---

## 👩‍💻 Author

**Sushmitha Kokkalgave** — Senior Data Analyst  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/sushmitha-kokkalgave)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:susmitha.data97@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/sushmitha-kokkalgave)

---

<p align="center"><i>Turning historical revenue data into forward-looking decisions — one forecast at a time.</i></p>
