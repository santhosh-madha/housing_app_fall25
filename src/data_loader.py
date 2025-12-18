from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "db" / "bank_marketing.sqlite3"


JOIN_QUERY = """
SELECT
    f.record_id,

    -- numeric
    f.age,
    f.duration,
    f.campaign,
    f.pdays,
    f.previous,
    f.emp_var_rate,
    f.cons_price_idx,
    f.cons_conf_idx,
    f.euribor3m,
    f.nr_employed,

    -- categorical (decoded from normalized dims)
    j.job AS job,
    m.marital AS marital,
    e.education AS education,
    d.default_flag AS default_flag,
    h.housing_flag AS housing_flag,
    l.loan_flag AS loan_flag,
    c.contact AS contact,
    dow.day_of_week AS day_of_week,
    mo.month AS month,
    po.poutcome AS poutcome,

    -- target
    f.y
FROM fact_marketing f
JOIN dim_job j ON f.job_id = j.job_id
JOIN dim_marital m ON f.marital_id = m.marital_id
JOIN dim_education e ON f.education_id = e.education_id
JOIN dim_default d ON f.default_id = d.default_id
JOIN dim_housing h ON f.housing_id = h.housing_id
JOIN dim_loan l ON f.loan_id = l.loan_id
JOIN dim_contact c ON f.contact_id = c.contact_id
JOIN dim_day_of_week dow ON f.day_of_week_id = dow.day_of_week_id
JOIN dim_month mo ON f.month_id = mo.month_id
JOIN dim_poutcome po ON f.poutcome_id = po.poutcome_id
;
"""


def load_dataframe(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load full joined dataframe from SQLite."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(JOIN_QUERY, conn)

    return df


def split_X_y(df: pd.DataFrame):
    """Split into X (features) and y (target)."""
    y = df["y"].astype(int)
    X = df.drop(columns=["y", "record_id"])
    return X, y


if __name__ == "__main__":
    df = load_dataframe()
    print("✅ Loaded joined dataframe")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3))
