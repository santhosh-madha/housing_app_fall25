import sqlite3
from pathlib import Path
import pandas as pd

# Project root (repo folder)
ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "data" / "raw" / "bank-additional.csv"
DB_PATH = ROOT / "data" / "db" / "bank_marketing.sqlite3"
SCHEMA_PATH = ROOT / "src" / "db_schema.sql"

# Map: csv_col -> (dimension_table, dimension_value_column)
DIMS = {
    "job": ("dim_job", "job"),
    "marital": ("dim_marital", "marital"),
    "education": ("dim_education", "education"),
    "default": ("dim_default", "default_flag"),
    "housing": ("dim_housing", "housing_flag"),
    "loan": ("dim_loan", "loan_flag"),
    "contact": ("dim_contact", "contact"),
    "day_of_week": ("dim_day_of_week", "day_of_week"),
    "month": ("dim_month", "month"),
    "poutcome": ("dim_poutcome", "poutcome"),
}

# Numeric columns for bank-additional.csv
# NOTE: bank-additional does NOT have "balance"
FACT_NUMERIC_COLS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

# Rename dotted column names to SQL-friendly column names
NUMERIC_RENAME_MAP = {
    "emp.var.rate": "emp_var_rate",
    "cons.price.idx": "cons_price_idx",
    "cons.conf.idx": "cons_conf_idx",
    "nr.employed": "nr_employed",
}


def insert_unique_values(conn: sqlite3.Connection, table: str, col: str, values):
    """Insert unique dimension values into a dimension table."""
    unique_vals = sorted(pd.Series(values).fillna("unknown").astype(str).unique())
    conn.executemany(
        f"INSERT OR IGNORE INTO {table}({col}) VALUES (?)",
        [(v,) for v in unique_vals],
    )


def fetch_dim_map(conn: sqlite3.Connection, table: str, col: str):
    """Return mapping: dim_value -> dim_id for a given dim table."""
    dim_df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    id_col = dim_df.columns[0]  # first column is *_id
    return dict(zip(dim_df[col], dim_df[id_col]))


def main():
    # Validate CSV exists
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    # Ensure DB folder exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV (bank-additional uses ; separator)
    df = pd.read_csv(CSV_PATH, sep=";")

    # Convert target to 0/1
    df["y"] = df["y"].map({"yes": 1, "no": 0}).astype(int)

    # Normalize categorical columns
    for col in DIMS.keys():
        df[col] = df[col].fillna("unknown").astype(str)

    # Build DB
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Reset schema
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema_sql)

        # Fill dimension tables
        for src_col, (dim_table, dim_val_col) in DIMS.items():
            insert_unique_values(conn, dim_table, dim_val_col, df[src_col])

        # Build mapping dicts: value -> id
        dim_maps = {}
        for src_col, (dim_table, dim_val_col) in DIMS.items():
            dim_maps[src_col] = fetch_dim_map(conn, dim_table, dim_val_col)

        # Build fact dataframe
        fact = pd.DataFrame()

        # Numeric columns (some are floats)
        for c in FACT_NUMERIC_COLS:
            fact[c] = pd.to_numeric(df[c], errors="raise")

        # Rename dotted numeric columns to match SQL schema
        fact = fact.rename(columns=NUMERIC_RENAME_MAP)

        # Foreign key columns
        fact["job_id"] = df["job"].map(dim_maps["job"]).astype(int)
        fact["marital_id"] = df["marital"].map(dim_maps["marital"]).astype(int)
        fact["education_id"] = df["education"].map(dim_maps["education"]).astype(int)
        fact["default_id"] = df["default"].map(dim_maps["default"]).astype(int)
        fact["housing_id"] = df["housing"].map(dim_maps["housing"]).astype(int)
        fact["loan_id"] = df["loan"].map(dim_maps["loan"]).astype(int)
        fact["contact_id"] = df["contact"].map(dim_maps["contact"]).astype(int)
        fact["day_of_week_id"] = df["day_of_week"].map(dim_maps["day_of_week"]).astype(int)
        fact["month_id"] = df["month"].map(dim_maps["month"]).astype(int)
        fact["poutcome_id"] = df["poutcome"].map(dim_maps["poutcome"]).astype(int)

        # Target
        fact["y"] = df["y"].astype(int)

        # Insert into fact table
        fact.to_sql("fact_marketing", conn, if_exists="append", index=False)

        # Quick verification counts
        n_fact = conn.execute("SELECT COUNT(*) FROM fact_marketing").fetchone()[0]
        n_jobs = conn.execute("SELECT COUNT(*) FROM dim_job").fetchone()[0]

    print("✅ SQLite DB created successfully!")
    print(f"DB Path: {DB_PATH}")
    print(f"Rows inserted into fact_marketing: {n_fact}")
    print(f"Unique job categories: {n_jobs}")


if __name__ == "__main__":
    main()
