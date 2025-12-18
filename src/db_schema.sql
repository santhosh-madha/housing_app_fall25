PRAGMA foreign_keys = OFF;

-- Drop in correct order (fact first, then dims)
DROP TABLE IF EXISTS fact_marketing;

DROP TABLE IF EXISTS dim_job;
DROP TABLE IF EXISTS dim_marital;
DROP TABLE IF EXISTS dim_education;
DROP TABLE IF EXISTS dim_default;
DROP TABLE IF EXISTS dim_housing;
DROP TABLE IF EXISTS dim_loan;
DROP TABLE IF EXISTS dim_contact;
DROP TABLE IF EXISTS dim_day_of_week;
DROP TABLE IF EXISTS dim_month;
DROP TABLE IF EXISTS dim_poutcome;

PRAGMA foreign_keys = ON;

-- Dimension tables
CREATE TABLE dim_job (
  job_id INTEGER PRIMARY KEY,
  job TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_marital (
  marital_id INTEGER PRIMARY KEY,
  marital TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_education (
  education_id INTEGER PRIMARY KEY,
  education TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_default (
  default_id INTEGER PRIMARY KEY,
  default_flag TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_housing (
  housing_id INTEGER PRIMARY KEY,
  housing_flag TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_loan (
  loan_id INTEGER PRIMARY KEY,
  loan_flag TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_contact (
  contact_id INTEGER PRIMARY KEY,
  contact TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_day_of_week (
  day_of_week_id INTEGER PRIMARY KEY,
  day_of_week TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_month (
  month_id INTEGER PRIMARY KEY,
  month TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_poutcome (
  poutcome_id INTEGER PRIMARY KEY,
  poutcome TEXT UNIQUE NOT NULL
);

-- Fact table (bank-additional.csv numeric columns)
CREATE TABLE fact_marketing (
  record_id INTEGER PRIMARY KEY AUTOINCREMENT,

  age INTEGER NOT NULL,
  duration INTEGER NOT NULL,
  campaign INTEGER NOT NULL,
  pdays INTEGER NOT NULL,
  previous INTEGER NOT NULL,

  emp_var_rate REAL NOT NULL,
  cons_price_idx REAL NOT NULL,
  cons_conf_idx REAL NOT NULL,
  euribor3m REAL NOT NULL,
  nr_employed REAL NOT NULL,

  job_id INTEGER NOT NULL,
  marital_id INTEGER NOT NULL,
  education_id INTEGER NOT NULL,
  default_id INTEGER NOT NULL,
  housing_id INTEGER NOT NULL,
  loan_id INTEGER NOT NULL,
  contact_id INTEGER NOT NULL,
  day_of_week_id INTEGER NOT NULL,
  month_id INTEGER NOT NULL,
  poutcome_id INTEGER NOT NULL,

  y INTEGER NOT NULL, -- 1=yes, 0=no

  FOREIGN KEY(job_id) REFERENCES dim_job(job_id),
  FOREIGN KEY(marital_id) REFERENCES dim_marital(marital_id),
  FOREIGN KEY(education_id) REFERENCES dim_education(education_id),
  FOREIGN KEY(default_id) REFERENCES dim_default(default_id),
  FOREIGN KEY(housing_id) REFERENCES dim_housing(housing_id),
  FOREIGN KEY(loan_id) REFERENCES dim_loan(loan_id),
  FOREIGN KEY(contact_id) REFERENCES dim_contact(contact_id),
  FOREIGN KEY(day_of_week_id) REFERENCES dim_day_of_week(day_of_week_id),
  FOREIGN KEY(month_id) REFERENCES dim_month(month_id),
  FOREIGN KEY(poutcome_id) REFERENCES dim_poutcome(poutcome_id)
);
