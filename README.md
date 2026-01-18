🏦 Bank Marketing Subscription Prediction

Project Overview
This project predicts whether a bank customer will subscribe to a term deposit using machine learning. It includes a full ML pipeline, from data engineering and preprocessing to model deployment and a web interface.

🚀 Features

Normalized Database (3NF) – Reduces redundancy and ensures data integrity

SQL + Pandas Integration – Extract data for ML-ready processing

Exploratory Data Analysis (EDA) – Feature distributions, correlation, missing value handling

Preprocessing Pipeline – Standard scaling, one-hot encoding, optional PCA

16 ML Experiments – Logistic Regression, Ridge, Random Forest, XGBoost, LightGBM

Model Selection – F1-score-based, cross-validation, hyperparameter tuning

FastAPI Deployment – Serve predictions via a REST API

Streamlit Web Interface – Interactive web app for real-time prediction

Dockerized – Both API and Streamlit app

Cloud Deployment – Render.com

MLflow Tracking – Experiment logging and reproducibility

📂 Repository Structure
housing_app_fall25/
│
├── api/
│   ├── app.py           # FastAPI backend
│   └── Dockerfile       # API Docker container
│
├── streamlit/
│   ├── app.py           # Streamlit frontend
│   └── Dockerfile       # Streamlit Docker container
│
├── models/
│   ├── global_best_model.pkl   # Best trained model
│   └── other model files
│
├── src/
│   ├── build_sqlite_db.py      # Create 3NF database
│   ├── data_loader.py          # SQL JOIN → Pandas DataFrame
│   ├── preprocess.py           # Feature preprocessing & pipelines
│   └── experiments.py          # ML experiments & logging
│
├── data/
│   ├── raw/                    # Original CSV (ignored in git)
│   └── db/                     # SQLite database
│
├── docker-compose.yml          # Local multi-container setup
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

🗂 Dataset

Source: UCI Bank Marketing Dataset

Target: y → whether the customer subscribed (1 = Yes, 0 = No)

Features:

Numerical: age, duration, campaign, pdays, previous, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed

Categorical: job, marital, education, default, housing, loan, contact, day_of_week, month, poutcome

⚙️ Setup & Installation

Clone the repository:

git clone https://github.com/santhosh-madha/housing_app_fall25.git
cd housing_app_fall25


Create a virtual environment:

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac


Install dependencies:

pip install -r requirements.txt


Build and run containers locally:

docker-compose up --build


FastAPI API available at: http://localhost:10000

Streamlit app at: http://localhost:8501

🧪 Usage
API

Send a POST request to /predict with JSON payload:

{
  "age": 35,
  "job": "technician",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day_of_week": "mon",
  "month": "may",
  "duration": 150,
  "campaign": 2,
  "pdays": 999,
  "previous": 0,
  "emp_var_rate": 1.1,
  "cons_price_idx": 93.994,
  "cons_conf_idx": -36.4,
  "euribor3m": 4.857,
  "nr_employed": 5191
}


Response:

{
  "prediction": 0,
  "probability_yes": 0.021
}

Streamlit App

Launches a web form to input customer data

Click Predict to see subscription probability

📊 ML Pipeline

Train/Test Split – Stratified to preserve class distribution

Preprocessing – StandardScaler + OneHotEncoder (+ optional PCA)

Experiments – 16 ML model experiments with/without PCA and hyperparameter tuning

Evaluation – F1-score with cross-validation and test data

Best Model – LightGBM, tuned, no PCA

☁️ Cloud Deployment

FastAPI API: https://housing-app-fall25-api.onrender.com

Streamlit UI: [Deployed Streamlit URL] (set via environment variable to use API)

📈 Model Tracking

All experiments logged using MLflow on DagsHub

Tracks:

Hyperparameters

CV metrics

Test F1-score

Confusion matrices

Facilitates reproducibility and model selection

📌 Key Learning Outcomes

Full ML lifecycle experience: data engineering → modeling → deployment

Database design and normalization (3NF)

Experiment tracking and visualization

REST API creation and containerization

Cloud deployment and live UI for users
