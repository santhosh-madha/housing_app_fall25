import os
import requests
import streamlit as st

st.set_page_config(page_title="Bank Marketing Prediction", layout="centered")

# For docker-compose (local): API_URL can be http://api:8000
# For Render: set API_URL in Render env vars to: https://YOUR-FASTAPI.onrender.com
API_BASE_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"

st.title("📞 Bank Marketing Subscription Prediction")
st.write("Predict whether a client will subscribe to a term deposit (yes/no).")

with st.form("input_form"):
    st.subheader("Client Info")

    age = st.number_input("age", min_value=18, max_value=100, value=30)
    duration = st.number_input("duration (seconds)", min_value=0, max_value=5000, value=200)
    campaign = st.number_input("campaign", min_value=1, max_value=100, value=1)
    pdays = st.number_input("pdays", min_value=-1, max_value=999, value=999)
    previous = st.number_input("previous", min_value=0, max_value=100, value=0)

    emp_var_rate = st.number_input("emp_var_rate", value=1.1)
    cons_price_idx = st.number_input("cons_price_idx", value=93.2)
    cons_conf_idx = st.number_input("cons_conf_idx", value=-36.4)
    euribor3m = st.number_input("euribor3m", value=4.8)
    nr_employed = st.number_input("nr_employed", value=5191.0)

    job = st.selectbox(
        "job",
        ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
         "retired", "self-employed", "services", "student", "technician",
         "unemployed", "unknown"]
    )

    marital = st.selectbox("marital", ["married", "single", "divorced", "unknown"])
    education = st.selectbox(
        "education",
        ["basic.4y", "basic.6y", "basic.9y", "high.school",
         "illiterate", "professional.course", "university.degree", "unknown"]
    )

    default_flag = st.selectbox("default", ["no", "yes", "unknown"])
    housing_flag = st.selectbox("housing", ["no", "yes", "unknown"])
    loan_flag = st.selectbox("loan", ["no", "yes", "unknown"])

    contact = st.selectbox("contact", ["cellular", "telephone"])
    day_of_week = st.selectbox("day_of_week", ["mon", "tue", "wed", "thu", "fri"])
    month = st.selectbox("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    poutcome = st.selectbox("poutcome", ["failure", "nonexistent", "success"])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": int(age),
        "duration": int(duration),
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "emp_var_rate": float(emp_var_rate),
        "cons_price_idx": float(cons_price_idx),
        "cons_conf_idx": float(cons_conf_idx),
        "euribor3m": float(euribor3m),
        "nr_employed": float(nr_employed),
        "job": job,
        "marital": marital,
        "education": education,
        "default_flag": default_flag,
        "housing_flag": housing_flag,
        "loan_flag": loan_flag,
        "contact": contact,
        "day_of_week": day_of_week,
        "month": month,
        "poutcome": poutcome,
    }

    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=15)
        resp.raise_for_status()
        out = resp.json()

        pred = out["prediction"]
        prob = out.get("probability_yes", None)

        st.write(f"Using API: `{API_BASE_URL}`")

        if pred == 1:
            st.success("✅ Prediction: YES (client will subscribe)")
        else:
            st.warning("❌ Prediction: NO (client will not subscribe)")

        if prob is not None:
            st.info(f"Probability(yes): {prob:.3f}")

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.error(f"Tried URL: {PREDICT_URL}")
