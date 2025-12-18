from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Bank Marketing Classification API")

# ---- Model path ----
# In your Dockerfile you likely set WORKDIR /app and copy repo contents into /app.
# So /app/models/global_best_model.pkl is the safest absolute path.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/global_best_model.pkl"))

model = None  # loaded on startup


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        # Don't crash startup silently: keep API up but return error on predict
        model = None
        print(f"❌ Model file not found at: {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")


# ---- Input schema ----
class BankInput(BaseModel):
    age: int
    duration: int
    campaign: int
    pdays: int
    previous: int
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    job: str
    marital: str
    education: str
    default_flag: str
    housing_flag: str
    loan_flag: str
    contact: str
    day_of_week: str
    month: str
    poutcome: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(payload: BankInput):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded. Expected model at {MODEL_PATH}.",
        )

    X = pd.DataFrame([payload.model_dump()])

    try:
        pred = int(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(X)[0][1])
        except Exception:
            proba = None

    return {"prediction": pred, "probability_yes": proba}
