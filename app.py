from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"
FEATURES_PATH = "artifacts/features.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
features = joblib.load(FEATURES_PATH)

app = FastAPI(title="Credit Card Fraud Detection API")


class Transaction(BaseModel):
    data: dict   # { "V1": value, "V2": value, ..., "Amount": value }

##prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):

    df = pd.DataFrame([transaction.data])
    X = preprocessor.transform(df)

    prob = model.predict_proba(X)[:, 1][0]

    threshold = 0.7
    prediction = int(prob >= threshold)

    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": prediction
    }
