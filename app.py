from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from src.logger import logging

MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"
FEATURES_PATH = "artifacts/features.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

app = FastAPI(title="Credit Card Fraud Detection API")

class Transaction(BaseModel):
    data: dict   # { "V1": value, "V2": value, ..., "Amount": value }

@app.get("/")
def home():
    return {"status": "API is running"}

##prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        df = pd.DataFrame([transaction.data])
        X = preprocessor.transform(df)

        prob = model.predict_proba(X)[:, 1][0]

        threshold = 0.7
        prediction = int(prob >= threshold)

        logging.info(f"Prediction made: {prediction[0]}")
        return {    
            "fraud_probability": round(float(prob), 4),
            "is_fraud": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))