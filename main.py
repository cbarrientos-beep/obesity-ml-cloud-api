# main.py
from fastapi import FastAPI
import pickle
import pandas as pd
import os

app = FastAPI(
    title="Obesity Prediction API",
    description="API para predecir nivel de obesidad usando Machine Learning",
    version="1.0"
)

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar modelo
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# Cargar encoders
with open(os.path.join(BASE_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "API de predicción de obesidad funcionando"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Aplicar encoders
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    prediction = model.predict(df)[0]

    return {"prediction": prediction}



