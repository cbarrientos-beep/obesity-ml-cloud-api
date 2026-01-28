from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(
    title="Obesity Prediction API",
    description="API para predecir nivel de obesidad usando Machine Learning",
    version="1.0"
)

# Cargar modelo
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "API de predicción de obesidad funcionando"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Aplicar encoder a Gender
    if "Gender" in df.columns:
        df["Gender"] = encoders["Gender"].transform(df["Gender"])

    # Asegurar orden correcto de columnas
    df = df[model.feature_names_in_]

    prediction = model.predict(df)[0]

    # IMPORTANTE: devolver string, no int
    return {"prediction": prediction}


