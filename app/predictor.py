from pathlib import Path
import json
import pickle

import pandas as pd


# ======================================================
# 1. Definir rutas de los artefactos
# ======================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "model.pkl"
ENCODER_PATH = ROOT / "models" / "encoders.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"


# ======================================================
# 2. Cargar modelo
# ======================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ======================================================
# 3. Cargar encoders
# ======================================================

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)


# ======================================================
# 4. Cargar metadata
# ======================================================

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ======================================================
# 5. Obtener features esperadas
# ======================================================

FEATURES = metadata["features"]


# ======================================================
# 6. Preparar datos de entrada
# ======================================================

def prepare_input(data: dict) -> pd.DataFrame:

    df = pd.DataFrame([data])

    df = df[FEATURES].copy()

    for col, encoder in encoders.items():

        if col in df.columns:

            try:
                df[col] = encoder.transform(
                    df[col].astype(str)
                )

            except ValueError as exc:

                valid_values = list(
                    encoder.classes_
                )

                raise ValueError(
                    f"Valor inválido para '{col}'. "
                    f"Valores permitidos: {valid_values}"
                ) from exc

    return df


# ======================================================
# 7. Ejecutar predicción
# ======================================================

def predict_single(data: dict) -> str:

    df = prepare_input(data)

    prediction = model.predict(df)[0]

    return str(prediction)


# ======================================================
# 8. Ejecutar predicción batch
# ======================================================

def predict_batch(items: list[dict]) -> list[str]:

    predictions = []

    for item in items:

        prediction = predict_single(item)

        predictions.append(prediction)

    return predictions