import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================================================
# 0. Definir rutas del proyecto
# ======================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "Obesity_prediction.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
ENCODER_PATH = ROOT / "models" / "encoders.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"


# ======================================================
# 1. Cargar dataset
# ======================================================

df = pd.read_csv(DATA_PATH)


# ======================================================
# 2. Definir variable objetivo
# ======================================================

target = "Obesity"


# ======================================================
# 3. Definir variables predictoras
# ======================================================

features = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]


# ======================================================
# 4. Seleccionar columnas necesarias
# ======================================================

df = df[features + [target]].copy()


# ======================================================
# 5. Identificar variables categóricas
# ======================================================

categorical_features = [
    col
    for col in features
    if df[col].dtype == "object"
]


# ======================================================
# 6. Codificar variables categóricas
# ======================================================

encoders = {}

for col in categorical_features:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = encoder


# ======================================================
# 7. Separar variables X e y
# ======================================================

X = df[features]

y = df[target]


# ======================================================
# 8. Train / Test split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ======================================================
# 9. Entrenar modelo
# ======================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)


# ======================================================
# 10. Evaluar modelo
# ======================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print(
    f"Accuracy del modelo: {accuracy:.4f}"
)


# ======================================================
# 11. Guardar modelo entrenado
# ======================================================

with open(MODEL_PATH, "wb") as f:

    pickle.dump(
        model,
        f,
    )


# ======================================================
# 12. Guardar encoders
# ======================================================

with open(ENCODER_PATH, "wb") as f:

    pickle.dump(
        encoders,
        f,
    )


# ======================================================
# 13. Crear metadata del modelo
# ======================================================

metadata = {

    "model_type": "RandomForestClassifier",

    "target": target,

    "features": features,

    "categorical_features": categorical_features,

    "accuracy": float(accuracy),

    "random_state": 42,

    "test_size_ratio": 0.20,

    "train_samples": len(X_train),

    "test_samples": len(X_test),

    "total_samples": len(df),

    "n_estimators": 100,
}


# ======================================================
# 14. Guardar metadata
# ======================================================

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ======================================================
# 15. Confirmar artefactos generados
# ======================================================

print(
    "Modelo, encoders y metadata guardados correctamente."
)

print(
    f"Modelo: {MODEL_PATH}"
)

print(
    f"Encoders: {ENCODER_PATH}"
)

print(
    f"Metadata: {METADATA_PATH}"
)