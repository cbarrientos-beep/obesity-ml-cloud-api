from fastapi.testclient import TestClient

from app.main import app


# ======================================================
# 1. Crear cliente de prueba
# ======================================================

client = TestClient(app)


# ======================================================
# 2. Payload válido de ejemplo
# ======================================================

VALID_PAYLOAD = {
    "Gender": "Male",
    "Age": 25,
    "Height": 1.75,
    "Weight": 82,
    "family_history": "yes",
    "FAVC": "yes",
    "FCVC": 2,
    "NCP": 3,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2,
    "SCC": "no",
    "FAF": 1,
    "TUE": 1,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation",
}


# ======================================================
# 3. Test endpoint raíz
# ======================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["message"] == "Obesity Prediction API"


# ======================================================
# 4. Test health
# ======================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    assert data["model_loaded"] is True


# ======================================================
# 5. Test schema del modelo
# ======================================================

def test_model_schema():

    response = client.get("/model/schema")

    assert response.status_code == 200

    data = response.json()

    assert "features" in data

    assert "accuracy" in data

    assert data["target"] == "Obesity"


# ======================================================
# 6. Test predicción válida
# ======================================================

def test_predict_valid():

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert isinstance(
        data["prediction"],
        str,
    )


# ======================================================
# 7. Test campo faltante
# ======================================================

def test_predict_missing_field():

    invalid_payload = VALID_PAYLOAD.copy()

    invalid_payload.pop("Weight")

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422


# ======================================================
# 8. Test valor categórico inválido
# ======================================================

def test_predict_invalid_gender():

    invalid_payload = VALID_PAYLOAD.copy()

    invalid_payload["Gender"] = "InvalidValue"

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 400


# ======================================================
# 9. Test batch válido
# ======================================================

def test_predict_batch():

    response = client.post(
        "/predict/batch",
        json=[
            VALID_PAYLOAD,
            VALID_PAYLOAD,
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data

    assert len(
        data["predictions"]
    ) == 2


# ======================================================
# 10. Test batch con entrada inválida
# ======================================================

def test_predict_batch_invalid():

    invalid_payload = VALID_PAYLOAD.copy()

    invalid_payload["Gender"] = "InvalidValue"

    response = client.post(
        "/predict/batch",
        json=[
            VALID_PAYLOAD,
            invalid_payload,
        ],
    )

    assert response.status_code == 400


# ======================================================
# 11. Test tipo incorrecto
# ======================================================

def test_predict_wrong_type():

    invalid_payload = VALID_PAYLOAD.copy()

    invalid_payload["Age"] = "texto"

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422