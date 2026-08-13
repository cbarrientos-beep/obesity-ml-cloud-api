from fastapi import FastAPI, HTTPException

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
)

from app.predictor import (
    predict_single,
    predict_batch,
    metadata,
)


# ======================================================
# 1. Crear aplicación FastAPI
# ======================================================

app = FastAPI(
    title="Obesity Prediction API",
    description="API para predecir el nivel de obesidad utilizando Machine Learning.",
    version="1.0.0",
)


# ======================================================
# 2. Endpoint raíz
# ======================================================

@app.get("/")
def root():

    return {
        "message": "Obesity Prediction API",
        "version": "1.0.0",
    }


# ======================================================
# 3. Endpoint Health
# ======================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return HealthResponse(
        status="ok",
        model_loaded=True,
    )


# ======================================================
# 4. Endpoint Model Schema
# ======================================================

@app.get("/model/schema")
def model_schema():

    return metadata


# ======================================================
# 5. Endpoint Predict
# ======================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    try:

        prediction = predict_single(
            request.model_dump()
        )

        return PredictionResponse(
            prediction=prediction
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ======================================================
# 6. Endpoint Predict Batch
# ======================================================

@app.post("/predict/batch")
def predict_multiple(
    requests: list[PredictionRequest],
):

    try:

        predictions = predict_batch(
            [
                r.model_dump()
                for r in requests
            ]
        )

        return {
            "predictions": predictions
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )