from pydantic import BaseModel, Field

# ======================================================
# 1. Esquema de entrada para predicción
# ======================================================

class PredictionRequest(BaseModel):

    Gender: str = Field(..., example="Male")

    Age: float = Field(..., example=25)

    Height: float = Field(..., example=1.75)

    Weight: float = Field(..., example=82)

    family_history: str = Field(..., example="yes")

    FAVC: str = Field(..., example="yes")

    FCVC: float = Field(..., example=2)

    NCP: float = Field(..., example=3)

    CAEC: str = Field(..., example="Sometimes")

    SMOKE: str = Field(..., example="no")

    CH2O: float = Field(..., example=2)

    SCC: str = Field(..., example="no")

    FAF: float = Field(..., example=1)

    TUE: float = Field(..., example=1)

    CALC: str = Field(..., example="Sometimes")

    MTRANS: str = Field(..., example="Public_Transportation")


# ======================================================
# 2. Esquema de salida
# ======================================================

class PredictionResponse(BaseModel):

    prediction: str


# ======================================================
# 3. Estado de salud de la API
# ======================================================

class HealthResponse(BaseModel):

    status: str

    model_loaded: bool