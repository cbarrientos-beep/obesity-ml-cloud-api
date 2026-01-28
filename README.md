# 🥗 Obesity Prediction API

DESCRIPCIÓN DEL PROBLEMA
-----------------------

La obesidad es un problema de salud pública que puede derivar en múltiples enfermedades crónicas.
El objetivo de este proyecto es predecir el nivel de obesidad de una persona utilizando un modelo
de Machine Learning entrenado a partir de variables demográficas y antropométricas básicas:

- Género
- Edad
- Altura
- Peso

El modelo se entrena de forma offline y luego se expone mediante una API REST desarrollada con FastAPI,
permitiendo realizar predicciones a través de solicitudes HTTP.


ESTRUCTURA DEL PROYECTO
----------------------

├── train.py          # Entrenamiento offline del modelo
├── main.py           # API REST con FastAPI
├── model.pkl         # Modelo entrenado
├── encoders.pkl      # Encoders de variables categóricas
├── requirements.txt  # Dependencias del proyecto
└── readme.py         # Documentación del proyecto


ENTRENAMIENTO DEL MODELO
-----------------------

El entrenamiento del modelo se realiza de forma offline ejecutando el siguiente comando:

    python train.py

Este proceso genera los archivos model.pkl y encoders.pkl, los cuales son utilizados
posteriormente por la API para realizar predicciones.


INSTRUCCIONES PARA CORRER LA API LOCALMENTE
-------------------------------------------

1. Clonar el repositorio

    git clone <url-del-repositorio>
    cd obesity-prediction-api

2. Crear y activar un entorno virtual

    python -m venv venv
    source venv/bin/activate      # macOS / Linux
    venv\\Scripts\\activate       # Windows

3. Instalar dependencias

    pip install -r requirements.txt

4. Levantar la API

    uvicorn main:app --reload

La API quedará disponible en:

    http://127.0.0.1:8000

La documentación interactiva se encuentra disponible en:

    http://127.0.0.1:8000/docs


EJEMPLO DE REQUEST AL ENDPOINT /predict
---------------------------------------

Endpoint:

    POST /predict

Variables de entrada:

- Gender (string): Género de la persona
- Age (int): Edad en años
- Height (float): Altura en metros
- Weight (float): Peso en kilogramos

Ejemplo de request (JSON):

    {
      "Gender": "Male",
      "Age": 25,
      "Height": 1.75,
      "Weight": 85
    }

Ejemplo de respuesta:

    {
      "prediction": "Overweight_Level_I"
    }

El valor retornado en "prediction" corresponde a la clase de obesidad predicha por el modelo.


PLATAFORMA CLOUD USADA PARA EL DEPLOY
-------------------------------------

La API fue desplegada en Google Cloud Platform (GCP) utilizando:

- Google Cloud Run para la ejecución de la aplicación FastAPI
- Docker para la containerización del servicio

Este enfoque permite un despliegue escalable, serverless y accesible mediante una URL pública.
"""
