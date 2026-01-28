# 🥗 """Obesity Prediction API
=====================

DESCRIPCIÓN DEL PROBLEMA
-----------------------

La obesidad es un problema de salud pública que puede derivar en múltiples enfermedades crónicas.
El objetivo de este proyecto es predecir el nivel de obesidad de una persona utilizando un modelo
de Machine Learning entrenado a partir de variables demográficas y antropométricas básicas:

- Género
- Edad
- Altura
- Peso

El modelo se entrena de forma offline y luego se expone mediante una API REST desarrollada con
FastAPI, permitiendo realizar predicciones a través de solicitudes HTTP.


ESTRUCTURA DEL PROYECTO
----------------------

El proyecto está compuesto por los siguientes archivos principales:

- train.py  
  Script encargado del entrenamiento offline del modelo de Machine Learning.

- main.py  
  Archivo principal que expone el modelo entrenado mediante una API REST usando FastAPI.

- model.pkl  
  Modelo de Machine Learning previamente entrenado y serializado.

- encoders.pkl  
  Encoders utilizados para transformar variables categóricas durante la predicción.

- requirements.txt  
  Archivo que contiene las dependencias necesarias para ejecutar el proyecto.

- readme.py  
  Archivo de documentación del proyecto.


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
    cd obesity-ml-cloud-api

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

La API fue desplegada en Render, una plataforma cloud que permite el despliegue continuo de
aplicaciones web directamente desde repositorios de GitHub.

El servicio fue configurado para ejecutar una aplicación FastAPI, permitiendo que cada cambio
en el repositorio genere automáticamente un nuevo deploy de la API, accesible mediante una URL
pública.
"""
