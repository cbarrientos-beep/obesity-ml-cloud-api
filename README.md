# Obesity Prediction API

Proyecto de Machine Learning desplegado como API REST utilizando FastAPI.
El objetivo es predecir el nivel de obesidad de una persona a partir de variables demográficas y de estilo de vida.

## Estructura del proyecto

- `train.py`: Entrenamiento offline del modelo
- `main.py`: API REST con FastAPI
- `model.pkl`: Modelo entrenado
- `encoders.pkl`: Encoders de variables categóricas
- `requirements.txt`: Dependencias del proyecto

## Entrenamiento del modelo

El entrenamiento se realiza de forma offline ejecutando:

```bash
python train.py
