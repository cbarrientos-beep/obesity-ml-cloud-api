# 🥗 Obesity Prediction API


Proyecto de Machine Learning desplegado como una API REST utilizando FastAPI.
El objetivo es predecir el nivel de obesidad de una persona a partir de
variables demográficas básicas.

--------------------------------------------------
Descripción del problema y del modelo
--------------------------------------------------

La obesidad es un problema de salud pública que puede ser analizado a partir
de características demográficas y físicas de una persona. En este proyecto,
se desarrolló un modelo de Machine Learning capaz de predecir el nivel de
obesidad utilizando las siguientes variables de entrada:

- Género (Gender)
- Edad (Age)
- Altura en metros (Height)
- Peso en kilogramos (Weight)

Estas variables son procesadas por el modelo para generar una clasificación
del nivel de obesidad del individuo.

El modelo es expuesto mediante una API REST desarrollada con FastAPI, lo que
permite consumir la predicción desde aplicaciones externas o mediante
documentación interactiva.

--------------------------------------------------
Estructura del proyecto
--------------------------------------------------

- train.py  
  Entrenamiento offline del modelo de Machine Learning.

- main.py  
  Implementación de la API REST utilizando FastAPI.

- model.pkl  
  Modelo de Machine Learning entrenado.

- encoders.pkl  
  Encoders utilizados para transformar variables categóricas.

- requirements.txt  
  Dependencias necesarias para ejecutar el proyecto.

- readme.py  
  Archivo de documentación del proyecto.

--------------------------------------------------
Instrucciones para correr la API localmente
--------------------------------------------------

1. Clonar el repositorio desde GitHub.
2. Crear y activar un entorno virtual.
3. Instalar las dependencias del proyecto usando requirements.txt.
4. Ejecutar la aplicación con FastAPI utilizando uvicorn.

La API quedará disponible por defecto en:
http://127.0.0.1:8000

--------------------------------------------------
API desplegada en la nube (Render)
--------------------------------------------------

La API se encuentra desplegada en la plataforma cloud Render y es accesible
públicamente.

URL base de la API:
https://obesity-ml-cloud-api.onrender.com

Documentación interactiva (Swagger UI):
https://obesity-ml-cloud-api.onrender.com/docs#/default/predict_predict_post

Desde esta documentación es posible visualizar los endpoints disponibles
y probar el modelo directamente desde el navegador.

Nota:
Al utilizar el plan gratuito de Render, la primera solicitud puede tardar
algunos segundos debido al cold start del servicio.

--------------------------------------------------
Cómo probar el endpoint /predict desde Swagger
--------------------------------------------------

Para probar la API desde el navegador, seguir los siguientes pasos:

1. Acceder a la documentación Swagger:
   https://obesity-ml-cloud-api.onrender.com/docs

2. Ubicar el endpoint POST /predict.

3. Presionar el botón "Try it out".

4. Ingresar un JSON con las variables de entrada requeridas por el modelo.

5. Presionar el botón "Execute".

6. El sistema devolverá la predicción del nivel de obesidad en formato JSON.

--------------------------------------------------
Ejemplo de input para el endpoint /predict
--------------------------------------------------

{
  "Gender": "Male",
  "Age": 25,
  "Height": 1.75,
  "Weight": 85
}

--------------------------------------------------
Ejemplo de respuesta
--------------------------------------------------

{
  "prediction": "Overweight_Level_I"
}

--------------------------------------------------
Plataforma Cloud utilizada
--------------------------------------------------

El despliegue de la API fue realizado utilizando Render, una plataforma cloud
que permite el despliegue automático de aplicaciones a partir de repositorios
en GitHub, integrando CI/CD de forma sencilla.
