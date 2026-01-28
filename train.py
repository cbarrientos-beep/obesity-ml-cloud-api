import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar dataset
df = pd.read_csv("Obesity_prediction.csv")

# 2. Variables usadas por la API
features = ["Gender", "Age", "Height", "Weight"]
target = df.columns[-1]  # NObeyesdad

# 3. Seleccionar solo columnas necesarias
df = df[features + [target]]

# 4. Codificar Gender
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

encoders = {
    "Gender": le_gender
}

# 5. Separar X e y
X = df[features]
y = df[target]

# 6. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 8. Evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo: {accuracy:.2f}")

# 9. Guardar modelo
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# 10. Guardar encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

