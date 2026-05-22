import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Cargar dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

# Convertir variables categóricas
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Variable objetivo
target = df.columns[-1]

X = df.drop(target, axis=1)
y = df[target]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelos
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()

# Entrenamiento
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Predicciones
rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)

# Métricas Random Forest
print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred, average='weighted'))
print("Recall:", recall_score(y_test, rf_pred, average='weighted'))
print("F1:", f1_score(y_test, rf_pred, average='weighted'))

# Métricas Decision Tree
print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred, average='weighted'))
print("Recall:", recall_score(y_test, dt_pred, average='weighted'))
print("F1:", f1_score(y_test, dt_pred, average='weighted'))

# Guardar modelo
joblib.dump(rf, "models/mejor_modelo.pkl")

print("\nModelo guardado correctamente.")