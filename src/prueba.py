import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Cargar modelo
modelo = joblib.load("models/mejor_modelo.pkl")

# Predicción
pred = modelo.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Matriz de Confusión")
plt.show()

print("Evaluación completada correctamente.")