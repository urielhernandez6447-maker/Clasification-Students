import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

# Primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Información general
print("\nInformación general:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Histogramas
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

# Matriz de correlación
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()