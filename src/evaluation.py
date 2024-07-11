import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Ruta del archivo del modelo (Modelo con mejores resultados)
model_path = 'models/supervised/rf_clf.pkl'

# Ruta del archivo de datos
data_path = 'data/train/train.csv'

# Cargar el modelo
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Leer los datos
data = pd.read_csv(data_path)

# Dividir los datos y el target
X = data.iloc[:, :-1]  # Datos
y = data.iloc[:, -1]  # Labels

# Hacer predicciones
predictions = model.predict(X)

# Calcular las métricas
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, average='weighted')
recall = recall_score(y, predictions, average='weighted')
f1 = f1_score(y, predictions, average='weighted')

# Imprimir los resultados y las métricas
print("Predicciones:", predictions)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Reporte de clasificación detallado
print("\nReporte de Clasificación:\n", classification_report(y, predictions))