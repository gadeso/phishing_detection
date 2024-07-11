import pickle as pkl
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout

df_train = pd.read_csv('../data/train/train.csv')
df_test = pd.read_csv('../data/test/test.csv')


# 1.1 Supervisado: Logistic Regresion
# Separar características y etiquetas
X_train = df_train.drop('Phishing', axis=1)
y_train = df_train['Phishing']
X_test = df_test.drop('Phishing', axis=1)
y_test = df_test['Phishing']

# Entrenar el modelo
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Hacer predicciones
y_pred_log_reg = log_reg.predict(X_test)

# Evaluar el modelo
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg)}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg)}")
print(f"F1 Score: {f1_score(y_test, y_pred_log_reg)}")
print(classification_report(y_test, y_pred_log_reg))

# Guardar modelo en pkl
with open("../models/supervised/log_reg.pkl", "wb") as file:
    pkl.dump(log_reg, file)


# 1.2 Supervisado: Random Forest
# Entrenar el modelo
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Hacer predicciones
y_pred_rf = rf_clf.predict(X_test)

# Evaluar el modelo
print("Random Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf)}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# Guardar modelo en pkl
with open("../models/supervised/rf_clf.pkl", "wb") as file:
    pkl.dump(rf_clf, file)


# 2.1 No Supervisado: K-Means
# Entrenar el modelo
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)

# Hacer predicciones
y_pred_kmeans = kmeans.predict(X_test)

# Convertir las etiquetas predichas a la misma escala que las etiquetas verdaderas
y_pred_kmeans = [1 if label == kmeans.cluster_centers_[1][0] > kmeans.cluster_centers_[0][0] else 0 for label in y_pred_kmeans]

# Evaluar el modelo
print("KMeans:")
print(confusion_matrix(y_test, y_pred_kmeans))
print(f"Accuracy: {accuracy_score(y_test, y_pred_kmeans)}")
print(f"Precision: {precision_score(y_test, y_pred_kmeans)}")
print(f"Recall: {recall_score(y_test, y_pred_kmeans)}")
print(f"F1 Score: {f1_score(y_test, y_pred_kmeans)}")
print(classification_report(y_test, y_pred_kmeans))

# Guardar modelo en pkl
with open("../models/unsupervised/kmeans.pkl", "wb") as file:
    pkl.dump(kmeans, file)


# 2.2 No Supervisado: Isolation Forest
# Entrenar el modelo
iso_forest = IsolationForest(random_state=42)
iso_forest.fit(X_train)

# Hacer predicciones
y_pred_iso_forest = iso_forest.predict(X_test)

# Convertir las etiquetas predichas (1 -> -1 y -1 -> 1)
y_pred_iso_forest = [0 if x == 1 else 1 for x in y_pred_iso_forest]

# Evaluar el modelo
print("Isolation Forest:")
print(confusion_matrix(y_test, y_pred_iso_forest))
print(f"Accuracy: {accuracy_score(y_test, y_pred_iso_forest)}")
print(f"Precision: {precision_score(y_test, y_pred_iso_forest)}")
print(f"Recall: {recall_score(y_test, y_pred_iso_forest)}")
print(f"F1 Score: {f1_score(y_test, y_pred_iso_forest)}")
print(classification_report(y_test, y_pred_iso_forest))

# Guardar modelo en pkl
with open("../models/unsupervised/iso_forest.pkl", "wb") as file:
    pkl.dump(iso_forest, file)


# 3 Deep Learning:
# Definir el modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Guardar el modelo
model.save('../models/deep_learning_model.keras')


# Definir el modelo con más capas y dropout para prevenir sobreajuste
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo con una tasa de aprendizaje ajustada
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Definir los callbacks
early_stopping = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('deep_learning_model.keras', monitor='accuracy', save_best_only=True)

# Entrenar el modelo
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint])

# Evaluar el modelo guardado
y_pred_dl = (model.predict(X_test) > 0.5).astype("int32")

# Evaluar el modelo
print("Deep Learning Improved Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dl)}")
print(f"Precision: {precision_score(y_test, y_pred_dl)}")
print(f"Recall: {recall_score(y_test, y_pred_dl)}")
print(f"F1 Score: {f1_score(y_test, y_pred_dl)}")
print(classification_report(y_test, y_pred_dl))
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_dl)}")

# Guardar el modelo
model.save('../models/dl_2.keras')


# Prueba de Deep Learning quitando outliers: 
df_undersampled = pd.read_csv('../data/processed_data/df_undersample.csv')

# Detectar outliers usando el método IQR
Q1 = df_undersampled.quantile(0.25)
Q3 = df_undersampled.quantile(0.75)
IQR = Q3 - Q1

# Eliminar outliers
df_no_outliers = df_undersampled[~((df_undersampled < (Q1 - 1.5 * IQR)) |(df_undersampled > (Q3 + 1.5 * IQR))).any(axis=1)]

# Separar características y etiquetas
X_no_outliers = df_no_outliers.drop('Phishing', axis=1)
y_no_outliers = df_no_outliers['Phishing']

# Shuffle manualmente y dividir en train y test
df_no_outliers_shuffled = df_no_outliers.sample(frac=1, random_state=42).reset_index(drop=True)
train_size_no_outliers = int(0.8 * len(df_no_outliers_shuffled))
df_train_no_outliers = df_no_outliers_shuffled[:train_size_no_outliers]
df_test_no_outliers = df_no_outliers_shuffled[train_size_no_outliers:]

X_train_no_outliers = df_train_no_outliers.drop('Phishing', axis=1)
y_train_no_outliers = df_train_no_outliers['Phishing']
X_test_no_outliers = df_test_no_outliers.drop('Phishing', axis=1)
y_test_no_outliers = df_test_no_outliers['Phishing']

# Definir el modelo con más capas y dropout para prevenir sobreajuste
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_no_outliers.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo con una tasa de aprendizaje ajustada
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Definir los callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('deep_learning_model.keras', monitor='val_loss', save_best_only=True)

# Entrenar el modelo
history = model.fit(X_train_no_outliers, y_train_no_outliers,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint])

# Hacer predicciones
y_pred_dl_no_outliers = (model.predict(X_test_no_outliers) > 0.5).astype("int32")

# Evaluar el modelo
print("Deep Learning Improved Model without Outliers:")
print(f"Accuracy: {accuracy_score(y_test_no_outliers, y_pred_dl_no_outliers)}")
print(f"Precision: {precision_score(y_test_no_outliers, y_pred_dl_no_outliers)}")
print(f"Recall: {recall_score(y_test_no_outliers, y_pred_dl_no_outliers)}")
print(f"F1 Score: {f1_score(y_test_no_outliers, y_pred_dl_no_outliers)}")
print(classification_report(y_test_no_outliers, y_pred_dl_no_outliers))
print(f"ROC AUC: {roc_auc_score(y_test_no_outliers, y_pred_dl_no_outliers)}")
# Como se esperaba, el mejor resultado se ha obtenido usando Random Forest Classifier. Avanzaré con ese modelo para ver si logro mejorarlo


# GridSearch:
# Definir los hiperparámetros a buscar
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Configurar la búsqueda de hiperparámetros
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1, scoring='f1')

# Realizar la búsqueda de hiperparámetros
grid_search.fit(X_train_no_outliers, y_train_no_outliers)

# Mejor conjunto de hiperparámetros
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Mejor modelo encontrado por GridSearchCV
best_rf_clf = grid_search.best_estimator_

# Hacer predicciones y evaluar el mejor modelo
y_pred_best_rf = best_rf_clf.predict(X_test_no_outliers)

print("Random Forest Classifier with GridSearchCV:")
print(f"Accuracy: {accuracy_score(y_test_no_outliers, y_pred_best_rf)}")
print(f"Precision: {precision_score(y_test_no_outliers, y_pred_best_rf)}")
print(f"Recall: {recall_score(y_test_no_outliers, y_pred_best_rf)}")
print(f"F1 Score: {f1_score(y_test_no_outliers, y_pred_best_rf)}")
print(classification_report(y_test_no_outliers, y_pred_best_rf))


# Pipelines:
# Crear el pipeline con preprocesamiento y modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalizar los datos
    ('classifier', RandomForestClassifier(random_state=42))
])

# Configurar la búsqueda de hiperparámetros del pipeline
param_grid_pipeline = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Realizar la búsqueda de hiperparámetros
grid_search_pipeline = GridSearchCV(pipeline, param_grid=param_grid_pipeline, cv=3, n_jobs=-1, verbose=2, scoring='f1')
grid_search_pipeline.fit(X_train_no_outliers, y_train_no_outliers)

# Mejor conjunto de hiperparámetros
best_params_pipeline = grid_search_pipeline.best_params_
print(f"Best parameters from pipeline: {best_params_pipeline}")

# Mejor modelo encontrado por GridSearchCV
best_pipeline_rf_clf = grid_search_pipeline.best_estimator_

# Hacer predicciones y evaluar el mejor modelo del pipeline
y_pred_best_pipeline_rf = best_pipeline_rf_clf.predict(X_test_no_outliers)

print("Random Forest Classifier with GridSearchCV and Pipeline:")
print(f"Accuracy: {accuracy_score(y_test_no_outliers, y_pred_best_pipeline_rf)}")
print(f"Precision: {precision_score(y_test_no_outliers, y_pred_best_pipeline_rf)}")
print(f"Recall: {recall_score(y_test_no_outliers, y_pred_best_pipeline_rf)}")
print(f"F1 Score: {f1_score(y_test_no_outliers, y_pred_best_pipeline_rf)}")
print(classification_report(y_test_no_outliers, y_pred_best_pipeline_rf))