# Importar librerías para el análisis y la limpieza de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Cargar el csv y empezar a explorar los datos
df = pd.read_csv('../data/raw_data/Phising_Detection_Dataset.csv')
df.drop(columns='Unnamed: 0', inplace=True)

# Faltan pocos datos, por lo que considero que la mejor solución es deshacerme de esas filas
df.dropna(inplace=True)

# Phishing es float por la falta de valores que había. Voy a cambiar la columna a enteros
df['Phising'] = df['Phising'].astype(int)

# Renombrar Columnas:
df.rename(columns={"HttpsInHostname": "HttpsHostname", "NumNumericChars": "NumChars", 'Phising': 'Phishing'}, inplace=True)

# Hay correlación entre PathLevel y PathLenght. Dejaré PathLenght, que es la que más correlaciona con Phishing
df.drop(columns='PathLevel', inplace=True)

# Función para generar distintos DataFrames con distintos tipos de balanceo de datos
def balance_data(df, target_col):
    
    # Separar clases mayoritaria y minoritaria
    df_majority = df[df[target_col] == 0]
    df_minority = df[df[target_col] == 1]

    # Undersampling
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,    
                                       n_samples=len(df_minority), 
                                       random_state=123)
    df_undersampled = pd.concat([df_majority_downsampled, df_minority])
    
    # Oversampling
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     
                                     n_samples=len(df_majority), 
                                     random_state=123)
    df_oversampled = pd.concat([df_majority, df_minority_upsampled])
    
    # SMOTE
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    smote = SMOTE(random_state=123)
    X_res, y_res = smote.fit_resample(X, y)
    df_smote = pd.DataFrame(X_res, columns=X.columns)
    df_smote[target_col] = y_res
    
    return {
        'undersampled': df_undersampled,
        'oversampled': df_oversampled,
        'smote': df_smote
    }

balanced_dataframes = balance_data(df, 'Phishing')

# Acceso a cada dataframe balanceado
df_undersampled = balanced_dataframes['undersampled']
df_oversampled = balanced_dataframes['oversampled']
df_smote = balanced_dataframes['smote']

# Verificación de balance
print(df_undersampled['Phishing'].value_counts())
print(df_oversampled['Phishing'].value_counts())
print(df_smote['Phishing'].value_counts())

# Ahora guardamos los nuevos DataFrames en processed data
df_undersampled.to_csv('../data/processed_data/df_undersample.csv', index=False)
df_oversampled.to_csv('../data/processed_data/df_oversampled.csv', index=False)
df_smote.to_csv('../data/processed_data/df_smote.csv', index=False)