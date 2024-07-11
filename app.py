import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

# 'NumDots', 'UrlLength', 'NumDash', 'AtSymbol', 'IpAddress', 'HttpsHostname', 'PathLength', 'NumChars', 'Phishing'

# Cargar el modelo desde el archivo
model_path = 'models/supervised/rfc_und.pkl'
with open(model_path, 'rb') as file:
    model = pkl.load(file)

# Función para verificar si la URL contiene una dirección IP
def has_ip_address(url):
    ip_pattern = re.compile(r'(\d{1,3}\.){3}\d{1,3}')
    return 1 if ip_pattern.search(url) else 0

# Función para preprocesar la URL
def preprocess_url(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    path = parsed_url.path

    # Extraer las variables
    num_dots = url.count('.')
    url_length = len(url)
    num_dash = url.count('-')
    at_symbol = 1 if '@' in url else 0
    ip_address = has_ip_address(url)
    https_in_hostname = 1 if 'https' in hostname else 0
    path_length = len(path)
    num_numeric_chars = sum(c.isdigit() for c in url)
    
    # Crear un dataframe con los datos preprocesados
    data = {
        'NumDots': num_dots,
        'UrlLength': url_length,
        'NumDash': num_dash,
        'AtSymbol': at_symbol,
        'IpAddress': ip_address,
        'HttpsHostname': https_in_hostname,
        'PathLength': path_length,
        'NumChars': num_numeric_chars
    }
    
    return pd.DataFrame([data])

# Función para hacer predicciones
def predict(features):
    return model.predict(features)[0]

# Título de la aplicación
st.title('Detección de Phishing en Páginas Web')

# Ingreso de URL por parte del usuario
url = st.text_input('Ingresa la URL para verificar')

if url:
    # Preprocesar la URL
    preprocessed_data = preprocess_url(url)
    
    # Hacer la predicción
    prediction = predict(preprocessed_data)
    
    # Mostrar el resultado
    st.write(f'La página es {"fraudulenta" if prediction == 0 else "legítima"}')