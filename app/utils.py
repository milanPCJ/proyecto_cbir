import cv2
import os
import pandas as pd
import numpy as np 
from pathlib import Path
import streamlit as st

################### Métricas #####################

# Función: precision_at_k
## Descripción: Calcula la precisión en las primeras K imágenes recuperadas.
'''
Inputs:
    retrieved_classes (list): Lista de clases de las imágenes recuperadas.
    query_class_label (str): Clase de la imagen de consulta.
    k (int): Número de imágenes a considerar para la precisión.

Outputs:
    float: Precisión@K.
'''
def precision_at_k(retrieved_classes, query_class_label, k):
    relevant_at_k = sum(1 for c in retrieved_classes[:k] if c == query_class_label)
    return relevant_at_k / k

# Función: reciprocal_rank
## Descripción: Calcula el Reciprocal Rank (RR) de las imágenes recuperadas.
'''
Inputs:
    retrieved_classes (list): Lista de clases de las imágenes recuperadas.
    query_class_label (str): Clase de la imagen de consulta.

Outputs:
    float: Reciprocal Rank (RR).
'''
def reciprocal_rank(retrieved_classes, query_class_label):
    try:
        idx = retrieved_classes.index(query_class_label) + 1
        return 1 / idx
    except ValueError:
        return 0

# Función: ndcg_at_k
## Descripción: Calcula el nDCG@K (Normalized Discounted Cumulative Gain) de las imágenes recuperadas.
'''
Inputs:
    retrieved_classes (list): Lista de clases de las imágenes recuperadas.
    query_class_label (str): Clase de la imagen de consulta.
    k (int): Número de imágenes a considerar para nDCG.
    total_relevant (int): Número total de imágenes relevantes en el dataset para la clase de consulta.

Outputs:
    float: nDCG@K.
'''
def ndcg_at_k(retrieved_classes, query_class_label, k, total_relevant):
    dcg = 0
    for i, cls in enumerate(retrieved_classes[:k], start=1):
        rel = 1 if cls == query_class_label else 0
        dcg += (2 ** rel - 1) / np.log2(i + 1)
    # IDCG: caso ideal donde todas las imágenes relevantes están al principio
    ideal_rels = [1] * min(k, total_relevant)
    idcg = sum((2 ** rel - 1) / np.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))
    ndcg_score = dcg / idcg if idcg > 0 else 0
    return ndcg_score

# Función: f1_score_calculator
## Descripción: Calcula el F1 Score a partir de la precisión y el recall.
'''
Inputs:
    precision (float): Precisión del modelo.
    recall (float): Recall del modelo.

Outputs:
    float: F1 Score.
'''
def f1_score_calculator(precision, recall):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0

################# Procesamiento de Imágenes #######################

# Función: reducir_ruido_handcrafted
## Descripción: Aplica técnicas de reducción de ruido a una imagen para mejorar la extracción de características Handcrafted.
'''
Inputs:
    image (numpy.ndarray): Imagen en formato BGR o escala de grises.

Outputs:
    numpy.ndarray: Imagen denoised en formato BGR.
'''
def reducir_ruido_handcrafted(image):
    # Convertir la imagen a escala de grises (si no lo está ya)
    if len(image.shape) == 3:  # Si la imagen tiene 3 canales (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro bilateral para preservar bordes
    denoised_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convertir la imagen a color (BGR) si está en escala de grises
    if len(denoised_image.shape) == 2:  # Si la imagen tiene un solo canal (grayscale)
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)

    return denoised_image

################### Carga y Verificación de Datos #####################

# Función: load_csv
## Descripción: Carga un archivo CSV que contiene etiquetas e información de imágenes.
'''
Inputs:
    DB_PATH (str): Ruta al directorio donde se encuentra el archivo CSV.

Outputs:
    pandas.DataFrame: DataFrame con los datos cargados del CSV.
'''
def load_csv(DB_PATH):
    csv_path = os.path.join(DB_PATH, 'images_labels.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo CSV no encontrado en {csv_path}")
    df = pd.read_csv(csv_path)
    return df

# Función: verificar_imagenes
## Descripción: Verifica que todas las imágenes listadas en el DataFrame existan en la carpeta de imágenes.
'''
Inputs:
    df (pandas.DataFrame): DataFrame que contiene las etiquetas y nombres de las imágenes.
    imagenes_path (str): Ruta al directorio que contiene las imágenes.

Outputs:
    None. Muestra mensajes de advertencia o éxito en Streamlit.
'''
def verificar_imagenes(df, imagenes_path):
    """
    Verifica que todas las imágenes listadas en el DataFrame existan en la carpeta de imágenes.
    
    Inputs:
        df (pandas.DataFrame): DataFrame que contiene las etiquetas y nombres de las imágenes.
        imagenes_path (str): Ruta al directorio que contiene las imágenes.
    
    Outputs:
        None. Muestra mensajes de advertencia o éxito en Streamlit.
    """
    faltantes = []
    for idx, row in df.iterrows():
        img_relative_path = Path(row['label']) / row['image_name']
        full_path = Path(imagenes_path) / img_relative_path
        if not full_path.exists():
            faltantes.append(str(img_relative_path))
    if faltantes:
        st.warning(f"Las siguientes imágenes no se encontraron en {imagenes_path}: {faltantes}")
    else:
        st.success("Todas las imágenes están presentes en la carpeta de imágenes.")

