import time
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import joblib
import io
from pathlib import Path
import tempfile
from collections import Counter


import streamlit as st
from streamlit_cropper import st_cropper

from feature_extractors import (
    Handcrafted,
    load_VGG16_model,
    extract_features_VGG16,
    normalizar_l2,
    load_indices
)
from utils import load_csv, reciprocal_rank, f1_score_calculator, precision_at_k, ndcg_at_k, verificar_imagenes


# Configuración de la página
st.set_page_config(layout="wide")
st.title('CBIR IMAGE SEARCH')
 
# Definir rutas
FILES_PATH = str(Path(__file__).parent.parent.resolve())
#st.write(f"FILES_PATH: {FILES_PATH}")  # Depuración

IMAGES_PATH = os.path.join(FILES_PATH, 'data', 'training_images')
#st.write(f"IMAGES_PATH: {IMAGES_PATH}")  # Depuración

DB_PATH = os.path.join(FILES_PATH, 'data', 'indices')
#st.write(f"DB_PATH: {DB_PATH}")  # Depuración

DB_FILE = 'images_labels.csv'  # Asegúrate de que este nombre coincide con tu archivo CSV
#st.write(f"DB_FILE: {DB_FILE}")  # Depuración

# Cargar datos al inicio para optimizar
@st.cache_resource
def cargar_datos():
    #st.write("Cargando CSV...")
    df = load_csv(DB_PATH)
    #st.write(f"DataFrame cargado con {len(df)} filas.")  # Depuración

    #st.write("Cargando índices FAISS...")
    index_handcrafted, index_vgg16 = load_indices(DB_PATH)
    #st.write("Índices FAISS cargados correctamente.")  # Depuración

    #t.write("Cargando scaler Handcrafted...")
    scaler_handcrafted_path = os.path.join(DB_PATH, 'scaler_handcrafted.pkl')
    #st.write(f"Ruta del scaler_handcrafted.pkl: {scaler_handcrafted_path}")  # Depuración
    scaler_handcrafted = joblib.load(scaler_handcrafted_path)
    #st.write(f"El StandardScaler espera vectores de dimensión: {scaler_handcrafted.mean_.shape[0]}")  # Depuración

    #st.write("Cargando modelo VGG16...")
    vgg_model = load_VGG16_model()
    #st.write("Modelo VGG16 cargado.")  # Depuración

    return df, index_handcrafted, index_vgg16, scaler_handcrafted, vgg_model


df, index_handcrafted, index_vgg16, scaler_handcrafted, vgg_model = cargar_datos()



# Llamar a la función de verificación después de cargar los datos
verificar_imagenes(df, IMAGES_PATH)

# Función para obtener la lista de imágenes (rutas relativas)
def get_image_list():
    image_list = df.apply(lambda row: Path(row['label']) / row['image_name'], axis=1).astype(str).tolist()
    image_list = [Path(p).as_posix() for p in image_list]
    #st.write(f"Primeras 5 rutas de imágenes en la lista: {image_list[:5]}")  # Depuración
    return image_list

def retrieve_image(img_path, feature_extractor, vgg_model, scaler_handcrafted, index_handcrafted, index_vgg16, n_imgs=30):
    #st.write(f"Ruta de la imagen de consulta: {img_path}")  # Depuración

    if feature_extractor == 'Handcrafted':
        #st.write("Usando extractor Handcrafted.")  # Depuración
        # Extraer características Handcrafted usando la ruta
        extractor = Handcrafted(img_path)
        extractor.extraccion_color_dcd()
        extractor.extraccion_textura_ldrp()
        extractor.extraccion_forma()
        extractor.extraccion_caracteristicas_HOG()
        extractor.extraccion_histograma_color()
        extractor.extraccion_sobel()
        extractor.extraccion_fft()
        extractor.extraccion_hu_moments()
        extractor.extraccion_wavelet()
 
        vector = extractor.concatenar_caracteristicas()
        #st.write("Características Handcrafted extraídas.")  # Depuración

        # Normalizar con StandardScaler
        vector_norm = scaler_handcrafted.transform([vector]).astype('float32')
        #st.write("Características Handcrafted normalizadas.")  # Depuración

        # Buscar en FAISS Handcrafted
        _, indices = index_handcrafted.search(vector_norm, k=n_imgs)
        #st.write(f"Índices obtenidos de FAISS Handcrafted: {indices}")  # Depuración

        return indices[0], 'Handcrafted'

    elif feature_extractor == 'VGG16':
        #st.write("Usando extractor VGG16.")  # Depuración
        # Extraer características VGG16 usando la ruta
        features_vgg = extract_features_VGG16(img_path, vgg_model)
        features_vgg_norm = normalizar_l2(features_vgg).astype('float32')
        # FAISS requiere una matriz 2D
        features_vgg_norm = np.expand_dims(features_vgg_norm, axis=0)
        #st.write("Características VGG16 extraídas y normalizadas.")  # Depuración

        # Buscar en FAISS VGG16
        _, indices = index_vgg16.search(features_vgg_norm, k=n_imgs)
        #st.write(f"Índices obtenidos de FAISS VGG16: {indices}")  # Depuración

        return indices[0], 'VGG16'

    else:
        st.error("Método de extracción no soportado.")
        raise ValueError("Método de extracción no soportado.")



def main():
    col1, col2 = st.columns(2)

    with col1:
        st.header('CONSULTA')

        st.subheader('Elige el extractor de características')
        option = st.selectbox('Método de Extracción', ('Handcrafted', 'VGG16'))
        st.write(f"Extractor seleccionado: {option}")  # Depuración

        st.subheader('Sube una imagen')
        img_file = st.file_uploader(label='Selecciona una imagen', type=['png', 'jpg', 'jpeg'])

        if img_file:
            st.write("Imagen subida por el usuario.")  # Depuración
            img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
            st.write("Abriendo imagen subida.")  # Depuración

            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')

            st.write("Vista previa de la imagen recortada")
            cropped_img_thumbnail = cropped_img.copy()
            cropped_img_thumbnail.thumbnail((150, 150))
            st.image(cropped_img_thumbnail)

            # Guardar la imagen recortada temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                cropped_img.save(tmp, format='JPEG')
                temp_img_path = tmp.name
                #st.write(f"Imagen temporal guardada en: {temp_img_path}")  # Depuración
                
            query_image_name = img_file.name
            if '_' in query_image_name:
                query_class_label = query_image_name.split('_')[0]
            else:
                query_class_label = Path(query_image_name).stem  # En caso de que no tenga '_'
            st.write(f"Clase de la imagen de consulta: **{query_class_label}**")

    with col2:
        st.header('RESULTADO')
        if img_file:
            st.markdown('**Buscando .......**')
            start = time.time()

            try:
                # Recuperar imágenes similares (30)
                indices, extractor_used = retrieve_image(
                    temp_img_path,  # Pasar la ruta de la imagen
                    option,
                    vgg_model,
                    scaler_handcrafted,
                    index_handcrafted,
                    index_vgg16,
                    n_imgs=20  # Recuperar 30 imágenes
                )

                # Obtener la lista de imágenes (rutas relativas)
                image_list = get_image_list()
                st.write(f"Total de imágenes en la base de datos: {len(image_list)}")  # Depuración

                end = time.time()
                st.markdown(f'**Terminado en {end - start:.2f} segundos**')

                # Inicializar lista para almacenar las clases de las imágenes recuperadas
                retrieved_classes = []

                # Recuperar clases de todas las 30 imágenes recuperadas
                for img_index in indices:
                    if not isinstance(img_index, (int, np.integer)):
                        continue
                    if img_index >= len(image_list):
                        continue
                    img_relative_path = image_list[img_index]
                    class_label = Path(img_relative_path).parts[0]
                    retrieved_classes.append(class_label)

                # Mostrar las primeras 10 imágenes similares
                n_display = 10  # Número de imágenes a mostrar
                st.subheader(f'Imágenes similares ({extractor_used} Features):')
                cols = st.columns(5)
                for i in range(min(n_display, len(indices))):
                    img_index = indices[i]
                    if not isinstance(img_index, (int, np.integer)):
                        st.warning(f"Índice no es entero: {img_index} (Tipo: {type(img_index)})")
                        continue
                    if img_index >= len(image_list):
                        st.warning(f"Índice de imagen fuera de rango: {img_index}")
                        continue
                    img_relative_path = image_list[img_index]  # Por ejemplo, 'Edificios/53.jpg'
                    img_path = os.path.join(IMAGES_PATH, img_relative_path)  # 'data/training_images/Edificios/53.jpg'
                    
                    # Extraer la clase de la imagen recuperada de manera robusta
                    class_label = Path(img_relative_path).parts[0]
    
                    if os.path.exists(img_path):
                        try:
                            image_similar = Image.open(img_path).convert('RGB')
                            cols[i % 5].image(image_similar, caption=os.path.basename(img_relative_path), use_container_width=True)
                        except Exception as e:
                            cols[i % 5].error(f"Error al abrir la imagen {img_path}: {e}")
                    else:
                        cols[i % 5].write(f"Imagen no encontrada: {img_path}")
                        st.warning(f"Imagen no encontrada: {img_path}")

                
                # **Añadir la siguiente línea para mostrar la cantidad de imágenes correctas en las 10 mostradas**
                display_classes = retrieved_classes[:n_display]
                num_correct_display = sum(1 for c in display_classes if c == query_class_label)
                st.write(f"**Primeras 10 imágenes correctamente clasificadas en \"{query_class_label}\" son:** {num_correct_display}/{n_display}")
                
                # Calcular métricas de rendimiento basadas en las 30 imágenes recuperadas
                num_same_class = sum(1 for c in retrieved_classes if c == query_class_label)
                total_retrieved = len(retrieved_classes)
                precision = num_same_class / total_retrieved if total_retrieved > 0 else 0

                # Calcular total_relevant dinámicamente
                total_relevant = df[df['label'] == query_class_label].shape[0]

                # Calcular Recall
                recall = num_same_class / total_relevant if total_relevant > 0 else 0

                # Calcular F1 Score
                f1 = f1_score_calculator(precision, recall)

                # Precisión@5 y Precisión@10
                p_at_5 = precision_at_k(retrieved_classes, query_class_label, k=5)

                # Reciprocal Rank (RR)
                rr = reciprocal_rank(retrieved_classes, query_class_label)

                # nDCG@10
                ndcg_score = ndcg_at_k(retrieved_classes, query_class_label, k=10, total_relevant=total_relevant)
                # Mostrar métricas utilizando st.metric para una visualización más elegante
                st.write("### Métricas de Rendimiento, 20 imágenes recuperadas de un total de 30 por clase")
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Precisión", f"{precision:.2f}")
                metric2.metric("Recall", f"{recall:.2f}")
                metric3.metric("F1 Score", f"{f1:.2f}")

                metric4, metric5, metric6 = st.columns(3)
                metric4.metric("Precisión@5", f"{p_at_5:.2f}")
                metric6.metric("Reciprocal Rank (RR)", f"{rr:.2f}")
                metric5.metric("nDCG@10", f"{ndcg_score:.2f}")


                # Mostrar distribución de clases en las imágenes recuperadas como tabla
                st.write("**Distribución de clases en las imágenes recuperadas:**")
                class_counts = Counter(retrieved_classes)
                
                classes = list(class_counts.keys())
                counts = list(class_counts.values())

                # Inicializar el estado de la vista en Streamlit (tabla o gráfico)
                if 'view' not in st.session_state:
                    st.session_state.view = 'table'  # Valor inicial

                # Definir una función para alternar la vista entre tabla y gráfico
                def toggle_view():
                    if st.session_state.view == 'table':
                        st.session_state.view = 'chart'
                    else:
                        st.session_state.view = 'table'

                # Definir el icono y el texto del botón basado en el estado actual
                if st.session_state.view == 'table':
                    button_label = "📊 Mostrar Gráfico"
                else:
                    button_label = "📋 Mostrar Tabla"

                # Botón para alternar la vista
                st.button(button_label, on_click=toggle_view)

                # Mostrar la vista basada en el estado
                if st.session_state.view == 'table':
                    # Crear un DataFrame con una sola fila: 'Cantidad' y columnas 'Clase'
                    df_counts = pd.DataFrame([counts], columns=classes)
                    
                    # Crear la tabla en formato Markdown
                    headers = " | ".join(df_counts.columns)
                    separators = " | ".join(["---"] * len(df_counts.columns))
                    values = " | ".join(map(str, df_counts.iloc[0]))
                    table_md = f"| {headers} |\n| {separators} |\n| {values} |"
                    
                    # Mostrar la tabla usando st.markdown
                    st.markdown(table_md, unsafe_allow_html=True)

                else:
                    # Crear un DataFrame para el gráfico de barras
                    class_counts_df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Cantidad'])
                    
                    # Mostrar gráfico de barras
                    st.bar_chart(class_counts_df)
            except Exception as e:
                st.error(f"Error durante la búsqueda: {e}")

if __name__ == '__main__':
    main()
