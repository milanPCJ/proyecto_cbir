import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from skimage.feature import local_binary_pattern, hog
import faiss
import joblib
import pandas as pd
import os
import pywt
from utils import reducir_ruido_handcrafted

################### Extracción de Características Handcrafted #####################

class Handcrafted:
    def __init__(self, imagen, target_size=(150, 150)):
        """
        Inicializa el extractor de características Handcrafted para una imagen específica.

        Args:
            imagen (str): Ruta a la imagen.
            target_size (tuple): Tamaño al que se redimensionará la imagen (ancho, alto).

        Raises:
            ValueError: Si no se puede leer la imagen proporcionada.
        """
        # Cargar la imagen desde la ruta proporcionada
        imagen0 = cv2.imread(imagen)
        # Reducir el ruido de la imagen utilizando la función definida en utils.py
        imagen = reducir_ruido_handcrafted(imagen0)
        if imagen is None:
            raise ValueError(f"No se pudo leer la imagen {imagen}")
        
        # Redimensionar la imagen si no coincide con el tamaño objetivo
        if imagen.shape[1] != target_size[0] or imagen.shape[0] != target_size[1]:
            self.imagen_color = cv2.resize(imagen, target_size)
        else:
            self.imagen_color = imagen
        
        # Convertir la imagen a escala de grises para ciertas extracciones de características
        self.imagen_grayscale = cv2.cvtColor(self.imagen_color, cv2.COLOR_BGR2GRAY)

        # Inicialización de vectores de características
        self.vector_color = []
        self.vector_textura = []
        self.vector_forma = []
        self.vector_hog = []
        self.vector_histograma_color = []
        self.vector_sobel = []
        self.vector_fft = []
        self.vector_hu = []
        self.vector_wavelet = []

    ################### Extracción de Características Específicas #####################

    def extraccion_color_dcd(self, num_colores_dominantes=8, max_samples=10000):
        """
        Extrae colores dominantes y sus frecuencias utilizando K-Means.

        Args:
            num_colores_dominantes (int): Número de colores dominantes a extraer.
            max_samples (int): Número máximo de píxeles a muestrear para K-Means.

        Outputs:
            self.vector_color (numpy.ndarray): Vector de características de color.
        """
        # Normalizar la imagen a [0, 1]
        imagen_norm = self.imagen_color.astype(np.float32) / 255.0
        pixels = imagen_norm.reshape(-1, 3)

        # Reducir la cantidad de píxeles usando muestreo si es necesario
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels

        # Aplicar K-Means para encontrar colores dominantes
        kmeans = MiniBatchKMeans(n_clusters=num_colores_dominantes, n_init=12, batch_size=1000)
        labels = kmeans.fit_predict(pixels_sample)
        colores_dominantes = kmeans.cluster_centers_

        # Calcular las frecuencias de cada clúster
        frecuencias = np.bincount(labels, minlength=num_colores_dominantes)
        frecuencias_normalizadas = frecuencias / frecuencias.sum()

        # Concatenar los colores dominantes y sus frecuencias
        self.vector_color = np.concatenate([colores_dominantes.flatten(), frecuencias_normalizadas]).astype(np.float32)

        print(f"Vector Color DCD: {len(self.vector_color)}")

    def extraccion_textura_ldrp(self, radio=3, puntos=40, metodo='uniform'):
        """
        Extrae características de textura utilizando Local Directional Relational Patterns (LDRP).

        Args:
            radio (int): Radio de los patrones LDRP.
            puntos (int): Número de puntos en cada patrón LDRP.
            metodo (str): Método para calcular los patrones LDRP.

        Outputs:
            self.vector_textura (numpy.ndarray): Vector de características de textura.
        """
        # Calcular los patrones LDRP
        ldrp = local_binary_pattern(self.imagen_grayscale, puntos, radio, method=metodo)
        # Crear un histograma de los patrones LDRP
        hist, _ = np.histogram(ldrp.ravel(), bins=np.arange(0, puntos + 3), range=(0, puntos + 2))
        # Normalizar el histograma
        self.vector_textura = (hist / hist.sum()).astype(np.float32)  # Normalización opcional global

        print(f"Vector Textura LDRP: {len(self.vector_textura)}")

    def extraccion_forma(self, max_contornos=16):
        """
        Extrae características de forma a partir de contornos en la imagen.

        Args:
            max_contornos (int): Número máximo de contornos a considerar.

        Outputs:
            self.vector_forma (numpy.ndarray): Vector de características de forma.
        """
        # Umbralizar la imagen para encontrar contornos
        _, binary_image = cv2.threshold(self.imagen_grayscale, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        forma_features = []

        for contour in contours[:max_contornos]:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
            forma_features.extend([area, perimeter, aspect_ratio, circularity])

        # Rellenar con ceros si hay menos contornos de los esperados
        while len(forma_features) < max_contornos * 4:
            forma_features.extend([0, 0, 0, 0])

        self.vector_forma = np.array(forma_features, dtype=np.float32)

        print(f"Vector Forma: {len(self.vector_forma)}")

    def extraccion_caracteristicas_HOG(self, orientaciones=6, pixeles_por_celda=(28, 28), celdas_por_bloque=(1, 1)):
        """
        Extrae características HOG (Histogram of Oriented Gradients) de la imagen.

        Args:
            orientaciones (int): Número de orientaciones para los histogramas.
            pixeles_por_celda (tuple): Tamaño en píxeles de cada celda.
            celdas_por_bloque (tuple): Número de celdas por bloque.

        Outputs:
            self.vector_hog (numpy.ndarray): Vector de características HOG.
        """
        # Normalizar la imagen a [0, 1]
        imagen_norm = self.imagen_grayscale.astype(np.float32) / 255.0
        # Calcular las características HOG
        fd = hog(
            imagen_norm,
            orientations=orientaciones,
            pixels_per_cell=pixeles_por_celda,
            cells_per_block=celdas_por_bloque,
            feature_vector=True
        )
        self.vector_hog = np.array(fd, dtype=np.float32)

        print(f"Vector HOG: {len(self.vector_hog)}")

    def extraccion_histograma_color(self, bins=64):
        """
        Extrae el histograma de colores de la imagen en los tres canales B, G y R.

        Args:
            bins (int): Número de bins para cada histograma de color.

        Outputs:
            self.vector_histograma_color (numpy.ndarray): Vector de características del histograma de color.
        """
        # Asegurar que la imagen está en formato uint8
        imagen_color_uint8 = self.imagen_color
        if imagen_color_uint8.dtype != np.uint8:
            imagen_color_uint8 = (imagen_color_uint8 * 255).astype(np.uint8)

        histograma = []
        for i in range(3):  # Canales B, G, R
            hist = cv2.calcHist([imagen_color_uint8], [i], None, [bins], [0, 256])
            hist = hist.flatten()
            histograma.extend(hist)
        self.vector_histograma_color = np.array(histograma, dtype=np.float32)

        print(f"Vector Histograma Color: {len(self.vector_histograma_color)}")

    def extraccion_sobel(self, bins=64):
        """
        Extrae el histograma de magnitudes de bordes utilizando el operador Sobel.

        Args:
            bins (int): Número de bins para el histograma de Sobel.

        Outputs:
            self.vector_sobel (numpy.ndarray): Vector de características del histograma de Sobel.
        """
        # Normalizar la imagen a [0, 1]
        imagen_norm = self.imagen_grayscale.astype(np.float32) / 255.0
        # Aplicar operadores Sobel para obtener las derivadas en x y y
        sobelx = cv2.Sobel(imagen_norm, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(imagen_norm, cv2.CV_64F, 0, 1, ksize=3)
        # Calcular la magnitud del gradiente
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # Crear un histograma de las magnitudes
        hist, _ = np.histogram(magnitude, bins=bins)
        self.vector_sobel = hist.astype(np.float32)

        print(f"Vector Sobel Histogram: {len(self.vector_sobel)}")

    def extraccion_fft(self, bins=64):
        """
        Extrae el histograma del espectro de frecuencia de la imagen utilizando la Transformada Rápida de Fourier (FFT).

        Args:
            bins (int): Número de bins para el histograma de FFT.

        Outputs:
            self.vector_fft (numpy.ndarray): Vector de características del histograma de FFT.
        """
        # Normalizar la imagen a [0, 1]
        imagen_norm = self.imagen_grayscale.astype(np.float32) / 255.0
        # Aplicar la Transformada Rápida de Fourier
        f_transform = np.fft.fft2(imagen_norm)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-7)  # Evitar log(0)
        # Crear un histograma de las magnitudes del espectro
        hist, _ = np.histogram(magnitude_spectrum, bins=bins)
        self.vector_fft = hist.astype(np.float32)

        print(f"Vector FFT Histogram: {len(self.vector_fft)}")

    def extraccion_hu_moments(self):
        """
        Extrae los Momentos de Hu de la imagen para capturar características de forma invariantes.

        Outputs:
            self.vector_hu (numpy.ndarray): Vector de características de los Momentos de Hu normalizados.
        """
        # Calcular los Momentos de Hu
        moments = cv2.moments(self.imagen_grayscale)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Normalizar los Momentos de Hu aplicando la transformación logarítmica
        epsilon = 1e-7  # Pequeño valor para evitar log(0)
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + epsilon)

        # Guardar el resultado normalizado
        self.vector_hu = hu_moments_log.astype(np.float32)

        print(f"Vector Hu Moments (Normalizados): {len(self.vector_hu)}")

    def extraccion_wavelet(self, wavelet='db1', niveles=6):
        """
        Extrae características de frecuencia utilizando la Transformada Wavelet Discreta.

        Args:
            wavelet (str): Tipo de wavelet a utilizar.
            niveles (int): Número de niveles de descomposición.

        Outputs:
            self.vector_wavelet (numpy.ndarray): Vector de características Wavelet.
        """
        # Realizar la descomposición Wavelet
        coeffs = pywt.wavedec2(self.imagen_grayscale, wavelet=wavelet, level=niveles)
        wavelet_features = []
        for level in coeffs[1:]:
            cH, cV, cD = level
            wavelet_features.extend([
                cH.mean(), cH.std(),
                cV.mean(), cV.std(),
                cD.mean(), cD.std()
            ])
        self.vector_wavelet = np.array(wavelet_features, dtype=np.float32)

        print(f"Vector Wavelet: {len(self.vector_wavelet)}")

    ################### Concatenación de Características #####################

    def concatenar_caracteristicas(self):
        """
        Concatenar todas las características extraídas en un solo vector.

        Returns:
            concatenated_vector (numpy.ndarray): Vector de características concatenado.
        """
        concatenated_vector = np.concatenate((
            self.vector_color,
            self.vector_textura,
            self.vector_forma,
            self.vector_hog,
            self.vector_histograma_color,
            self.vector_sobel,
            self.vector_fft,
            self.vector_hu, 
            self.vector_wavelet
        ))
        print('largo_vector_concatenado', len(concatenated_vector))

        return concatenated_vector

################### Extracción de Características VGG16 #####################

def load_VGG16_model():
    """
    Carga el modelo VGG16 preentrenado y lo modifica para extraer características de la capa 'block5_conv3'.

    Returns:
        model (tensorflow.keras.models.Model): Modelo de Keras para la extracción de características.
    """
    # Cargar el modelo VGG16 preentrenado con pesos de ImageNet
    base_model = VGG16(weights='imagenet')
    # Definir un nuevo modelo que termina en la capa 'block5_conv3'
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
    return model

def extract_features_VGG16(image_path, model):
    """
    Extrae características de una imagen utilizando el modelo VGG16 preentrenado.

    Args:
        image_path (str): Ruta a la imagen.
        model (tensorflow.keras.models.Model): Modelo VGG16 modificado para la extracción de características.

    Returns:
        features_flattened (numpy.ndarray): Vector de características extraídas.
    """
    # Cargar la imagen y redimensionarla a 224x224 píxeles
    img = image.load_img(image_path, target_size=(224, 224))
    # Convertir la imagen a un array numpy
    img_array = image.img_to_array(img)
    # Expandir las dimensiones para que sea compatible con el modelo (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocesar la imagen según los requisitos de VGG16
    img_array = preprocess_input(img_array)
    # Extraer las características utilizando el modelo
    features = model.predict(img_array)
    # Aplanar el vector de características
    features_flattened = features.flatten()
    return features_flattened

def normalizar_l2(vector):
    """
    Normaliza un vector utilizando la normalización L2.

    Args:
        vector (numpy.ndarray): Vector a normalizar.

    Returns:
        numpy.ndarray: Vector normalizado.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

################### Carga de Índices FAISS #####################

def load_indices(DB_PATH):
    """
    Carga los índices FAISS preentrenados para Handcrafted y VGG16.

    Args:
        DB_PATH (str): Ruta al directorio donde se encuentran los índices FAISS.

    Returns:
        tuple: Índice FAISS para Handcrafted y VGG16.

    Raises:
        FileNotFoundError: Si alguno de los índices FAISS no se encuentra en la ruta especificada.
    """
    index_handcrafted_path = os.path.join(DB_PATH, 'index_handcrafted.index')
    index_vgg16_path = os.path.join(DB_PATH, 'index_vgg16.index')

    if not os.path.exists(index_handcrafted_path):
        raise FileNotFoundError(f"Índice Handcrafted no encontrado en {index_handcrafted_path}")
    if not os.path.exists(index_vgg16_path):
        raise FileNotFoundError(f"Índice VGG16 no encontrado en {index_vgg16_path}")

    # Leer los índices FAISS desde los archivos
    index_handcrafted = faiss.read_index(index_handcrafted_path)
    index_vgg16 = faiss.read_index(index_vgg16_path)

    return index_handcrafted, index_vgg16
