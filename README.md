# **Proyecto: Extracción de Características y Búsqueda de Imágenes**

Este proyecto, desarrollado como parte del curso *Algoritmos y Arquitecturas para el Procesado de Imágenes*, implementa un sistema de recuperación de imágenes basado en el contenido (*Content-Based Image Retrieval*, CBIR). Utiliza métodos de extracción de características tanto manuales como basados en redes neuronales profundas, junto con el índice FAISS, para realizar búsquedas eficientes.

---

## **Objetivo**
El objetivo es implementar un sistema CBIR que permita extraer características visuales relevantes de imágenes y buscar eficientemente en una base de datos para encontrar imágenes similares, comparando distintos enfoques.

---

## **Métodos de Extracción de Características**

### **Handcrafted (Características Manuales):**
- **Reducción de Ruido:** Se aplica un preprocesamiento para reducir el ruido en las imágenes.
- **Métodos Implementados:**
  - **Color (DCD):** Identifica colores dominantes y su distribución.
  - **Textura (LDRP):** Extrae patrones locales mediante Local Binary Patterns (LBP).
  - **Forma:** Calcula características geométricas como área, perímetro y circularidad.
  - **HOG:** Representa estructuras visuales con gradientes orientados.
  - **Histograma de Color:** Analiza la distribución de colores en cada canal RGB.
  - **Sobel:** Detecta bordes usando gradientes.
  - **FFT:** Describe frecuencias de la imagen.
  - **Momentos de Hu:** Caracteriza formas invariantes.
  - **Wavelet:** Representa descomposiciones multinivel de la imagen.

### **VGG16 (Red Neuronal):**
- Se utiliza una arquitectura preentrenada VGG16, adaptada para la extracción de características de alto nivel.
- **Nota:** No se aplica reducción de ruido, ya que la arquitectura de la red maneja directamente imágenes crudas.

---

## **Base de Datos**
- **Base Principal:** 180 imágenes de 6 clases (*Edificios, Bosques, Glaciares, Montañas, Mar, Calles*), obtenidas de Kaggle, utilizadas para construir el índice.
- **Imágenes de Consulta:** 120 imágenes de las mismas clases, pero distintas a las de la base, utilizadas para realizar las consultas y evaluaciones.

---

## **Búsqueda y Herramientas**
- **Índice FAISS:** Se utiliza el índice *Flat*, debido a la cantidad moderada de imágenes en la base de datos.
- **Interfaz de Usuario (Streamlit):** Una aplicación interactiva permite a los usuarios cargar imágenes, realizar búsquedas y comparar los resultados obtenidos por los métodos *Handcrafted* y *VGG16*.
