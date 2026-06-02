# Highway Panel OCR — Lectura Automática de Paneles de Autopista

> Sistema de Reconocimiento Óptico de Caracteres (OCR) que **localiza y lee el texto** de los paneles informativos de autopista. Combina un clasificador de caracteres entrenado con *machine learning* clásico (LDA + Normal Bayes, **96% de accuracy**) con un lector de paneles basado en componentes conexas y RANSAC, e integra el detector de la Práctica 1 para formar un sistema completo de detección + lectura.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--learn-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/scikit--image-HOG-yellow.svg" alt="scikit-image">
  <img src="https://img.shields.io/badge/NumPy-informational.svg" alt="NumPy">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

-----

## Descripción general

Este proyecto construye un sistema OCR completo para leer el contenido textual de los paneles azules de señalización de autopista. Partiendo de los paneles detectados en la [Práctica 1](https://github.com/joel-domene/Higway-Sign-Detection-OpenCV), el sistema localiza los caracteres individuales, los agrupa en líneas de texto y los reconoce mediante un clasificador entrenado.

El núcleo del proyecto es un **pipeline de machine learning clásico de extremo a extremo**: extracción de características → reducción de dimensionalidad → clasificación → evaluación rigurosa. Se entrena y valida un clasificador de caracteres, se comparan sistemáticamente varias combinaciones de características, reductores y clasificadores, y se aplica el mejor a la lectura de paneles reales, evaluando el resultado con la distancia de edición de Levenshtein.

> **Contexto:** proyecto académico desarrollado **en equipo de 3 personas** como Práctica 2 de la asignatura Visión Artificial (URJC, curso 2025/26). Continúa la Práctica 1 (detección de paneles).

-----

## Características principales

- **Clasificador de caracteres** (62 clases: dígitos, minúsculas y mayúsculas) con accuracy de **0.960**.
- **Diseño orientado a objetos**: clase base `OCRClassifier` y clasificadores derivados, organizados en el paquete `ocr/`.
- **Sistema configurable** que permite intercambiar características (gris / HOG), reducción (LDA / PCA / ninguna) y clasificador (Normal Bayes, KNN, SVM, euclídeo) mediante un único parámetro `--classifier`.
- **Comparativa rigurosa** de 8 combinaciones de clasificadores con métricas y matriz de confusión.
- **Lector de paneles** completo: detección de caracteres por componentes conexas, agrupación en líneas de texto con **RANSAC** y separación de caracteres pegados por proyección vertical.
- **Evaluación por distancia de Levenshtein** contra anotaciones manuales.
- **Sistema integrado** detección (P1) + OCR (P2): localiza el panel en la imagen de carretera y lee su contenido en un solo paso.

-----

## Pipeline del sistema

El proyecto se organiza en cuatro ejercicios encadenados:

```
                ┌──────────────────────────────────────────────────────────┐
   train_ocr ──▶│ Ej.1  Clasificador de caracteres                          │
                │   gris → umbralizado adaptativo → contorno → resize 25×25  │
                │   → vector 625 → LDA (625→61) → Normal Bayes               │
                │   Accuracy 0.960                                           │
                └──────────────────────────────────────────────────────────┘
                                          │
                ┌──────────────────────────────────────────────────────────┐
                │ Ej.2  Comparativa de alternativas                         │
                │   {gris, HOG} × {LDA, PCA, none} × {Bayes, KNN, SVM, eucl} │
                │   Mejor sistema: LDA + Normal Bayes                        │
                └──────────────────────────────────────────────────────────┘
                                          │
                ┌──────────────────────────────────────────────────────────┐
 panel  ───────▶│ Ej.3  Lectura de paneles recortados                       │
 recortado      │   CLAHE + Otsu → componentes conexas → filtrado           │
                │   → RANSAC (líneas de texto) → separación por proyección   │
                │   → clasificación → texto del panel                        │
                │   Levenshtein medio 10.1 · ≤5 en 36% · ≤10 en 54%          │
                └──────────────────────────────────────────────────────────┘
                                          │
                ┌──────────────────────────────────────────────────────────┐
 imagen ───────▶│ Ej.4  Sistema completo (detección P1 + OCR P2)            │
 de carretera   │   detector MSER → recorte de panel → lector OCR           │
                │   102 imágenes · 161 detecciones con texto                 │
                └──────────────────────────────────────────────────────────┘
```

Para el detalle de cada etapa y las decisiones de diseño, consulta [`ARCHITECTURE.md`](ARCHITECTURE.md).

-----

## Tecnologías utilizadas

|Categoría            |Tecnología                                                                                        |
|---------------------|--------------------------------------------------------------------------------------------------|
|Lenguaje             |Python 3.10+                                                                                      |
|Visión por computador|OpenCV (`cv2`)                                                                                    |
|Machine learning     |scikit-learn (LDA, PCA, KNN, SVM, NearestCentroid, GaussianNB) · OpenCV `ml.NormalBayesClassifier`|
|Características      |niveles de gris · HOG (scikit-image)                                                              |
|Cálculo numérico     |NumPy                                                                                             |
|Evaluación           |accuracy, precisión/recall/F1 macro, matriz de confusión, distancia de Levenshtein                |

-----

## Estructura del proyecto

```
.
├── ocr/                                   # Paquete OCR (diseño orientado a objetos)
│   ├── ocr_classifier.py                  # Clase base: mapeo carácter↔etiqueta, preprocesado, dataset
│   ├── char_segmentation.py               # Umbralizado adaptativo + findContours + boundingRect
│   ├── features.py                        # Vector de características (gris / HOG)
│   ├── lda_normal_bayes_classifier.py     # Sistema básico del Ejercicio 1
│   ├── classifiers.py                     # Sistema configurable del Ejercicio 2
│   ├── panel_reader.py                    # Detección de caracteres, RANSAC y lectura (Ejercicio 3)
│   └── data_loader.py                     # Carga de train_ocr / test_ocr
├── evaluar_clasificadores_OCR.py          # Ejercicios 1 y 2 (entrenar + validar, --classifier)
├── main_panels_ocr.py                     # Ejercicio 3 (lectura de paneles recortados)
├── evaluar_resultados_test_ocr_panels.py  # Evaluación por distancia de Levenshtein
├── main.py                                # Ejercicio 4 (detección MSER P1 + OCR P2)
├── detectors.py                           # Detector MSER reutilizado de la Práctica 1
├── resultado.txt                          # Salida generada (nombre;x1;y1;x2;y2;tipo;score;texto_ocr)
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── .gitignore
└── LICENSE
```

> **Nota sobre los datos:** los directorios de imágenes (`train_ocr`, `test_ocr`, `test_ocr_panels`) **no se incluyen** en el repositorio por su tamaño (el entrenamiento son 38 750 imágenes). Las rutas se pasan por parámetro al ejecutar los scripts.

-----

## Clasificadores disponibles

Seleccionables con el parámetro `--classifier`:

|Nombre              |Características|Reducción|Clasificador              |
|--------------------|---------------|---------|--------------------------|
|`lda_bayes` *(base)*|gris (625)     |LDA      |Normal Bayes (OpenCV)     |
|`lda_knn`           |gris (625)     |LDA      |KNN (k=3)                 |
|`lda_svm`           |gris (625)     |LDA      |SVM (RBF)                 |
|`lda_euclid`        |gris (625)     |LDA      |Euclídeo (NearestCentroid)|
|`pca_knn`           |gris (625)     |PCA      |KNN                       |
|`hog_lda_knn`       |HOG            |LDA      |KNN                       |
|`hog_lda_svm`       |HOG            |LDA      |SVM (RBF)                 |
|`hog_none_svm`      |HOG            |—        |SVM (RBF)                 |

-----

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/joel-domene/Practica2.git
cd Practica2

# 2. (Recomendado) Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```
opencv-python
numpy
scikit-learn
scikit-image
```

-----

## Uso

```bash
# Ejercicios 1 y 2 — entrenar y validar un clasificador
python evaluar_clasificadores_OCR.py --classifier lda_bayes \
    --train_path ./train_ocr --validation_path ./test_ocr

# Ejercicio 3 — lectura de paneles recortados (genera resultado.txt)
python main_panels_ocr.py --classifier lda_bayes \
    --train_path ./train_ocr --test_path ./test_ocr_panels
python evaluar_resultados_test_ocr_panels.py --panels_path ./test_ocr_panels

# Ejercicio 4 — sistema completo (detección MSER + OCR)
python main.py --detector mser --test_path <dir_test_practica1> \
    --ocr_classifier lda_bayes --ocr_train_path ./train_ocr \
    --visualize_ocr
```

-----

## Resultados

### Clasificador de caracteres (Ejercicio 1)

Entrenamiento: 625 imágenes/clase (38 750 en total). Validación: 15 500 imágenes.

|Métrica          |Valor|
|-----------------|-----|
|Accuracy         |0.960|
|Precisión (macro)|0.960|
|Recall (macro)   |0.960|
|F1 (macro)       |0.960|

La mayoría de clases alcanzan F1 ≈ 1.0. El error residual se concentra en ambigüedades intrínsecas de caracteres aislados sin contexto de palabra: `O`/`o`/`0`, `I`/`l`/`1` y mayúscula vs. minúscula de igual forma (`V`/`v`, `W`/`w`).

![Matriz de confusión del clasificador LDA + Normal Bayes sobre test_ocr](docs/matriz-confusion.png)

*Matriz de confusión (lda_bayes, acc=0.961). La diagonal dominante refleja el alto acierto; los focos fuera de la diagonal corresponden a las confusiones de caso descritas.*

### Comparativa de clasificadores (Ejercicio 2)

|Clasificador          |Características|Reducción|Accuracy |
|----------------------|---------------|---------|---------|
|**lda_bayes** *(base)*|gris           |LDA      |**0.960**|
|lda_knn               |gris           |LDA      |0.959    |
|hog_lda_svm           |HOG            |LDA      |0.958    |
|lda_euclid            |gris           |LDA      |0.923    |

LDA + Normal Bayes y LDA + KNN son prácticamente equivalentes y los mejores. El clasificador euclídeo es el peor porque asume clases esféricas e isótropas, hipótesis que no se cumple tras la proyección LDA. HOG no mejora a los niveles de gris: los caracteres ya están centrados y normalizados, por lo que el gradiente no aporta discriminación adicional.

### Lectura de paneles recortados (Ejercicio 3)

74 paneles evaluados con distancia de Levenshtein contra anotaciones manuales:

|Métrica        |Valor      |
|---------------|-----------|
|Distancia media|10.1       |
|Distancia ≤ 5  |27/74 (36%)|
|Distancia ≤ 10 |40/74 (54%)|

![Histograma de la distancia de Levenshtein sobre 74 paneles](docs/histograma-levenshtein.png)

*Distribución del error de edición entre el texto reconocido y el real.*

![Detección de caracteres y agrupación en líneas con RANSAC](docs/deteccion-caracteres-ransac.png)

*Detección de caracteres (verde) y líneas de texto agrupadas por RANSAC en un panel recortado.*

### Sistema completo (Ejercicio 4)

Integración del detector MSER (P1) con el lector OCR (P2) sobre 102 imágenes de carretera (161 detecciones con texto). Ejemplos reales de lectura: `'TÚNEL DE LA ESCRITA'` → `'TUNELDE+LAEscRjTA'`; `'500m'` → `'5oom'`.

![Sistema completo: detección de paneles y lectura de su contenido](docs/sistema-completo.png)

*Panel detectado (rojo) con su score, caracteres (verde), líneas de texto (cian) y texto OCR reconocido.*

-----

## Decisiones técnicas relevantes

- **Mismo preprocesado en entrenamiento y en paneles reales:** el clasificador ve siempre el mismo tipo de entrada (carácter binarizado, blanco sobre negro, 25×25), lo que mantiene la coherencia entre el dominio sintético de entrenamiento y el texto real.
- **Ajuste automático de polaridad:** tras umbralizar se inspecciona el borde de la imagen (casi siempre fondo); si queda blanco, se invierte. Esto unifica el texto oscuro sobre claro del entrenamiento con el texto claro sobre azul de los paneles.
- **LDA frente a PCA:** LDA es supervisado y maximiza la separación entre clases, mientras que PCA solo conserva varianza; con 62 clases bien definidas, LDA da mejor reducción (625 → 61) para clasificación.
- **Otsu + CLAHE en la lectura de paneles:** se eligió Otsu frente al umbralizado adaptativo porque conserva trazos sólidos y la topología de los caracteres (los huecos de `0`, `A`, `e`), manteniendo la correspondencia con el entrenamiento.
- **RANSAC para las líneas de texto:** robusto frente a caracteres mal segmentados; encuentra una línea, separa los inliers e itera, ordenando después de arriba a abajo y de izquierda a derecha (orden de lectura).
- **Separación de caracteres pegados:** las cajas anchas se dividen según los valles de la proyección vertical, con el ancho de carácter auto-estimado a partir de los glifos ya aislados.

-----

## Aprendizajes obtenidos

- Construcción de un **pipeline de machine learning completo** (características → reducción → clasificación → evaluación) con scikit-learn y OpenCV.
- **Evaluación rigurosa**: accuracy, precisión/recall/F1 macro, matriz de confusión y distancia de edición, con discusión razonada de los errores.
- **Comparación sistemática de modelos** y comprensión de por qué unos funcionan mejor que otros (supuestos del clasificador euclídeo, utilidad real de HOG según los datos).
- **Diseño orientado a objetos** con una jerarquía de clasificadores extensible y configurable.
- **Integración de sistemas**: reutilización del detector de la Práctica 1 para formar un sistema de detección + lectura de extremo a extremo.

-----

## Limitaciones conocidas

- **Ambigüedad de caracteres aislados** sin contexto de palabra (`O`/`o`/`0`, `I`/`l`/`1`), inevitable a nivel de carácter.
- **Caracteres pegados** que la proyección vertical no siempre separa con precisión.
- **Diferencia de dominio** entre los caracteres sintéticos de entrenamiento y el texto real de los paneles (baja resolución, perspectiva), que penaliza la lectura completa.

-----

## Mejoras futuras

- **Modelo de lenguaje / corrección posterior** para resolver las ambigüedades de caso usando contexto (diccionario de topónimos, validación de patrones).
- **Aprendizaje profundo**: un OCR basado en redes (CRNN, Transformers de texto) entrenado con datos reales superaría al pipeline clásico.
- **Data augmentation realista** que acerque el dominio de entrenamiento al de los paneles reales.

-----

## Contexto académico

- **Asignatura:** Visión Artificial — Grado en Ingeniería de Computadores, Universidad Rey Juan Carlos (URJC).
- **Curso:** 2025/26 — Práctica 2 grupal (continúa la Práctica 1).
- **Peso:** 20% de la asignatura.

-----

## Contribuidores

Proyecto desarrollado en equipo, con contribución conjunta de los tres autores:

- **Nicolás Wenceslao Muñoz Ciudad**
- **Jorge Bernabé Molinero**
- **Joel Domené Álvaro** · [GitHub](https://github.com/joel-domene)

-----

## Licencia

Distribuido bajo licencia MIT. Consulta [`LICENSE`](LICENSE) para más información.