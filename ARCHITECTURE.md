# Arquitectura del Sistema

Este documento describe en detalle el pipeline de OCR de paneles de autopista, organizado en los cuatro ejercicios encadenados de la práctica. El sistema sigue un **diseño orientado a objetos** centrado en el paquete `ocr/`, con una clase base `OCRClassifier` y clasificadores derivados.

-----

## 1. Organización del código

```
ocr/
├── ocr_classifier.py              # Clase base: mapeo carácter↔etiqueta, preprocesado común, build_dataset
├── char_segmentation.py           # binarize(): umbralizado adaptativo + ajuste de polaridad
├── features.py                    # image_to_feature(): vector de gris (625) o HOG
├── lda_normal_bayes_classifier.py # LdaNormalBayesClassifier (Ejercicio 1)
├── classifiers.py                 # GenericOCRClassifier + registro AVAILABLE_CLASSIFIERS (Ejercicio 2)
├── panel_reader.py                # PanelReader: detect_chars, RANSAC, split_wide_box (Ejercicio 3)
└── data_loader.py                 # Carga de train_ocr / test_ocr
```

Scripts de nivel superior: `evaluar_clasificadores_OCR.py` (Ej. 1 y 2), `main_panels_ocr.py` y `evaluar_resultados_test_ocr_panels.py` (Ej. 3), `main.py` + `detectors.py` (Ej. 4).

-----

## 2. Ejercicio 1 — Clasificador de caracteres

Pipeline idéntico en entrenamiento y en inferencia, de modo que el clasificador ve siempre el mismo tipo de entrada:

1. **Conversión a gris** (`cv2.cvtColor`).
1. **Umbralizado adaptativo** (`cv2.adaptiveThreshold`, `ADAPTIVE_THRESH_GAUSSIAN_C`, `THRESH_BINARY_INV`). La **polaridad se ajusta automáticamente** mirando el borde de la imagen (casi siempre fondo): si queda blanco, se invierte. Así se unifica el texto oscuro sobre claro del entrenamiento con el texto claro sobre azul de los paneles.
1. **Localización del carácter** (`cv2.findContours` + `cv2.contourArea` + `cv2.boundingRect`): se toma el mayor contorno.
1. **Recorte y redimensionado a 25×25** (`cv2.resize`).
1. **Vector de características**: la matriz 25×25 de gris se aplana a una fila de **625** columnas (`np.reshape`).
1. **Reducción de dimensionalidad con LDA** (`sklearn ... LinearDiscriminantAnalysis`, `fit` + `transform`). Con 62 clases, LDA reduce **625 → 61**.
1. **Clasificación** con `cv2.ml.NormalBayesClassifier` (bayesiano gaussiano). Requiere `np.float32` para las matrices y `np.int32` para las etiquetas.

**Clases (62):** `0-9` (dígitos) + `a-z` (minúsculas) + `A-Z` (mayúsculas). Las imágenes de entrenamiento se generaron a partir de la fuente *true type* `highway-gothic` con transformaciones geométricas y ruido añadido.

-----

## 3. Ejercicio 2 — Sistema configurable y comparativa

`GenericOCRClassifier` permite intercambiar los tres componentes del sistema:

- **Características:** `gray` (625 niveles de gris) o `hog` (descriptor HOG de scikit-image, 9 orientaciones, celdas 5×5).
- **Reducción:** `lda`, `pca` (con *whitening*) o `none`.
- **Clasificador:** `knn` (k=3), `svm` (RBF, C=10), `euclid` (`NearestCentroid`) o `bayes` (`GaussianNB`).

Las combinaciones se registran en `AVAILABLE_CLASSIFIERS` y se eligen con `--classifier`.

### Resultados (validación sobre test_ocr)

|Clasificador    |Accuracy|
|----------------|--------|
|lda_bayes (base)|0.960   |
|lda_knn         |0.959   |
|hog_lda_svm     |0.958   |
|lda_euclid      |0.923   |

**Discusión:** con un conjunto de entrenamiento grande y limpio casi todos los sistemas superan el 95%. El euclídeo es el peor porque asume clases esféricas e isótropas, hipótesis que no se cumple tras la proyección LDA. HOG no mejora a los niveles de gris porque los caracteres ya están centrados y normalizados. Se mantiene **LDA + Normal Bayes** como mejor sistema para los Ejercicios 3 y 4.

-----

## 4. Ejercicio 3 — Lectura de paneles recortados

Implementado en `ocr/panel_reader.py`.

### 4.1. Detección de caracteres (`detect_chars`)

- **Realce de contraste** con CLAHE (`cv2.createCLAHE`, 2.0, 8×8).
- **Umbralizado de Otsu** (`cv2.threshold` + `THRESH_OTSU`), con inversión si el texto queda como mayoría. Se eligió Otsu frente al adaptativo porque **conserva la topología** de los caracteres (huecos de `0`, `A`, `e`).
- **Apertura morfológica mínima** (elipse 2×2) para quitar motas sin tapar huecos.
- **Componentes conexas** (`cv2.connectedComponentsWithStats`) y **filtrado** por altura, anchura, área, relación de aspecto y solidez; se descartan las componentes pegadas al borde (marco, flechas).

### 4.2. Líneas de texto (RANSAC propio)

RANSAC sobre los centros de los caracteres para ajustar rectas `y = m·x + b`: se toman los inliers como una línea y se itera con el resto. Las líneas se ordenan de arriba a abajo y, dentro de cada una, de izquierda a derecha (orden de lectura).

### 4.3. Separación de caracteres pegados (`split_wide_box`)

Tras umbralizar, una palabra puede quedar como una sola componente. Las cajas anchas se dividen según los **valles de la proyección vertical**; el número de cortes se estima dividiendo el ancho de la caja entre el ancho de carácter auto-estimado.

### 4.4. Salida

Un único string por panel, sin espacios y con `+` como separador de línea, volcado a `resultado.txt` con el formato `<fichero>;0;0;<W>;<H>;1;1;<texto_ocr>`.

### 4.5. Evaluación

Distancia de Levenshtein contra `gt.txt`: 74 paneles, distancia media 10.1; ≤5 en el 36%; ≤10 en el 54%.

-----

## 5. Ejercicio 4 — Sistema completo (detección P1 + OCR P2)

`main.py` une el detector MSER de la Práctica 1 (`MSERBluePanelDetector`, en `detectors.py`) con el lector OCR. Por cada imagen de carretera, el detector devuelve las cajas de panel; cada panel se recorta y se pasa a `PanelReader.read_annotated()`, reutilizando íntegramente el pipeline del Ejercicio 3. El texto reconocido se escribe como octavo campo de `resultado.txt`:

```
nombre;x1;y1;x2;y2;1;score;texto_ocr
```

El parámetro `--visualize_ocr` (por defecto `False`) muestra la imagen procesada, los paneles con su score (rojo/amarillo), la caja de cada carácter (verde), las líneas de texto (cian) y el carácter reconocido (magenta).

Resultado sobre el test de la Práctica 1: 102 imágenes, 161 detecciones con texto OCR.

-----

## 6. Métodos y librerías utilizados

- **OpenCV (`cv2`):** `cvtColor`, `adaptiveThreshold`, `findContours`, `contourArea`, `boundingRect`, `connectedComponentsWithStats`, `createCLAHE`, `threshold`/Otsu, `morphologyEx`, `getStructuringElement`, `bitwise_not`, `ml.NormalBayesClassifier`.
- **scikit-learn:** `LinearDiscriminantAnalysis`, `PCA`, `KNeighborsClassifier`, `SVC`, `NearestCentroid`, `GaussianNB`.
- **scikit-image:** `feature.hog`.
- **NumPy:** manejo de matrices, conversiones de tipo y proyección vertical.

-----

## 7. Mejoras propias sobre el enunciado

- Ajuste automático de polaridad en el umbralizado.
- Uso de Otsu + CLAHE en la lectura para conservar la topología del carácter.
- Separación de caracteres pegados por proyección vertical con ancho auto-estimado.
- Filtrado de componentes por contacto con el borde y por outliers de altura dentro de cada línea.