"""
Extrae los vectores de características de los caracteres.

En la versión básica se usa la imagen en escala de grises como vector.
También se puede usar HOG como alternativa para comparar resultados.
"""

import numpy as np
from skimage.feature import hog


def extract_features(norm_char, mode="gray"):
    """
    Crea el vector de características de un carácter ya preparado.
    Puede usar la imagen en gris directamente o calcular características HOG.
    Devuelve el vector listo para usar en el clasificador.
    """
    if mode == "gray":
        #Convertimos la matriz de niveles de gris en una unica fila.
        return norm_char.astype(np.float32).reshape(1, -1)

    if mode == "hog":
        feat = hog(
            norm_char.astype(np.float32) / 255.0,
            orientations=9,
            pixels_per_cell=(5, 5),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True)
        return feat.astype(np.float32).reshape(1, -1)

    raise ValueError("Modo de caracteristicas desconocido: %s" % mode)
