"""
Segmenta y normaliza los caracteres para el OCR.

Este preprocesado se usa tanto en las imágenes de entrenamiento como
en los caracteres recortados de los paneles reales, para que el clasificador
reciba siempre imágenes parecidas.
"""

import cv2
import numpy as np


def to_gray(img):
    #Devuelve la imagen en niveles de gris
    if img is None:
        raise ValueError("Imagen nula")
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def binarize(gray, block_size=21, c=10):
    """
    Aplica un umbral adaptativo para separar el carácter del fondo.

    El objetivo es dejar siempre el carácter en blanco y el fondo en negro,
    aunque la imagen original tenga los colores al revés.

    Para decidir si hay que invertir la imagen, se mira el color del borde,
    que normalmente pertenece al fondo.
    """
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c)

    h, w = binary.shape
    border = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if border.mean() > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def largest_char_bbox(binary):
    """
    Busca el carácter principal dentro de la imagen.

    Para ello usa los contornos y se queda con el más grande. Devuelve su caja
    o None si no encuentra ningún contorno válido.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 3:
        return None
    return cv2.boundingRect(c)


def segment_char(img, out_size=(25, 25), block_size=21, c=10,
                  already_binary=False):
    """
    Preprocesa una imagen que contiene un solo carácter.

    Convierte la imagen a gris, la umbraliza, recorta el carácter principal
    y lo redimensiona al tamaño necesario.

    Si la imagen ya es binaria, solo recorta y redimensiona sin volver a
    umbralizar.
    """
    if already_binary:
        binary = img if len(img.shape) == 2 else to_gray(img)
    else:
        gray = to_gray(img)
        binary = binarize(gray, block_size, c)

    bbox = largest_char_bbox(binary)
    if bbox is None:
        return cv2.resize(binary, out_size, interpolation=cv2.INTER_AREA)
    x, y, w, h = bbox
    crop = binary[y:y + h, x:x + w]
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
