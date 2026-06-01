"""
Carga las imágenes de caracteres usadas para entrenar o probar el OCR.

Las imágenes están organizadas por carpetas: números, letras mayúsculas
y letras minúsculas.
"""

import glob
import os

import cv2


def _load_dir(path):
    imgs = []
    for f in sorted(glob.glob(os.path.join(path, "*.png"))):
        im = cv2.imread(f)
        if im is not None:
            imgs.append(im)
    return imgs


def load_ocr_dataset(root):
    """
    Devuelve un diccionario {caracter: [imagenes]}
    """
    data = {}

    #Digitos 0-9
    for d in "0123456789":
        p = os.path.join(root, d)
        if os.path.isdir(p):
            data[d] = _load_dir(p)

    #Mayusculas
    may = os.path.join(root, "may")
    if os.path.isdir(may):
        for name in sorted(os.listdir(may)):
            sub = os.path.join(may, name)
            if os.path.isdir(sub):
                data[name.upper()] = _load_dir(sub)

    #Minusculas
    mn = os.path.join(root, "min")
    if os.path.isdir(mn):
        for name in sorted(os.listdir(mn)):
            sub = os.path.join(mn, name)
            if os.path.isdir(sub):
                data[name.lower()] = _load_dir(sub)

    #Eliminamos clases vacias
    return {k: v for k, v in data.items() if len(v) > 0}
