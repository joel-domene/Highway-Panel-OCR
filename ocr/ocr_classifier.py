"""
Clase base para los clasificadores OCR.

Contiene la estructura común del enunciado y algunas funciones compartidas,
como la segmentación del carácter y la extracción de características.
"""

import string

import numpy as np

from .char_segmentation import segment_char
from .features import extract_features


class OCRClassifier:
    #Clasificador para Reconocimiento Optico de Caracteres (clase base).

    def __init__(self, ocr_char_size=(25, 25), feature_mode="gray"):
        self.ocr_char_size = ocr_char_size
        self.feature_mode = feature_mode
        self.classifier_name = None

    #Mapeo caracter <-> etiqueta entera
    def char2label(self, c):
        all_chars = '0123456789' + string.ascii_letters
        return all_chars.find(c) + 1

    def label2char(self, label):
        all_chars = '0123456789' + string.ascii_letters
        return all_chars[label - 1]

    #Preprocesado comun: imagen -> vector de caracteristicas
    def image_to_feature(self, img, already_binary=False):
        #Segmenta el caracter y devuelve su vector de caracteristicas
        norm = segment_char(img, out_size=self.ocr_char_size,
                            already_binary=already_binary)
        return extract_features(norm, self.feature_mode)

    def build_dataset(self, images_dict):
        """
        Convierte las imágenes de cada carácter en datos para entrenar.
        Genera la matriz X con las características de cada imagen y el vector y
        con la etiqueta correspondiente de cada carácter.
        """
        feats, labels = [], []
        for key in images_dict:
            label = self.char2label(key)
            for img in images_dict[key]:
                feats.append(self.image_to_feature(img))
                labels.append(label)
        X = np.vstack(feats).astype(np.float32)
        y = np.array(labels, dtype=np.int32)
        return X, y

    #Utilidades de evaluacion
    def get_labels_dict(self, images_dict):
        responses = []
        for key in images_dict:
            for img in images_dict[key]:
                responses.append(self.char2label(key))
        return responses

    def predict_dict(self, images_dict):
        responses = []
        for key in images_dict:
            for img in images_dict[key]:
                responses.append(self.predict(img))
        return responses

    #Las clases derivadas deben implementar train() y predict()
    def train(self, images_dict):
        raise NotImplementedError

    def predict(self, img):
        raise NotImplementedError
