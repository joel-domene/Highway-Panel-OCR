"""
Clasificador básico del ejercicio 1.

Usa la imagen del carácter en escala de grises como características,
reduce la dimensión con LDA y después clasifica con Normal Bayes.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import cv2

from .ocr_classifier import OCRClassifier


class LdaNormalBayesClassifier(OCRClassifier):
    #OCR mediante LDA + clasificador Bayesiano Gaussiano (Normal Bayes)

    def __init__(self, ocr_char_size=(25, 25)):
        super().__init__(ocr_char_size, feature_mode="gray")
        self.classifier_name = "lda_bayes"
        self.lda = None
        self.classifier = None

    def train(self, images_dict):
        """
        Recibe las imágenes de cada carácter y las convierte en datos de entrenamiento.
        Devuelve las muestras y sus etiquetas correspondientes.
        """
        #1. Caracteristicas (matriz C) y etiquetas (vector E)
        X, y = self.build_dataset(images_dict)          #C  y  E

        #2. LDA: fit + transform -> matriz reducida CR
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X, y)
        CR = self.lda.transform(X).astype(np.float32)   #CR

        #3. Clasificador Normal Bayes
        self.classifier = cv2.ml.NormalBayesClassifier_create()
        self.classifier.train(CR, cv2.ml.ROW_SAMPLE,
                              y.astype(np.int32))

        return X, y

    def predict(self, img, already_binary=False):
        #Clasifica una imagen de un caracter ya recortado
        feat = self.image_to_feature(img, already_binary).astype(np.float32)
        reduced = self.lda.transform(feat).astype(np.float32)
        _, resp = self.classifier.predict(reduced)
        return int(resp[0, 0])
