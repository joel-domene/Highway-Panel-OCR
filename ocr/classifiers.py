"""
Clasificadores alternativos del ejercicio 2.

Este módulo permite probar distintas combinaciones para el OCR:
características en gris o HOG, reducción con LDA, PCA o ninguna,
y clasificadores Bayes, KNN, SVM o Euclid.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

from .ocr_classifier import OCRClassifier
from .lda_normal_bayes_classifier import LdaNormalBayesClassifier


class GenericOCRClassifier(OCRClassifier):
    #OCR configurable (caracteristicas + reduccion + clasificador).

    def __init__(self, feature_mode="gray", reducer="lda", clf="knn",
                 ocr_char_size=(25, 25), pca_components=40, knn_k=3):
        super().__init__(ocr_char_size, feature_mode=feature_mode)
        self.reducer_name = reducer
        self.clf_name = clf
        self.classifier_name = "%s_%s_%s" % (feature_mode, reducer, clf)
        self.pca_components = pca_components
        self.knn_k = knn_k
        self.reducer = None
        self.classifier = None

    def _build_reducer(self):
        if self.reducer_name == "lda":
            return LinearDiscriminantAnalysis()
        if self.reducer_name == "pca":
            return PCA(n_components=self.pca_components, whiten=True)
        if self.reducer_name == "none":
            return None
        raise ValueError("Reductor desconocido: %s" % self.reducer_name)

    def _build_classifier(self):
        if self.clf_name == "knn":
            return KNeighborsClassifier(n_neighbors=self.knn_k)
        if self.clf_name == "svm":
            return SVC(kernel="rbf", C=10, gamma="scale")
        if self.clf_name == "euclid":
            return NearestCentroid()         #Clasificador euclideo
        if self.clf_name == "bayes":
            return GaussianNB()
        raise ValueError("Clasificador desconocido: %s" % self.clf_name)

    def train(self, images_dict):
        X, y = self.build_dataset(images_dict)

        self.reducer = self._build_reducer()
        if self.reducer is not None:
            if self.reducer_name == "lda":
                self.reducer.fit(X, y)
            else:
                self.reducer.fit(X)
            Xr = self.reducer.transform(X)
        else:
            Xr = X

        self.classifier = self._build_classifier()
        self.classifier.fit(Xr, y)
        return X, y

    def predict(self, img, already_binary=False):
        feat = self.image_to_feature(img, already_binary)
        if self.reducer is not None:
            feat = self.reducer.transform(feat)
        return int(self.classifier.predict(feat)[0])


#Registro de clasificadores disponibles
AVAILABLE_CLASSIFIERS = {
    #Sistema basico (Ejercicio 1)
    "lda_bayes":     lambda: LdaNormalBayesClassifier(),
    
    #Alternativas (Ejercicio 2)
    "lda_knn":       lambda: GenericOCRClassifier("gray", "lda", "knn"),
    "lda_svm":       lambda: GenericOCRClassifier("gray", "lda", "svm"),
    "lda_euclid":    lambda: GenericOCRClassifier("gray", "lda", "euclid"),
    "pca_knn":       lambda: GenericOCRClassifier("gray", "pca", "knn"),
    "hog_lda_knn":   lambda: GenericOCRClassifier("hog", "lda", "knn"),
    "hog_lda_svm":   lambda: GenericOCRClassifier("hog", "lda", "svm"),
    "hog_none_svm":  lambda: GenericOCRClassifier("hog", "none", "svm"),
}


def build_classifier(name):
    """Devuelve una instancia del clasificador a partir de su nombre."""
    if name not in AVAILABLE_CLASSIFIERS:
        raise ValueError(
            "Clasificador '%s' no disponible. Opciones: %s"
            % (name, ", ".join(sorted(AVAILABLE_CLASSIFIERS))))
    return AVAILABLE_CLASSIFIERS[name]()
