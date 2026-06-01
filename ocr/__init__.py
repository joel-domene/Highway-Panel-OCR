#Paquete OCR
from .ocr_classifier import OCRClassifier
from .lda_normal_bayes_classifier import LdaNormalBayesClassifier
from .classifiers import build_classifier, AVAILABLE_CLASSIFIERS

__all__ = [
    "OCRClassifier",
    "LdaNormalBayesClassifier",
    "build_classifier",
    "AVAILABLE_CLASSIFIERS",
]
