"""
Script de evaluación de clasificadores OCR.

Sirve para entrenar y validar distintos clasificadores de caracteres
usando las carpetas de entrenamiento y validación.

Se puede elegir el clasificador con el parámetro --classifier.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from ocr import build_classifier, AVAILABLE_CLASSIFIERS
from ocr.data_loader import load_ocr_dataset


def plot_confusion_matrix(cm, classes, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Dada una matriz de confusion cm (np.array) la dibuja.
    """
    plt.figure(figsize=(14, 14))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over '
                    'testing images')
    parser.add_argument(
        '--classifier', type=str, default="lda_bayes",
        help='Classifier string name. Opciones: %s'
             % ", ".join(sorted(AVAILABLE_CLASSIFIERS)))
    parser.add_argument(
        '--train_path', default="./train_ocr",
        help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./test_ocr",
        help='Select the validation data dir')

    args = parser.parse_args()

    #1. Cargar las imagenes de entrenamiento y sus etiquetas.
    print("Cargando entrenamiento desde:", args.train_path)
    train_dict = load_ocr_dataset(args.train_path)
    n_train = sum(len(v) for v in train_dict.values())
    print("  %d clases, %d imagenes" % (len(train_dict), n_train))

    #2. Cargar datos de validacion y sus etiquetas
    print("Cargando validacion desde:", args.validation_path)
    val_dict = load_ocr_dataset(args.validation_path)
    n_val = sum(len(v) for v in val_dict.values())
    print("  %d clases, %d imagenes" % (len(val_dict), n_val))

    #3. Entrenar clasificador
    print("Clasificador:", args.classifier)
    clf = build_classifier(args.classifier)
    t0 = time.time()
    clf.train(train_dict)
    print("Entrenado en %.2f s" % (time.time() - t0))

    #4. Ejecutar el clasificador sobre los datos de validacion
    gt_labels = clf.get_labels_dict(val_dict)
    t0 = time.time()
    predicted_labels = clf.predict_dict(val_dict)
    print("Inferencia en %.2f s" % (time.time() - t0))

    #5. Evaluar los resultados
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    print("\n================ RESULTADOS ================")
    print("Accuracy = ", accuracy)

    #Medidas adicionales (precision / recall / F1 macro)
    p, r, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        gt_labels, predicted_labels, average='macro', zero_division=0)
    print("Precision (macro) = %.4f" % p)
    print("Recall    (macro) = %.4f" % r)
    print("F1-score  (macro) = %.4f" % f1)

    labels_present = sorted(set(gt_labels) | set(predicted_labels))
    class_names = [clf.label2char(l) for l in labels_present]
    print("\nClassification report:")
    print(sklearn.metrics.classification_report(
        gt_labels, predicted_labels,
        labels=labels_present, target_names=class_names,
        zero_division=0))

    cm = sklearn.metrics.confusion_matrix(
        gt_labels, predicted_labels, labels=labels_present)
    plot_confusion_matrix(
        cm, class_names,
        title='Matriz de confusion - %s (acc=%.3f)'
              % (args.classifier, accuracy))
    out_png = "confusion_%s.png" % args.classifier
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print("Matriz de confusion guardada en", out_png)
    plt.show()
