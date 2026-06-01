"""
Ejercicio 3

Lee automáticamente paneles ya recortados usando OCR. Primero entrena
el clasificador con las imágenes de train_ocr y después procesa los paneles
indicados en test_path.

Guarda el resultado en resultado.txt con las cajas detectadas y el texto leído.
"""

import argparse
import glob
import os

import cv2

from ocr import build_classifier
from ocr.data_loader import load_ocr_dataset
from ocr.panel_reader import PanelReader


def draw_results(img, lines, text):
    vis = img.copy()
    for line in lines:
        cxs = []
        for (x, y, w, h) in line:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cxs.append((x + w // 2, y + h // 2))
        for i in range(1, len(cxs)):
            cv2.line(vis, cxs[i - 1], cxs[i], (255, 255, 0), 1)
    cv2.putText(vis, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes the OCR over cropped road panels')
    parser.add_argument(
        '--classifier', type=str, nargs="?", default="lda_bayes",
        help='Classifier string name')
    parser.add_argument(
        '--train_path', default="./train_ocr",
        help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="./test_ocr_panels",
        help='Select the testing (panels) data dir')
    parser.add_argument(
        '--visualize_ocr', action='store_true',
        help='Muestra la deteccion/lectura de cada panel')

    args = parser.parse_args()

    # Carga los datos de entrenamiento y crea el clasificador OCR
    print("Entrenando clasificador '%s'..." % args.classifier)
    train_dict = load_ocr_dataset(args.train_path)
    clf = build_classifier(args.classifier)
    clf.train(train_dict)
    reader = PanelReader(clf)

    # Carga los datos de prueba (paneles recortados)
    panel_files = sorted(glob.glob(os.path.join(args.test_path, "*.png")))
    print("%d paneles a procesar" % len(panel_files))

    out_path = os.path.join(args.test_path, "resultado.txt")
    with open(out_path, "w", encoding="utf-8") as fout:
        for f in panel_files:
            img = cv2.imread(f)
            if img is None:
                continue
            name = os.path.basename(f)
            H, W = img.shape[:2]
            text, lines = reader.read(img)
            fout.write("%s;0;0;%d;%d;1;1;%s\n" % (name, W, H, text))
            print("%s -> %s" % (name, text))

            if args.visualize_ocr:
                vis = draw_results(img, lines, text)
                cv2.imshow("OCR panel", vis)
                if cv2.waitKey(0) & 0xFF == 27:
                    break

    if args.visualize_ocr:
        cv2.destroyAllWindows()
    print("\nResultados escritos en", out_path)
