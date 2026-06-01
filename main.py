"""
Este programa junta el detector de paneles de la práctica anterior
con el OCR para leer el texto de los paneles detectados.

Se puede usar solo para detectar paneles o también para aplicar OCR.
El resultado se guarda en resultado.txt y las imágenes con las
detecciones se guardan en resultado_imgs/.

Si se activa --visualize_ocr, se muestran los paneles detectados,
los caracteres encontrados y el texto reconocido.
"""

import argparse
import os

import cv2
import numpy as np

from detectors import build_detector
from ocr import build_classifier
from ocr.data_loader import load_ocr_dataset
from ocr.panel_reader import PanelReader


RESULT_IMAGES_DIR = "resultado_imgs"
RESULT_FILE = "resultado.txt"


def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def list_test_images(test_path):
    files = [f for f in os.listdir(test_path) if f.lower().endswith(".png")]
    files.sort()
    return files


def draw(img, detections, panel_annotations):
    """
    Dibuja en la imagen los paneles detectados, sus caracteres y el texto OCR.

    También muestra el score de cada panel y las líneas de texto encontradas.
    Las anotaciones de los caracteres vienen en coordenadas relativas al panel.
    """
    out = img.copy()
    for (x1, y1, x2, y2, score), (text, ann_lines) in zip(
            detections, panel_annotations):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, "%.2f" % score, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for line in ann_lines:
            centers = []
            for (cx, cy, cw, ch, character) in line:
                gx, gy = x1 + cx, y1 + cy
                cv2.rectangle(out, (gx, gy), (gx + cw, gy + ch),
                              (0, 255, 0), 1)
                cv2.putText(out, character, (gx, gy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                centers.append((gx + cw // 2, gy + ch // 2))
            for i in range(1, len(centers)):
                cv2.line(out, centers[i - 1], centers[i], (255, 255, 0), 1)
        if text:
            cv2.putText(out, text, (x1, min(out.shape[0] - 5, y2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return out


def run(train_path, test_path, detector_name,
        ocr_classifier_name, ocr_train_path, visualize_ocr):

    detector = build_detector(detector_name)

    if train_path and os.path.isdir(train_path):
        print("[info] train_path = %s (el detector no aprende parametros)"
              % train_path)
    if not os.path.isdir(test_path):
        raise SystemExit("[error] test_path no existe: %s" % test_path)

    # Entrenar el lector OCR (mejor sistema del Ejercicio 2)
    print("[ocr] Entrenando clasificador '%s'..." % ocr_classifier_name)
    clf = build_classifier(ocr_classifier_name)
    clf.train(load_ocr_dataset(ocr_train_path))
    reader = PanelReader(clf)
    print("[ocr] Clasificador entrenado.")

    os.makedirs(RESULT_IMAGES_DIR, exist_ok=True)
    image_files = list_test_images(test_path)
    print("[info] %d imagenes de test, detector=%s"
          % (len(image_files), detector_name))

    stop = False
    with open(RESULT_FILE, "w", encoding="utf-8") as fout:
        for i, name in enumerate(image_files, 1):
            img = imread_unicode(os.path.join(test_path, name))
            if img is None:
                continue

            detections = detector.detect(img)
            annotations = []
            for (x1, y1, x2, y2, score) in detections:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    annotations.append(("", []))
                    text = ""
                else:
                    text, ann = reader.read_annotated(crop)
                    annotations.append((text, ann))
                fout.write("%s;%d;%d;%d;%d;1;%.4f;%s\n"
                           % (name, x1, y1, x2, y2, score, text))

            out_img = draw(img, detections, annotations)
            cv2.imwrite(os.path.join(RESULT_IMAGES_DIR, name), out_img)

            if visualize_ocr and not stop:
                cv2.imshow("Practica 2 - Ej.4: deteccion + OCR", out_img)
                key = cv2.waitKey(0) & 0xFF
                if key == 27:          # ESC -> terminar visualizacion
                    stop = True
                    cv2.destroyAllWindows()

            if i % 10 == 0 or i == len(image_files):
                print("  procesadas %d/%d (ultima: %s, %d paneles)"
                      % (i, len(image_files), name, len(detections)))

    if visualize_ocr:
        cv2.destroyAllWindows()
    print("[ok] resultados en %s y %s/" % (RESULT_FILE, RESULT_IMAGES_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deteccion de paneles (Practica 1) + lectura OCR "
                    "(Practica 2) - Ejercicio 4")
    parser.add_argument("--detector", type=str, nargs="?", default="mser",
                        help="Detector: mser | hough")
    parser.add_argument("--train_path", default="",
                        help="Directorio de entrenamiento del detector")
    parser.add_argument("--test_path", default="",
                        help="Directorio de imagenes de test (Practica 1)")
    parser.add_argument("--ocr_classifier", default="lda_bayes",
                        help="Clasificador OCR (ver evaluar_clasificadores)")
    parser.add_argument("--ocr_train_path", default="./train_ocr",
                        help="Directorio train_ocr de la Practica 2")
    parser.add_argument("--visualize_ocr", action="store_true",
                        default=False,
                        help="Visualizar deteccion + OCR panel a panel")

    args = parser.parse_args()
    run(args.train_path, args.test_path, args.detector,
        args.ocr_classifier, args.ocr_train_path, args.visualize_ocr)
