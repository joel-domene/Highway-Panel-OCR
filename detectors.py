"""
Detectores de paneles azules.

Incluye un detector principal con MSER y máscara azul, y otro detector
opcional usando Hough para buscar bordes rectos.

También se usa Non Maximum Suppression para quitar detecciones repetidas.
"""

import os
import cv2
import numpy as np


# Tamanyo fijo al que se redimensionan las ventanas candidatas para
# correlarlas con la mascara ideal del panel (alto x ancho).
PANEL_SIZE = (40, 80)

# Rango de azul saturado en HSV que usamos en toda la practica.
# H esta en [0, 180] en OpenCV (no [0, 360]).
BLUE_HSV_LOWER = np.array([95, 110, 50], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)


def blue_mask_hsv(bgr_image):
    """Devuelve mascara binaria (0/1, uint8) con los pixeles azules muy saturados."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    return (mask > 0).astype(np.uint8)


def build_ideal_blue_mask(size=PANEL_SIZE, border=2):
    """Construye la mascara ideal de un panel: casi todo azul, dejando un borde a 0
    para tolerar el marco blanco que rodea la zona azul."""
    h, w = size
    ideal = np.zeros((h, w), dtype=np.float32)
    ideal[border:h - border, border:w - border] = 1.0
    return ideal


def expand_box(x, y, w, h, img_shape, ratio=0.10):
    """Expande la caja un pequenyo porcentaje en cada lado para incluir el borde
    blanco que MSER no detecta (se queda con la zona azul interna)."""
    H, W = img_shape[:2]
    dx = int(round(w * ratio))
    dy = int(round(h * ratio))
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(W - 1, x + w + dx)
    y2 = min(H - 1, y + h + dy)
    return x1, y1, x2, y2


def iou(a, b):
    """IoU entre dos cajas en formato (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / float(aa + bb - inter)


def non_maximum_suppression(detections, iou_thr=0.3):
    """
    Elimina detecciones repetidas.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou(best[:4], d[:4]) < iou_thr]
    return kept


# ---------------------------------------------------------------------------
# Detector 1: MSER + correlacion con mascara de azul (el del enunciado)
# ---------------------------------------------------------------------------
class MSERBluePanelDetector:
    """
    Detector de paneles azules usando MSER y una máscara HSV.

    Primero mejora la imagen, busca regiones candidatas con MSER y elimina
    las que no tienen forma de panel. Después comprueba si cada región tiene
    suficiente color azul comparándola con una máscara ideal.

    Si la coincidencia es buena se acepta la detección y se le asigna un score.
    """

    def __init__(self,
                 mser_delta=7,
                 mser_min_area=400,
                 mser_max_area=80000,
                 min_aspect=0.4,
                 max_aspect=4.0,
                 corr_threshold=0.45,
                 expand_ratio=0.10):
        self.mser = cv2.MSER_create(
            delta=mser_delta,
            min_area=mser_min_area,
            max_area=mser_max_area)
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.corr_threshold = corr_threshold
        self.expand_ratio = expand_ratio

        self.ideal_mask = build_ideal_blue_mask(PANEL_SIZE, border=2)
        self.ideal_sum = float(self.ideal_mask.sum())
        # Mascara complementaria (zona donde NO deberia haber azul, p.ej. el borde
        # blanco). La usamos para penalizar (precision).
        self.ideal_inv = 1.0 - self.ideal_mask
        self.ideal_inv_sum = float(self.ideal_inv.sum())

    # ------------------------------------------------------------------
    def _candidate_boxes(self, bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # Mejora de contraste local (CLAHE) para ayudar a MSER en zonas con sombra.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        regions, _ = self.mser.detectRegions(gray)
        boxes = []
        H, W = bgr_image.shape[:2]
        for pts in regions:
            x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
            if w < 15 or h < 10:
                continue
            ar = w / float(h)
            if ar < self.min_aspect or ar > self.max_aspect:
                continue
            x1, y1, x2, y2 = expand_box(x, y, w, h, (H, W), self.expand_ratio)
            boxes.append((x1, y1, x2, y2))
        return boxes

    # ------------------------------------------------------------------
    def _score_box(self, bgr_image, box):
        x1, y1, x2, y2 = box
        crop = bgr_image[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            return 0.0
        crop_resized = cv2.resize(crop, (PANEL_SIZE[1], PANEL_SIZE[0]),
                                  interpolation=cv2.INTER_AREA)
        m = blue_mask_hsv(crop_resized).astype(np.float32)

        # "Recall": proporcion de pixeles azules dentro de la zona ideal.
        hits = float((m * self.ideal_mask).sum())
        recall = hits / max(self.ideal_sum, 1.0)
        # "Precision": penaliza azul donde no deberia haber (zonas borde).
        misses = float((m * self.ideal_inv).sum())
        precision_penalty = misses / max(self.ideal_inv_sum, 1.0)

        # Score F-like: recall alto y poca azul fuera. Lo dejamos en [0, 1].
        score = recall * (1.0 - precision_penalty)
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    def detect(self, bgr_image):
        boxes = self._candidate_boxes(bgr_image)
        detections = []
        for box in boxes:
            score = self._score_box(bgr_image, box)
            if score >= self.corr_threshold:
                detections.append((*box, score))
        detections = non_maximum_suppression(detections, iou_thr=0.3)
        return detections


# ---------------------------------------------------------------------------
# Detector 2: Mascara de azul + Hough probabilistica + componentes conexas
# ---------------------------------------------------------------------------
class HoughBluePanelDetector:
    """
    Detector alternativo de paneles azules usando color y líneas.

    Primero busca zonas azules con una máscara HSV y las une con operaciones
    morfológicas. 
    Después obtiene posibles paneles y comprueba si tienen bordes
    rectos usando Canny y HoughLinesP.

    El score depende de cuánto azul hay dentro de la caja y de si la forma se
    parece a un rectángulo.
    """

    def __init__(self,
                 min_area=600,
                 min_aspect=0.4,
                 max_aspect=5.0,
                 hough_threshold=40,
                 min_line_length=20,
                 max_line_gap=8,
                 score_threshold=0.35):
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    def detect(self, bgr_image):
        H, W = bgr_image.shape[:2]
        blue = blue_mask_hsv(bgr_image) * 255

        # Cierre morfologico para unir las letras blancas internas con el azul.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Componentes conexas como candidatos.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        # Bordes para Hough.
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)

        detections = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            ar = w / float(h)
            if ar < self.min_aspect or ar > self.max_aspect:
                continue
            x1, y1, x2, y2 = expand_box(x, y, w, h, (H, W), 0.05)

            # Recortes para evaluar.
            roi_blue = blue[y1:y2 + 1, x1:x2 + 1]
            roi_edges = edges[y1:y2 + 1, x1:x2 + 1]
            if roi_blue.size == 0:
                continue

            blue_ratio = float(roi_blue.sum()) / (255.0 * roi_blue.size)
            rect_ratio = area / float((x2 - x1 + 1) * (y2 - y1 + 1))

            lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180,
                                    threshold=self.hough_threshold,
                                    minLineLength=self.min_line_length,
                                    maxLineGap=self.max_line_gap)
            n_lines = 0 if lines is None else len(lines)
            # Necesitamos al menos algun par de lineas (un panel rectangular tiene
            # bordes horizontales y verticales claros).
            line_score = min(1.0, n_lines / 8.0)

            score = 0.5 * blue_ratio + 0.3 * rect_ratio + 0.2 * line_score
            if score < self.score_threshold:
                continue
            detections.append((x1, y1, x2, y2, float(np.clip(score, 0.0, 1.0))))

        detections = non_maximum_suppression(detections, iou_thr=0.3)
        return detections


# ---------------------------------------------------------------------------
# Factoria
# ---------------------------------------------------------------------------
DETECTORS = {
    "mser": MSERBluePanelDetector,
    "hough": HoughBluePanelDetector,
}


def build_detector(name):
    name = (name or "mser").lower()
    if name not in DETECTORS:
        raise ValueError("Detector desconocido: %s. Disponibles: %s"
                         % (name, list(DETECTORS.keys())))
    return DETECTORS[name]()
