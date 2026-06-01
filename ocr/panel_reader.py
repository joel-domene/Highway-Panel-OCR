"""
Lectura automática de paneles recortados.

Primero detecta los caracteres del panel usando umbralizado y contornos.
Después los agrupa en líneas de texto y los ordena para leerlos de
izquierda a derecha y de arriba a abajo.

Cada carácter se clasifica y se genera el texto final del panel.
"""

import numpy as np

import cv2

from .char_segmentation import to_gray, binarize


class PanelReader:

    def __init__(self, classifier, min_h_ratio=0.04, max_h_ratio=0.6):
        self.clf = classifier
        self.min_h_ratio = min_h_ratio
        self.max_h_ratio = max_h_ratio

    #1. Deteccion de caracteres candidatos
    def detect_chars(self, img):
        """
        Devuelve (boxes, mask) donde mask es la mascara binaria del panel
        (caracter=255, fondo=0) y boxes la lista de cajas de caracteres
        candidatos obtenidas con componentes conexas.
        """
        gray = to_gray(img)
        H, W = gray.shape

        #Realce de contraste
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

        #Umbralizado de Otsu
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #El texto siempre es minoritario en un panel -> que sea el frente.
        if mask.mean() > 127:
            mask = cv2.bitwise_not(mask)

        #Apertura minima para eliminar motas de ruido (sin tapar huecos).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        boxes = []
        min_h = self.min_h_ratio * H
        max_h = self.max_h_ratio * H
        panel_area = float(H * W)
        for i in range(1, n):           
            x, y, w, h, area = stats[i]
            if h < min_h or h > max_h:
                continue
            if w < 2 or w > 0.5 * W:
                continue
            if (w * h) > 0.18 * panel_area:   #Descarta marco
                continue
            if area < 12:                     #Descarta motas de ruido
                continue
            #Descarta componentes pegadas al borde (marco, flechas, exterior)
            if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                continue
            ar = w / float(h)
            if ar < 0.05 or ar > 2.4:         #Relacion de aspecto de caracter
                continue
            fill = area / float(w * h + 1e-6)
            if fill < 0.10 or fill > 0.98:    #Rescarta lineas/ruido y bloques
                continue
            boxes.append((x, y, w, h))

        return boxes, mask

    #2. Agrupacion en lineas de texto con RANSAC
    @staticmethod
    def _ransac_line(points, thr, iters=200):
        #RANSAC para una recta y = m*x + b. Devuelve mascara de inliers.
        best_inliers = np.zeros(len(points), dtype=bool)
        n = len(points)
        if n < 2:
            return np.ones(n, dtype=bool) if n == 1 else best_inliers
        xs = points[:, 0]
        ys = points[:, 1]
        rng = np.random.default_rng(42)
        for _ in range(iters):
            i, j = rng.choice(n, 2, replace=False)
            if xs[j] == xs[i]:
                continue
            m = (ys[j] - ys[i]) / (xs[j] - xs[i])
            b = ys[i] - m * xs[i]
            dist = np.abs(m * xs - ys + b) / np.sqrt(m * m + 1.0)
            inliers = dist < thr
            if inliers.sum() > best_inliers.sum():
                best_inliers = inliers
        return best_inliers

    @staticmethod
    def split_wide_box(binary, box, char_w):
        """
        En los paneles los caracteres de una palabra suelen quedar pegados
        tras umbralizar (una sola componente conexa por palabra). Si la caja
        es mucho mas ancha que un caracter, se divide usando la proyeccion
        vertical (valles = separaciones entre caracteres).
        """
        x, y, w, h = box
        char_w = max(4.0, char_w)
        n = int(round(w / char_w))
        if n <= 1:
            return [box]

        region = binary[y:y + h, x:x + w]
        col = (region > 0).sum(axis=0).astype(np.float32)

        cuts = [0]
        for k in range(1, n):
            center = int(k * w / n)
            win = max(2, int(0.35 * w / n))
            a = max(cuts[-1] + 2, center - win)
            b = min(w - 2, center + win)
            if b <= a:
                cut = center
            else:
                cut = a + int(np.argmin(col[a:b]))
            cuts.append(cut)
        cuts.append(w)

        sub = []
        for i in range(len(cuts) - 1):
            sx, ex = cuts[i], cuts[i + 1]
            if ex - sx < 2:
                continue
            sub.append((x + sx, y, ex - sx, h))
        return sub if sub else [box]

    def group_lines(self, boxes):
        #Devuelve una lista de lineas. Cada linea es una lista de boxes
        if not boxes:
            return []
        centers = np.array(
            [[x + w / 2.0, y + h / 2.0] for (x, y, w, h) in boxes])
        heights = np.array([h for (_, _, _, h) in boxes])
        thr = max(4.0, 0.5 * np.median(heights))

        remaining = list(range(len(boxes)))
        lines = []
        while len(remaining) >= 1:
            pts = centers[remaining]
            if len(remaining) == 1:
                lines.append([boxes[remaining[0]]])
                break
            mask = self._ransac_line(pts, thr)
            if mask.sum() < 2:
                for idx in remaining:
                    lines.append([boxes[idx]])
                break
            line_idx = [remaining[k] for k in range(len(remaining))
                        if mask[k]]
            line_boxes = sorted((boxes[i] for i in line_idx),
                                key=lambda b: b[0])
            """      
            En una linea los caracteres tienen alturas similares: se
            descartan los outliers (restos de marco, acentos, ruido)
            """
            med_h = np.median([b[3] for b in line_boxes])
            line_boxes = [b for b in line_boxes
                          if 0.45 * med_h <= b[3] <= 1.9 * med_h]
            if line_boxes:
                lines.append(line_boxes)
            remaining = [remaining[k] for k in range(len(remaining))
                         if not mask[k]]

        #Ordenar lineas de arriba a abajo por la 'y' media
        lines.sort(key=lambda ln: np.mean([b[1] + b[3] / 2.0 for b in ln]))
        return lines

    #3. Lectura completa del panel
    def read(self, img):
        """
        Devuelve (texto, lines) donde texto es el string del panel
        y lines la estructura geometrica
        """
        text, annotated = self.read_annotated(img)
        lines = [[(x, y, w, h) for (x, y, w, h, _ch) in ln]
                 for ln in annotated]
        return text, lines

    def read_annotated(self, img):
        #Igual que read() pero devuelve donde cada linea es una lista de tuplas
        boxes, binary = self.detect_chars(img)
        lines = self.group_lines(boxes)

        #Separar caracteres pegados dentro de cada linea de texto
        split_lines = []
        for line in lines:
            line_h = float(np.median([b[3] for b in line]))
            #Ancho de caracter estimado con las cajas que ya parecen un unico caracter
            singles = [b[2] for b in line if 0.25 <= b[2] / b[3] <= 1.05]
            char_w = float(np.median(singles)) if singles else 0.55 * line_h
            new_line = []
            for box in line:
                subs = self.split_wide_box(binary, box, char_w)
                #Descarta restos demasiado estrechos tras la division
                subs = [s for s in subs if s[2] >= 0.18 * line_h]
                new_line.extend(subs if subs else [box])
            new_line.sort(key=lambda b: b[0])
            split_lines.append(new_line)
        lines = split_lines

        out_lines = []
        annotated_lines = []
        for line in lines:
            chars = []
            ann = []
            for (x, y, w, h) in line:
                pad = max(2, int(0.15 * h))
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(binary.shape[1], x + w + pad)
                y1 = min(binary.shape[0], y + h + pad)
                roi = binary[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                label = self.clf.predict(roi, already_binary=True)
                try:
                    ch = self.clf.label2char(label)
                except Exception:
                    ch = '?'
                chars.append(ch)
                ann.append((x, y, w, h, ch))
            if chars:
                out_lines.append("".join(chars))
                annotated_lines.append(ann)

        text = "+".join(out_lines)
        return text, annotated_lines
