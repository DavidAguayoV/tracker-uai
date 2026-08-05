"""Tracking automático (v2 movida a v1 por pedido docente).

Tres métodos, elegidos por robustez y por no exigir instalar paquetes
extra a los estudiantes:

  * "plantilla" — template matching (cv2.matchTemplate). El usuario marca
    un recuadro del objeto en el primer frame; se busca ese patrón en los
    frames siguientes (primero en una ventana alrededor de la última
    posición, y en todo el frame si la coincidencia es baja).
  * "color" — segmentación HSV + centroide del contorno más grande. Ideal
    para objetos de color distintivo (carro, plomada, trazador de látex).
  * "csrt" — tracker CSRT de OpenCV. Solo disponible si está instalado
    ``opencv-contrib-python``; si no, no se ofrece.

Todas las funciones devuelven una lista de tuplas
``(frame_idx, t, x_px, y_px)`` en píxeles de la imagen original.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from core.video import VideoReader

Point = Tuple[int, float, float, float]  # (frame_idx, t, x_px, y_px)
ProgressCb = Optional[Callable[[float], None]]
# Vista previa: recibe (frame_rgb, x_px, y_px, frame_idx).
PreviewCb = Optional[Callable[[np.ndarray, float, float, int], None]]


# ==========================================================================
# CSRT (opcional, requiere opencv-contrib-python)
# ==========================================================================
def csrt_available() -> bool:
    """Indica si hay un tracker CSRT disponible en esta instalación de OpenCV."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return True
    legacy = getattr(cv2, "legacy", None)
    return bool(legacy and hasattr(legacy, "TrackerCSRT_create"))


def _create_csrt():
    """Crea un tracker CSRT desde el módulo principal o el legacy."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


# ==========================================================================
# Utilidades de color
# ==========================================================================
def hsv_range_from_pixel(
    frame_rgb: np.ndarray,
    x: int,
    y: int,
    h_tol: int = 10,
    s_min: int = 60,
    v_min: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construye un rango HSV alrededor del color del píxel (x, y).

    Devuelve (lower, upper) como arreglos uint8 aptos para ``cv2.inRange``.
    OpenCV usa H ∈ [0,179], S,V ∈ [0,255].
    """
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[int(y), int(x)]
    lower = np.array([max(0, int(h) - h_tol), s_min, v_min], dtype=np.uint8)
    upper = np.array([min(179, int(h) + h_tol), 255, 255], dtype=np.uint8)
    return lower, upper


# ==========================================================================
# Métodos de seguimiento
# ==========================================================================
def track_template(
    reader: VideoReader,
    roi: Tuple[int, int, int, int],
    start: int,
    end: int,
    step: int = 1,
    fps_capture: Optional[float] = None,
    search_margin: float = 2.0,
    min_score: float = 0.4,
    progress_cb: ProgressCb = None,
    preview_cb: PreviewCb = None,
) -> List[Point]:
    """Sigue el objeto por coincidencia de plantilla.

    ``roi`` = (x, y, w, h) en píxeles originales del frame ``start``.
    """
    x, y, w, h = (int(round(v)) for v in roi)
    if w <= 0 or h <= 0:
        raise ValueError("El recuadro del objeto no es válido.")
    tmpl = cv2.cvtColor(reader.get_frame(start), cv2.COLOR_RGB2GRAY)[y:y + h, x:x + w]

    results: List[Point] = []
    last_cx, last_cy = x + w / 2.0, y + h / 2.0
    H, W = reader.height, reader.width

    idxs = list(range(start, end + 1, step))
    for i, idx in enumerate(idxs):
        frame_rgb = reader.get_frame(idx)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        # Ventana de búsqueda alrededor de la última posición.
        mx = int(search_margin * w)
        my = int(search_margin * h)
        x0 = max(0, int(last_cx - w / 2 - mx))
        y0 = max(0, int(last_cy - h / 2 - my))
        x1 = min(W, int(last_cx + w / 2 + mx))
        y1 = min(H, int(last_cy + h / 2 + my))
        window = gray[y0:y1, x0:x1]

        top_left = None
        if window.shape[0] >= h and window.shape[1] >= w:
            res = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval >= min_score:
                top_left = (x0 + maxloc[0], y0 + maxloc[1])
        if top_left is None:  # búsqueda en todo el frame
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, _, _, maxloc = cv2.minMaxLoc(res)
            top_left = maxloc

        cx = top_left[0] + w / 2.0
        cy = top_left[1] + h / 2.0
        last_cx, last_cy = cx, cy
        results.append((idx, reader.frame_time(idx, fps_capture), cx, cy))
        if preview_cb:
            preview_cb(frame_rgb, cx, cy, idx)
        if progress_cb:
            progress_cb((i + 1) / len(idxs))
    return results


def track_color(
    reader: VideoReader,
    lower: np.ndarray,
    upper: np.ndarray,
    start: int,
    end: int,
    step: int = 1,
    fps_capture: Optional[float] = None,
    min_area: int = 20,
    progress_cb: ProgressCb = None,
    preview_cb: PreviewCb = None,
) -> List[Point]:
    """Sigue el objeto por color: centroide del mayor contorno dentro del rango HSV.

    Los frames donde no se detecta el objeto se omiten (no generan punto).
    """
    results: List[Point] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    idxs = list(range(start, end + 1, step))
    for i, idx in enumerate(idxs):
        frame_rgb = reader.get_frame(idx)
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= min_area:
                m = cv2.moments(c)
                if m["m00"] != 0:
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]
                    results.append(
                        (idx, reader.frame_time(idx, fps_capture), cx, cy)
                    )
                    if preview_cb:
                        preview_cb(frame_rgb, cx, cy, idx)
        if progress_cb:
            progress_cb((i + 1) / len(idxs))
    return results


def track_csrt(
    reader: VideoReader,
    roi: Tuple[int, int, int, int],
    start: int,
    end: int,
    step: int = 1,
    fps_capture: Optional[float] = None,
    progress_cb: ProgressCb = None,
    preview_cb: PreviewCb = None,
) -> List[Point]:
    """Sigue el objeto con el tracker CSRT (requiere opencv-contrib-python).

    CSRT necesita procesar frames consecutivos, así que se actualiza en
    todos los frames y se registra uno cada ``step``.
    """
    if not csrt_available():
        raise RuntimeError(
            "CSRT no está disponible. Instala opencv-contrib-python para usarlo."
        )
    tracker = _create_csrt()
    first_rgb = reader.get_frame(start)
    tracker.init(cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR),
                 tuple(int(round(v)) for v in roi))

    x, y, w, h = roi
    results: List[Point] = [
        (start, reader.frame_time(start, fps_capture), x + w / 2.0, y + h / 2.0)
    ]
    if preview_cb:
        preview_cb(first_rgb, x + w / 2.0, y + h / 2.0, start)
    total = max(end - start, 1)
    for idx in range(start + 1, end + 1):
        frame_rgb = reader.get_frame(idx)
        ok, box = tracker.update(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if ok and (idx - start) % step == 0:
            bx, by, bw, bh = box
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            results.append((idx, reader.frame_time(idx, fps_capture), cx, cy))
            if preview_cb:
                preview_cb(frame_rgb, cx, cy, idx)
        if progress_cb:
            progress_cb((idx - start) / total)
    return results
