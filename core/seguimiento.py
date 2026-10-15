"""Seguimiento robusto para los Laboratorios 3 (varios objetos, trazador pequeño).

Los métodos de :mod:`core.autotrack` buscan **un** objeto: el color toma el
contorno más grande de todo el cuadro y la plantilla, si pierde la
coincidencia, salta al lugar más parecido de toda la imagen. Con dos carros
marcados con la misma cinta, o con un trazador de 2 mm entre burbujas, eso
salta de un objeto a otro. Aquí se agregan:

  * **Predicción** de la posición siguiente a velocidad constante (en px por
    cuadro), y elección del candidato más cercano a la predicción.
  * **Compuerta de salto**: un candidato demasiado lejos de la predicción se
    rechaza y el cuadro queda sin punto (se avisa como hueco).
  * **Exclusión** de las posiciones de otro objeto ya seguido (el carro 1
    cuando se sigue el carro 2).
  * **Recuperación**: si el objeto se pierde varios cuadros seguidos, la
    compuerta se abre y se vuelve a buscar en todo el cuadro, cerca de la
    última posición conocida.
  * Para color, una **muestra** de 5×5 píxeles en vez de un solo píxel, con
    el tono tratado como circular (el rojo está en ambos extremos de H), y
    una apertura morfológica pequeña que no borra objetos de pocos píxeles.

Los métodos originales no cambian: el flujo de un solo objeto (Labs 1 y 2)
sigue usándolos tal cual.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.video import VideoReader

Point = Tuple[int, float, float, float]  # (frame_idx, t, x_px, y_px)
ProgressCb = Optional[Callable[[float], None]]
PreviewCb = Optional[Callable[[np.ndarray, float, float, int], None]]

#: Cuadros perdidos seguidos tras los cuales se busca en todo el cuadro.
RECUPERAR_TRAS = 8


@dataclass
class MuestraColor:
    """Color de referencia del objeto (OpenCV: H ∈ [0, 179], S, V ∈ [0, 255])."""

    h: int
    s: int
    v: int
    h_tol: int = 12
    s_min: int = 60
    v_min: int = 60


@dataclass
class InfoSeguimiento:
    """Resumen del seguimiento, para avisar al estudiante."""

    n_cuadros: int = 0
    n_detectados: int = 0
    n_perdidos: int = 0
    recuperaciones: int = 0
    perdidos: List[int] = field(default_factory=list)


def muestra_color(
    frame_rgb: np.ndarray, x: float, y: float, h_tol: int = 12,
    s_min: int = 60, v_min: int = 60, radio: int = 2,
) -> MuestraColor:
    """Color del objeto alrededor del clic (mediana de un parche, tono circular)."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    H, W = hsv.shape[:2]
    x0, x1 = max(0, int(x) - radio), min(W, int(x) + radio + 1)
    y0, y1 = max(0, int(y) - radio), min(H, int(y) + radio + 1)
    parche = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    ang = parche[:, 0] * (2 * np.pi / 180.0)
    h = int(round((np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180))
    s = int(np.median(parche[:, 1]))
    v = int(np.median(parche[:, 2]))
    return MuestraColor(h % 180, s, v, int(h_tol), int(s_min), int(v_min))


def mascara_color(frame_rgb: np.ndarray, m: MuestraColor) -> np.ndarray:
    """Máscara binaria del color de la muestra (con el tono circular)."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    lo_h, hi_h = m.h - m.h_tol, m.h + m.h_tol
    rangos = []
    if lo_h < 0:
        rangos += [(0, hi_h), (180 + lo_h, 179)]
    elif hi_h > 179:
        rangos += [(lo_h, 179), (0, hi_h - 180)]
    else:
        rangos.append((lo_h, hi_h))
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for a, b in rangos:
        lower = np.array([a, m.s_min, m.v_min], np.uint8)
        upper = np.array([b, 255, 255], np.uint8)
        mask |= cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def candidatos_color(
    frame_rgb: np.ndarray, m: MuestraColor, area_min: float = 4.0
) -> List[Tuple[float, float, float]]:
    """Centroides ``(x, y, área)`` de las manchas del color de la muestra."""
    mask = mascara_color(frame_rgb, m)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area >= area_min:
            out.append((float(cents[i][0]), float(cents[i][1]), area))
    return out


class _Predictor:
    """Posición prevista a velocidad constante, con compuerta que crece al perderse."""

    def __init__(self, compuerta: float):
        self.compuerta = float(compuerta)
        self.ultimo: Optional[Tuple[int, float, float]] = None   # (frame, x, y)
        self.vel = (0.0, 0.0)                                      # px/cuadro
        self.perdidos = 0

    def prevista(self, idx: int) -> Optional[Tuple[float, float]]:
        if self.ultimo is None:
            return None
        f, x, y = self.ultimo
        df = idx - f
        return (x + self.vel[0] * df, y + self.vel[1] * df)

    def radio(self) -> float:
        # Crece al perder el objeto, pero despacio: abrirla de golpe deja
        # entrar a otro objeto parecido que esté cerca (la mina, el otro carro).
        return self.compuerta * (1 + 0.5 * self.perdidos)

    def acepta(self, idx: int, x: float, y: float) -> None:
        if self.ultimo is not None:
            f, x0, y0 = self.ultimo
            df = idx - f
            if 0 < df <= 3:
                nv = ((x - x0) / df, (y - y0) / df)
                # Suaviza la velocidad para que un cuadro ruidoso no la desvíe.
                self.vel = (0.5 * self.vel[0] + 0.5 * nv[0], 0.5 * self.vel[1] + 0.5 * nv[1])
            else:
                self.vel = (0.0, 0.0)
        self.ultimo = (idx, x, y)
        self.perdidos = 0

    def pierde(self) -> None:
        self.perdidos += 1


def _subpixel(res: np.ndarray, loc: Tuple[int, int]) -> Tuple[float, float]:
    """Corrección subpíxel del máximo de correlación (parábola en x y en y).

    Sin ella la plantilla entrega posiciones enteras: con desplazamientos de
    un píxel por cuadro ese redondeo es ruido de ±0,3 px, que infla y sesga
    hacia arriba la rapidez cuadro a cuadro.
    """
    x, y = int(loc[0]), int(loc[1])
    dx = dy = 0.0
    if 0 < x < res.shape[1] - 1:
        a, b, c = res[y, x - 1], res[y, x], res[y, x + 1]
        den = a - 2 * b + c
        if den < 0:
            dx = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    if 0 < y < res.shape[0] - 1:
        a, b, c = res[y - 1, x], res[y, x], res[y + 1, x]
        den = a - 2 * b + c
        if den < 0:
            dy = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return dx, dy


def _posiciones(excluir, idx) -> List[Tuple[float, float]]:
    """Posiciones a excluir en un cuadro: acepta un punto o una lista de puntos."""
    if not excluir:
        return []
    p = excluir.get(idx)
    if p is None:
        return []
    if len(p) == 2 and np.isscalar(p[0]):
        return [(float(p[0]), float(p[1]))]
    return [(float(q[0]), float(q[1])) for q in p]


def _excluido(x, y, idx, excluir, radio) -> bool:
    if radio <= 0:
        return False
    return any(float(np.hypot(x - px, y - py)) < radio
               for px, py in _posiciones(excluir, idx))


def seguir_color(
    reader: VideoReader,
    muestra: MuestraColor,
    start: int,
    end: int,
    step: int = 1,
    fps_capture: Optional[float] = None,
    semilla: Optional[Tuple[float, float]] = None,
    area_min: float = 4.0,
    compuerta: Optional[float] = None,
    excluir: Optional[Dict[int, object]] = None,
    radio_excluir: Optional[float] = None,
    recuperar_tras: int = RECUPERAR_TRAS,
    progress_cb: ProgressCb = None,
    preview_cb: PreviewCb = None,
) -> Tuple[List[Point], InfoSeguimiento]:
    """Sigue un objeto por color eligiendo, en cada cuadro, la mancha prevista.

    ``semilla`` (px) es la posición aproximada del objeto en el primer cuadro
    (el clic del estudiante, o la mina en el modo por lotes): se toma la
    mancha más cercana. Sin semilla se toma la más grande.
    """
    idxs = list(range(int(start), int(end) + 1, max(1, int(step))))
    info = InfoSeguimiento(n_cuadros=len(idxs))
    pts: List[Point] = []
    area_ref: Optional[float] = None
    pred = _Predictor(compuerta or 15.0)

    for i, idx in enumerate(idxs):
        frame = reader.get_frame(idx)
        cands = candidatos_color(frame, muestra, area_min)
        tam = np.sqrt(area_ref) if area_ref else None
        if compuerta is None and tam:
            pred.compuerta = max(15.0, 3.0 * tam)
        r_exc = radio_excluir if radio_excluir is not None else (1.5 * tam if tam else 20.0)
        cands = [c for c in cands if not _excluido(c[0], c[1], idx, excluir, r_exc)]
        if area_ref:
            cands = [c for c in cands if 0.2 * area_ref <= c[2] <= 5.0 * area_ref] or cands

        elegido = None
        if cands:
            p = pred.prevista(idx)
            if p is None:
                if semilla is not None:
                    elegido = min(cands, key=lambda c: np.hypot(c[0] - semilla[0], c[1] - semilla[1]))
                else:
                    elegido = max(cands, key=lambda c: c[2])
            else:
                c = min(cands, key=lambda c: np.hypot(c[0] - p[0], c[1] - p[1]))
                dist = float(np.hypot(c[0] - p[0], c[1] - p[1]))
                if dist <= pred.radio():
                    elegido = c
                elif pred.perdidos >= recuperar_tras:
                    # Recuperación: lo más cercano a la última posición conocida.
                    _, ux, uy = pred.ultimo
                    elegido = min(cands, key=lambda c: np.hypot(c[0] - ux, c[1] - uy))
                    info.recuperaciones += 1

        if elegido is not None:
            x, y, area = elegido
            area_ref = area if area_ref is None else 0.8 * area_ref + 0.2 * area
            pred.acepta(idx, x, y)
            pts.append((idx, reader.frame_time(idx, fps_capture), x, y))
            if preview_cb:
                preview_cb(frame, x, y, idx)
        else:
            pred.pierde()
            info.perdidos.append(idx)
        if progress_cb:
            progress_cb((i + 1) / len(idxs))

    info.n_detectados = len(pts)
    info.n_perdidos = len(info.perdidos)
    return pts, info


def seguir_plantilla(
    reader: VideoReader,
    roi: Tuple[float, float, float, float],
    start: int,
    end: int,
    step: int = 1,
    fps_capture: Optional[float] = None,
    min_score: float = 0.5,
    compuerta: Optional[float] = None,
    excluir: Optional[Dict[int, object]] = None,
    radio_excluir: Optional[float] = None,
    recuperar_tras: int = RECUPERAR_TRAS,
    progress_cb: ProgressCb = None,
    preview_cb: PreviewCb = None,
) -> Tuple[List[Point], InfoSeguimiento]:
    """Coincidencia de plantilla (en color) con predicción, compuerta y exclusión.

    A diferencia de :func:`core.autotrack.track_template`, un cuadro sin
    coincidencia suficiente o demasiado lejos de lo previsto queda **sin
    punto**, en vez de saltar al objeto más parecido del cuadro. «Suficiente»
    es al menos ``min_score`` y al menos el 80 % de la coincidencia típica de
    los cuadros ya aceptados: así un objeto parecido (la mina oscura junto a un
    trazador oscuro) no reemplaza al verdadero cuando este se oculta.
    """
    x, y, w, h = (int(round(v)) for v in roi)
    if w <= 0 or h <= 0:
        raise ValueError("El recuadro del objeto no es válido.")
    # La plantilla se compara EN COLOR: en gris, un trazador naranja y una mina
    # negra son dos discos oscuros casi idénticos para la correlación.
    tmpl = reader.get_frame(int(start))[y:y + h, x:x + w].copy()
    if tmpl.size == 0:
        raise ValueError("El recuadro del objeto queda fuera del cuadro.")
    H, W = reader.height, reader.width
    lado = float(max(w, h))
    pred = _Predictor(compuerta or max(15.0, 1.5 * lado))
    r_exc = radio_excluir if radio_excluir is not None else 0.8 * lado
    centro0 = (x + w / 2.0, y + h / 2.0)   # el primer cuadro se busca aquí

    idxs = list(range(int(start), int(end) + 1, max(1, int(step))))
    info = InfoSeguimiento(n_cuadros=len(idxs))
    pts: List[Point] = []
    puntajes: List[float] = []

    def umbral() -> float:
        if not puntajes:
            return min_score
        return max(min_score, 0.8 * float(np.median(puntajes[-15:])))

    def mejor(img, x0, y0, x1, y1, idx):
        ventana = img[y0:y1, x0:x1]
        if ventana.shape[0] < h or ventana.shape[1] < w:
            return None
        res = cv2.matchTemplate(ventana, tmpl, cv2.TM_CCOEFF_NORMED)
        otros = _posiciones(excluir, idx)
        if otros:
            yy, xx = np.mgrid[0:res.shape[0], 0:res.shape[1]]
            cxm = xx + x0 + w / 2.0
            cym = yy + y0 + h / 2.0
            res = res.copy()
            for px, py in otros:
                res[np.hypot(cxm - px, cym - py) < r_exc] = -1.0
        _, val, _, loc = cv2.minMaxLoc(res)
        dx, dy = _subpixel(res, loc)
        return float(val), x0 + loc[0] + dx + w / 2.0, y0 + loc[1] + dy + h / 2.0

    for i, idx in enumerate(idxs):
        frame = reader.get_frame(idx)
        p = pred.prevista(idx) or centro0
        m = int(max(2 * lado, pred.radio()))
        x0 = max(0, int(p[0] - w / 2 - m)); y0 = max(0, int(p[1] - h / 2 - m))
        x1 = min(W, int(p[0] + w / 2 + m)); y1 = min(H, int(p[1] + h / 2 + m))
        cand = mejor(frame, x0, y0, x1, y1, idx)
        elegido = None
        if cand and cand[0] >= umbral() and np.hypot(cand[1] - p[0], cand[2] - p[1]) <= pred.radio():
            elegido = cand
        elif pred.perdidos >= recuperar_tras or pred.ultimo is None:
            glob = mejor(frame, 0, 0, W, H, idx)
            if glob and glob[0] >= umbral():
                elegido = glob
                if pred.ultimo is not None:
                    info.recuperaciones += 1
        if elegido is not None:
            val, cx, cy = elegido
            puntajes.append(val)
            pred.acepta(idx, cx, cy)
            pts.append((idx, reader.frame_time(idx, fps_capture), cx, cy))
            if preview_cb:
                preview_cb(frame, cx, cy, idx)
        else:
            pred.pierde()
            info.perdidos.append(idx)
        if progress_cb:
            progress_cb((i + 1) / len(idxs))

    info.n_detectados = len(pts)
    info.n_perdidos = len(info.perdidos)
    return pts, info
