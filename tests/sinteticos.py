"""Vídeos sintéticos con resultado conocido, para verificar los Laboratorios 3.

No se versionan vídeos (``*.mp4`` está en ``.gitignore``): se generan en cada
corrida de pytest, en una carpeta temporal, en pocos segundos.

* :func:`video_colision` — dos carros sobre un riel, vistos de lado, con la
  **misma marca naranja** (el caso difícil: el seguimiento no puede
  distinguirlos por color). El choque se modela como una interacción suave de
  ~0,1 s, como la repulsión de los imanes, que conserva el momentum en cada
  instante. Fuera de ella las velocidades son exactamente las dadas.
* :func:`video_orbita` — trazador de 2 mm girando alrededor de una mina
  oscura, con ruido de imagen, temblor de posición, burbujas blancas que
  aparecen y desaparecen y algunos cuadros en que el trazador no se ve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

SUB = 4                 # bits de subpíxel para cv2 (dibujo con precisión 1/16 px)
NARANJA = (255, 140, 0)  # RGB


def _escribir(frames: List[np.ndarray], ruta: Path, fps: float) -> Path:
    """Escribe los cuadros RGB en mp4 (o avi MJPG si falta el códec)."""
    h, w = frames[0].shape[:2]
    ruta = Path(ruta)
    vw = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        ruta = ruta.with_suffix(".avi")
        vw = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    return ruta


def _suave(s: np.ndarray) -> np.ndarray:
    """Escalón suave de 0 a 1 (smoothstep), con s ∈ [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


# ==========================================================================
# FIS101: dos carros
# ==========================================================================
@dataclass
class Colision:
    """Parámetros y valores exactos de un ensayo sintético."""

    m1: float
    m2: float
    v1i: float
    v2i: float
    e: float               # coeficiente de restitución impuesto
    t_choque: float = 1.2
    tau: float = 0.10      # duración de la interacción magnética (s)
    duracion: float = 2.8
    fps: float = 30.0
    escala: float = 0.0015  # m/px

    @property
    def v1f(self) -> float:
        m1, m2, e = self.m1, self.m2, self.e
        return ((m1 - e * m2) * self.v1i + (1 + e) * m2 * self.v2i) / (m1 + m2)

    @property
    def v2f(self) -> float:
        m1, m2, e = self.m1, self.m2, self.e
        return ((m2 - e * m1) * self.v2i + (1 + e) * m1 * self.v1i) / (m1 + m2)

    def posiciones(self, t: np.ndarray, x1c: float, x2c: float):
        """Posición (m) de cada carro; en t_choque valen x1c y x2c."""
        def x(t, xc, vi, vf):
            # Integral exacta de v(t) = vi + (vf − vi)·S((t − tc)/τ + ½).
            out = np.empty_like(t)
            for i, ti in enumerate(t):
                g = np.linspace(self.t_choque, ti, 400)
                v = vi + (vf - vi) * _suave((g - self.t_choque) / self.tau + 0.5)
                out[i] = xc + np.trapezoid(v, g)
            return out
        return x(t, x1c, self.v1i, self.v1f), x(t, x2c, self.v2i, self.v2f)


CASO_A = Colision(m1=0.250, m2=0.250, v1i=0.30, v2i=0.0, e=1.0)
CASO_D = Colision(m1=0.250, m2=0.500, v1i=0.30, v2i=-0.20, e=0.7)

#: Geometría del dibujo (px).
ANCHO, ALTO = 1280, 480
Y_RIEL = 300
LARGO_CARRO = 113        # 0,17 m a 1,5 mm/px
ALTO_CARRO = 40
LADO_MARCA = 22


def video_colision(caso: Colision, ruta: Path, invertir: bool = False, semilla: int = 1):
    """Genera el vídeo del ensayo y devuelve ``(ruta, datos)``.

    ``datos`` trae los centros de las marcas (px) por cuadro, para fijar el
    clic del estudiante en los tests. Con ``invertir=True`` el carro 1 parte
    hacia la izquierda (el eje x del vídeo apunta al revés que la guía).
    """
    rng = np.random.default_rng(semilla)
    n = int(round(caso.duracion * caso.fps))
    t = np.arange(n) / caso.fps
    x1c = 390 * caso.escala
    x2c = x1c + (LARGO_CARRO + 20) * caso.escala
    x1, x2 = caso.posiciones(t, x1c, x2c)
    px1, px2 = x1 / caso.escala, x2 / caso.escala
    if invertir:
        px1, px2 = ANCHO - px1, ANCHO - px2

    fondo = np.full((ALTO, ANCHO, 3), 225, np.uint8)
    cv2.line(fondo, (0, Y_RIEL + ALTO_CARRO // 2 + 6), (ANCHO, Y_RIEL + ALTO_CARRO // 2 + 6),
             (70, 70, 70), 6)
    for xr in range(0, ANCHO, 100):   # regla del riel
        cv2.line(fondo, (xr, Y_RIEL + 40), (xr, Y_RIEL + 52), (40, 40, 40), 2)

    frames = []
    k = 1 << SUB
    for i in range(n):
        img = fondo.copy()
        for cx in (px1[i], px2[i]):
            x0, x1_ = cx - LARGO_CARRO / 2, cx + LARGO_CARRO / 2
            y0, y1_ = Y_RIEL - ALTO_CARRO / 2, Y_RIEL + ALTO_CARRO / 2
            pts = np.array([[x0, y0], [x1_, y0], [x1_, y1_], [x0, y1_]]) * k
            cv2.fillPoly(img, [pts.astype(np.int32)], (40, 70, 160), cv2.LINE_AA, SUB)
            m = LADO_MARCA / 2
            pm = np.array([[cx - m, Y_RIEL - m], [cx + m, Y_RIEL - m],
                           [cx + m, Y_RIEL + m], [cx - m, Y_RIEL + m]]) * k
            cv2.fillPoly(img, [pm.astype(np.int32)], NARANJA, cv2.LINE_AA, SUB)
        ruido = rng.normal(0, 4, img.shape)
        frames.append(np.clip(img + ruido, 0, 255).astype(np.uint8))
    ruta = _escribir(frames, ruta, caso.fps)
    return ruta, {"t": t, "px1": px1, "px2": px2, "y": float(Y_RIEL)}


# ==========================================================================
# FIS301: trazador en órbita
# ==========================================================================
@dataclass
class Orbita:
    """Órbita sintética: radio (m), rapidez (m/s), sentido y escala."""

    radio: float = 0.005
    rapidez: float = 0.005
    horario: bool = True
    duracion: float = 8.0
    fps: float = 30.0
    escala: float = 0.0002          # m/px (0,2 mm/px)
    t_arranque: float = 0.0         # s con rapidez creciente al inicio
    temblor_px: float = 0.08        # desviación del temblor de posición (px)
    perdidos: Tuple[int, ...] = (60, 61, 62, 150)


ANCHO_O, ALTO_O = 640, 480
MINA_PX = (320.0, 240.0)


def video_orbita(o: Orbita, ruta: Path, semilla: int = 2):
    """Genera el vídeo de la órbita y devuelve ``(ruta, datos)``."""
    rng = np.random.default_rng(semilla)
    n = int(round(o.duracion * o.fps))
    t = np.arange(n) / o.fps
    w = o.rapidez / o.radio
    if o.t_arranque > 0:
        # Rapidez que crece de 0 a la final en t_arranque; luego constante.
        g = np.linspace(0, t[-1], 4000)
        wg = w * _suave(g / o.t_arranque)
        th_g = np.concatenate([[0], np.cumsum(0.5 * (wg[1:] + wg[:-1]) * np.diff(g))])
        th = np.interp(t, g, th_g)
    else:
        th = w * t
    signo = -1.0 if o.horario else 1.0
    r_px = o.radio / o.escala
    # Ángulo físico (y hacia arriba): horario = ángulo decreciente.
    cx = MINA_PX[0] + r_px * np.cos(signo * th)
    cy = MINA_PX[1] - r_px * np.sin(signo * th)
    cx += rng.normal(0, o.temblor_px, n)
    cy += rng.normal(0, o.temblor_px, n)

    fondo = np.full((ALTO_O, ANCHO_O, 3), (190, 205, 215), np.uint8)
    cv2.circle(fondo, (320, 240), 230, (170, 185, 195), 3, cv2.LINE_AA)   # borde de la cápsula
    cv2.circle(fondo, (int(MINA_PX[0]), int(MINA_PX[1])), 6, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.circle(fondo, (320, 400), 6, (25, 25, 25), -1, cv2.LINE_AA)       # la otra mina
    k = 1 << SUB
    r_traz = 0.001 / o.escala      # trazador de 2 mm de diámetro
    burbujas = [(rng.uniform(150, 490), rng.uniform(100, 380), rng.uniform(2, 6),
                 rng.integers(0, n), rng.integers(15, 60)) for _ in range(40)]
    frames = []
    for i in range(n):
        img = fondo.copy()
        if i not in o.perdidos:
            cv2.circle(img, (int(cx[i] * k), int(cy[i] * k)), int(r_traz * k), NARANJA,
                       -1, cv2.LINE_AA, SUB)
        for bx, by, br, b0, bd in burbujas:
            if b0 <= i < b0 + bd:
                cv2.circle(img, (int(bx), int(by)), int(br), (245, 245, 245), -1, cv2.LINE_AA)
                cv2.circle(img, (int(bx), int(by)), int(br), (160, 170, 175), 1, cv2.LINE_AA)
        ruido = rng.normal(0, 4, img.shape)
        frames.append(np.clip(img + ruido, 0, 255).astype(np.uint8))
    ruta = _escribir(frames, ruta, o.fps)
    return ruta, {"t": t, "cx": cx, "cy": cy, "mina": MINA_PX}
