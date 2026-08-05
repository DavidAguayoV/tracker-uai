"""Calibración espacial: escala píxel→metro, origen y eje Y físico.

Convención de coordenadas:
  * La imagen tiene su origen (0,0) arriba-izquierda y el eje Y crece
    hacia abajo (convención de imagen).
  * En física queremos el origen donde el usuario lo marque y el eje Y
    creciendo hacia arriba. Por eso, al pasar a metros, invertimos Y.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

Number = Union[float, np.ndarray]


def scale_from_two_points(
    p1_px: Tuple[float, float],
    p2_px: Tuple[float, float],
    real_distance_m: float,
) -> float:
    """Factor de escala en metros por píxel a partir de dos puntos.

    ``p1_px`` y ``p2_px`` son (x, y) en píxeles de la imagen original y
    ``real_distance_m`` es la distancia real conocida entre ambos, en metros.

    Lanza ``ValueError`` si los puntos coinciden o la distancia no es positiva.
    """
    dx = p2_px[0] - p1_px[0]
    dy = p2_px[1] - p1_px[1]
    dist_px = math.hypot(dx, dy)
    if dist_px <= 0:
        raise ValueError(
            "Los dos puntos de calibración no pueden ser el mismo punto."
        )
    if real_distance_m <= 0:
        raise ValueError("La distancia real debe ser mayor que cero.")
    return real_distance_m / dist_px


def angle_from_points(
    origin_px: Tuple[float, float], x_dir_px: Tuple[float, float]
) -> float:
    """Ángulo (rad) del eje +X, desde el origen hacia el punto de dirección.

    Medido en coordenadas de imagen (Y hacia abajo). Un punto a la derecha
    del origen da 0; hacia arriba-derecha, un ángulo negativo.
    """
    dx = x_dir_px[0] - origin_px[0]
    dy = x_dir_px[1] - origin_px[1]
    return math.atan2(dy, dx)


@dataclass
class Calibration:
    """Transforma coordenadas de píxel a metros en el marco físico.

    Atributos
    ---------
    m_per_px : factor de escala (metros por píxel).
    origin_px : (x0, y0) del origen físico, en píxeles de la imagen.
    angle_rad : orientación del eje +X respecto de la horizontal de la
        imagen (rad, en coordenadas de imagen con Y hacia abajo). 0 = el
        eje X apunta a la derecha (comportamiento por defecto). Permite
        alinear los ejes con la geometría real del montaje (p. ej. un
        plano inclinado o un péndulo cónico visto desde arriba).
    """

    m_per_px: float
    origin_px: Tuple[float, float]
    angle_rad: float = 0.0

    def to_meters(self, x_px: Number, y_px: Number) -> Tuple[Number, Number]:
        """Convierte píxeles a metros: traslada al origen, rota y escala.

        El eje +Y físico queda perpendicular a +X y apuntando "hacia
        arriba" en el marco rotado. Acepta escalares o arreglos de numpy.

        Fórmula (α = angle_rad; dx, dy respecto del origen, Y de imagen):
            x_m = ( dx·cosα + dy·sinα) · escala
            y_m = ( dx·sinα − dy·cosα) · escala
        Con α = 0 se recupera x_m = dx·escala, y_m = −dy·escala.
        """
        x0, y0 = self.origin_px
        ca, sa = math.cos(self.angle_rad), math.sin(self.angle_rad)
        dx = np.asarray(x_px, dtype=float) - x0
        dy = np.asarray(y_px, dtype=float) - y0
        x_m = (dx * ca + dy * sa) * self.m_per_px
        y_m = (dx * sa - dy * ca) * self.m_per_px
        if np.ndim(x_px) == 0 and np.ndim(y_px) == 0:
            return float(x_m), float(y_m)
        return x_m, y_m

    @property
    def angle_deg(self) -> float:
        """Orientación del eje +X en grados (para mostrar en la UI)."""
        return math.degrees(self.angle_rad)

    def summary(self) -> str:
        """Texto breve con el estado de la calibración (para la UI)."""
        x0, y0 = self.origin_px
        base = (
            f"escala = {self.m_per_px:.6g} m/px  ·  "
            f"origen = ({x0:.0f}, {y0:.0f}) px"
        )
        if abs(self.angle_rad) > 1e-9:
            base += f"  ·  eje X rotado {self.angle_deg:.1f}°"
        return base
