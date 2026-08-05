"""Tracking manual: tabla de puntos marcados por el usuario.

Guarda un punto por frame (frame_idx, t, x_px, y_px) y, si hay
calibración, calcula (x_m, y_m). El usuario puede agregar, corregir
o borrar puntos.

Este módulo es independiente de Streamlit.

TODO (v2): tracking automático (seguimiento por color u optical flow)
para no tener que marcar frame a frame. Ver core/tracking.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from core.calibration import Calibration


@dataclass
class TrackPoint:
    """Un punto marcado en un frame."""

    frame_idx: int
    t: float
    x_px: float
    y_px: float


@dataclass
class TrackTable:
    """Colección de puntos marcados, con a lo más uno por frame."""

    points: List[TrackPoint] = field(default_factory=list)

    # -- operaciones de edición -------------------------------------------
    def add_or_update(self, frame_idx: int, t: float, x_px: float, y_px: float) -> None:
        """Agrega un punto o reemplaza el existente para ese frame."""
        for p in self.points:
            if p.frame_idx == frame_idx:
                p.t, p.x_px, p.y_px = t, x_px, y_px
                return
        self.points.append(TrackPoint(frame_idx, t, x_px, y_px))
        self.points.sort(key=lambda p: p.frame_idx)

    def delete(self, frame_idx: int) -> None:
        """Elimina el punto de un frame, si existe."""
        self.points = [p for p in self.points if p.frame_idx != frame_idx]

    def clear(self) -> None:
        """Borra todos los puntos."""
        self.points = []

    def get(self, frame_idx: int) -> Optional[TrackPoint]:
        """Devuelve el punto de un frame, o None."""
        for p in self.points:
            if p.frame_idx == frame_idx:
                return p
        return None

    def __len__(self) -> int:
        return len(self.points)

    # -- exportar a DataFrame ---------------------------------------------
    def to_dataframe(self, calibration: Optional[Calibration] = None) -> pd.DataFrame:
        """Devuelve la tabla como DataFrame ordenado por frame.

        Columnas: frame, t, x_px, y_px y, si hay calibración, x_m, y_m.
        """
        cols = ["frame", "t", "x_px", "y_px", "x_m", "y_m"]
        if not self.points:
            return pd.DataFrame(columns=cols)

        pts = sorted(self.points, key=lambda p: p.frame_idx)
        frame = np.array([p.frame_idx for p in pts], dtype=int)
        t = np.array([p.t for p in pts], dtype=float)
        x_px = np.array([p.x_px for p in pts], dtype=float)
        y_px = np.array([p.y_px for p in pts], dtype=float)

        if calibration is not None:
            x_m, y_m = calibration.to_meters(x_px, y_px)
            x_m = np.asarray(x_m, dtype=float)
            y_m = np.asarray(y_m, dtype=float)
        else:
            x_m = np.full(len(pts), np.nan)
            y_m = np.full(len(pts), np.nan)

        return pd.DataFrame(
            {
                "frame": frame,
                "t": t,
                "x_px": x_px,
                "y_px": y_px,
                "x_m": x_m,
                "y_m": y_m,
            }
        )


def from_dataframe(df: pd.DataFrame) -> TrackTable:
    """Reconstruye una ``TrackTable`` desde un DataFrame editado.

    Ignora filas sin frame o sin coordenadas de píxel válidas (p. ej.
    filas nuevas vacías que agregue el editor de tabla).
    """
    table = TrackTable()
    for _, row in df.iterrows():
        if pd.isna(row.get("frame")) or pd.isna(row.get("x_px")) or pd.isna(
            row.get("y_px")
        ):
            continue
        table.add_or_update(
            frame_idx=int(row["frame"]),
            t=float(row["t"]) if not pd.isna(row.get("t")) else 0.0,
            x_px=float(row["x_px"]),
            y_px=float(row["y_px"]),
        )
    return table
