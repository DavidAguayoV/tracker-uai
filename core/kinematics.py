"""Cinemática: velocidad y aceleración por diferencias finitas centradas.

Usa ``numpy.gradient``, que aplica diferencias centradas en el interior
y diferencias de un lado en los extremos, y — muy importante para
videos de celular — soporta paso de tiempo **no uniforme** (frame rate
variable). Opcionalmente suaviza la posición con Savitzky-Golay antes de
derivar, lo que reduce mucho el ruido en la aceleración.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def adjust_window(n: int, window: int, polyorder: int) -> Optional[int]:
    """Ajusta la ventana de Savitzky-Golay a un valor válido.

    Reglas de scipy: la ventana debe ser impar, ``<= n`` y ``> polyorder``.
    Devuelve una ventana válida o ``None`` si no es posible suavizar
    (por ejemplo, hay muy pocos puntos).
    """
    if n <= polyorder + 1:
        return None
    w = int(window)
    if w % 2 == 0:          # debe ser impar
        w += 1
    if w > n:               # no puede exceder el nº de puntos
        w = n if n % 2 == 1 else n - 1
    if w <= polyorder:      # debe ser mayor que el orden del polinomio
        return None
    return w


def savgol_smooth(
    values: np.ndarray, window: int, polyorder: int = 2
) -> np.ndarray:
    """Suaviza una señal con Savitzky-Golay, ajustando la ventana si hace falta.

    Si no se puede suavizar (muy pocos puntos), devuelve la señal original.
    """
    values = np.asarray(values, dtype=float)
    w = adjust_window(len(values), window, polyorder)
    if w is None:
        return values
    return savgol_filter(values, window_length=w, polyorder=polyorder)


def finite_diff(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Derivada por diferencias finitas centradas respecto de ``t`` (no uniforme)."""
    f = np.asarray(f, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(f) < 2:
        return np.full_like(f, np.nan)
    return np.gradient(f, t, edge_order=2 if len(f) > 2 else 1)


def compute_kinematics(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    smooth: bool = False,
    window: int = 5,
    polyorder: int = 2,
) -> Dict[str, np.ndarray]:
    """Calcula velocidad, aceleración y rapidez a partir de la posición.

    Parámetros
    ----------
    t, x, y : arreglos de tiempo (s) y posición (m), del mismo largo y
        ordenados por tiempo.
    smooth : si True, suaviza x e y con Savitzky-Golay antes de derivar.
    window, polyorder : parámetros del filtro Savitzky-Golay.

    Devuelve un diccionario con arreglos: ``x_s``, ``y_s`` (posición usada,
    suavizada o no), ``vx``, ``vy``, ``ax``, ``ay`` y ``speed``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if smooth:
        x_s = savgol_smooth(x, window, polyorder)
        y_s = savgol_smooth(y, window, polyorder)
    else:
        x_s, y_s = x.copy(), y.copy()

    vx = finite_diff(x_s, t)
    vy = finite_diff(y_s, t)
    ax = finite_diff(vx, t)
    ay = finite_diff(vy, t)
    speed = np.hypot(vx, vy)

    return {
        "x_s": x_s,
        "y_s": y_s,
        "vx": vx,
        "vy": vy,
        "ax": ax,
        "ay": ay,
        "speed": speed,
    }


def kinematics_dataframe(
    df: pd.DataFrame,
    smooth: bool = False,
    window: int = 5,
    polyorder: int = 2,
) -> pd.DataFrame:
    """Agrega columnas cinemáticas a un DataFrame con columnas t, x_m, y_m.

    Devuelve una copia con vx, vy, ax, ay, speed (y x_s, y_s si se suavizó).
    """
    out = df.copy().reset_index(drop=True)
    if out.empty or out["x_m"].isna().all():
        for c in ["vx", "vy", "ax", "ay", "speed"]:
            out[c] = np.nan
        return out

    k = compute_kinematics(
        out["t"].to_numpy(),
        out["x_m"].to_numpy(),
        out["y_m"].to_numpy(),
        smooth=smooth,
        window=window,
        polyorder=polyorder,
    )
    if smooth:
        out["x_s"] = k["x_s"]
        out["y_s"] = k["y_s"]
    out["vx"] = k["vx"]
    out["vy"] = k["vy"]
    out["ax"] = k["ax"]
    out["ay"] = k["ay"]
    out["speed"] = k["speed"]
    return out
