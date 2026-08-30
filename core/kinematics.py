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


def position_columns(df: pd.DataFrame) -> tuple:
    """Columnas de posición a graficar: suavizadas si existen, crudas si no.

    Devuelve ``(nombre_x, nombre_y, suavizado)``. Centraliza la regla para que
    la pantalla y las figuras exportadas grafiquen **la misma señal**: antes,
    la app mostraba ``x_s`` con el suavizado activo mientras el PNG exportado
    seguía usando ``x_m``, y el informe terminaba con una figura distinta de
    la que el estudiante había visto.
    """
    if "x_s" in df.columns and "y_s" in df.columns:
        return "x_s", "y_s", True
    return "x_m", "y_m", False


def motion_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Resumen numérico del movimiento, para mostrar junto a los gráficos.

    Espera un DataFrame ya procesado por :func:`kinematics_dataframe`.
    Devuelve un diccionario con ``nan`` en las magnitudes que no se puedan
    calcular (por ejemplo, con menos de dos puntos).
    """
    nan = float("nan")
    vacio = {
        "n_puntos": 0, "duracion": nan, "dt_medio": nan, "fps_efectivo": nan,
        "desplazamiento": nan, "recorrido": nan, "rapidez_media": nan,
        "rapidez_maxima": nan, "aceleracion_media": nan, "ruido_rms": nan,
    }
    if df is None or df.empty:
        return vacio

    cx, cy, _ = position_columns(df)
    t = np.asarray(df["t"], dtype=float)
    x = np.asarray(df[cx], dtype=float)
    y = np.asarray(df[cy], dtype=float)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    n = len(t)
    if n < 2:
        out = dict(vacio)
        out["n_puntos"] = int(n)
        return out

    dt = np.diff(t)
    dt_medio = float(np.mean(dt)) if len(dt) else nan
    pasos = np.hypot(np.diff(x), np.diff(y))

    speed = np.asarray(df["speed"], dtype=float) if "speed" in df else np.array([])
    speed = speed[np.isfinite(speed)]
    acc = (np.hypot(np.asarray(df["ax"], dtype=float),
                    np.asarray(df["ay"], dtype=float))
           if "ax" in df and "ay" in df else np.array([]))
    acc = acc[np.isfinite(acc)]

    # Ruido de marcado: dispersión del residuo respecto de la posición suavizada.
    ruido = nan
    if n >= 7:
        rx = x - savgol_smooth(x, 7, 2)
        ry = y - savgol_smooth(y, 7, 2)
        ruido = float(np.sqrt(np.mean(rx**2 + ry**2)))

    return {
        "n_puntos": int(n),
        "duracion": float(t[-1] - t[0]),
        "dt_medio": dt_medio,
        "fps_efectivo": float(1.0 / dt_medio) if dt_medio and dt_medio > 0 else nan,
        "desplazamiento": float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
        "recorrido": float(np.sum(pasos)),
        "rapidez_media": float(np.mean(speed)) if speed.size else nan,
        "rapidez_maxima": float(np.max(speed)) if speed.size else nan,
        "aceleracion_media": float(np.mean(acc)) if acc.size else nan,
        "ruido_rms": ruido,
    }


def frame_gaps(frames, start: int, end: int, step: int = 1) -> list:
    """Frames del intervalo ``[start, end]`` (cada ``step``) que no tienen punto.

    Sirve para avisar al estudiante de que el seguimiento automático perdió el
    objeto en algún tramo, antes de que lo descubra mirando un gráfico raro.
    """
    marcados = {int(f) for f in frames}
    step = max(1, int(step))
    return [f for f in range(int(start), int(end) + 1, step) if f not in marcados]
