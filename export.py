"""Exportación de datos y figuras de Tracker UAI.

Genera el CSV con datos crudos + derivados y las figuras en PNG
(matplotlib, backend Agg, sin ventana). Todo se puede empaquetar en un
ZIP para descarga. Módulo independiente de Streamlit.
"""

from __future__ import annotations

import io
import zipfile
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # backend sin GUI, apto para servidor
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.kinematics import savgol_smooth  # noqa: E402


def _fig_to_png(fig) -> bytes:
    """Convierte una figura de matplotlib a bytes PNG y la cierra."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_position(df: pd.DataFrame) -> bytes:
    """Figura de posición x(t) e y(t)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["t"], df["x_m"], "o-", label="x(t)", ms=3)
    ax.plot(df["t"], df["y_m"], "s-", label="y(t)", ms=3)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("posición (m)")
    ax.set_title("Posición vs tiempo")
    ax.legend()
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def fig_velocity(df: pd.DataFrame) -> bytes:
    """Figura de velocidad vx(t) y vy(t)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["t"], df["vx"], "o-", label="vx(t)", ms=3)
    ax.plot(df["t"], df["vy"], "s-", label="vy(t)", ms=3)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("velocidad (m/s)")
    ax.set_title("Velocidad vs tiempo")
    ax.legend()
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def fig_acceleration(df: pd.DataFrame) -> bytes:
    """Figura de aceleración ax(t) y ay(t)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["t"], df["ax"], "o-", label="ax(t)", ms=3)
    ax.plot(df["t"], df["ay"], "s-", label="ay(t)", ms=3)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("aceleración (m/s²)")
    ax.set_title("Aceleración vs tiempo")
    ax.legend()
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def fig_trajectory(df: pd.DataFrame) -> bytes:
    """Figura de la trayectoria x-y con aspecto igual."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df["x_m"], df["y_m"], "o-", ms=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trayectoria x-y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def fig_detection_error(df: pd.DataFrame, window: int = 7) -> bytes:
    """Figura del error de detección: residuo (posición cruda − suavizada).

    Estima el ruido de marcado/detección comparando la posición medida
    con una versión suavizada (Savitzky-Golay).
    """
    x = df["x_m"].to_numpy()
    y = df["y_m"].to_numpy()
    xs = savgol_smooth(x, window, 2)
    ys = savgol_smooth(y, window, 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["t"], x - xs, "o-", label="residuo x", ms=3)
    ax.plot(df["t"], y - ys, "s-", label="residuo y", ms=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("residuo (m)")
    ax.set_title("Error de detección (crudo − suavizado)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


# Registro de figuras disponibles: etiqueta -> (función, nombre de archivo).
FIGURES = {
    "Posición vs tiempo": (fig_position, "posicion_t.png"),
    "Velocidad vs tiempo": (fig_velocity, "velocidad_t.png"),
    "Aceleración vs tiempo": (fig_acceleration, "aceleracion_t.png"),
    "Trayectoria x-y": (fig_trajectory, "trayectoria.png"),
    "Error de detección": (fig_detection_error, "error_deteccion.png"),
}


def experimental_scatter_bytes(
    series: list,
    title: str,
    xlabel: str,
    ylabel: str,
    fmt: str = "png",
) -> bytes:
    """Genera un gráfico de dispersión con ajustes, apto para informes.

    ``series`` es una lista de diccionarios con las claves:
      ``name``, ``color``, ``x``, ``y`` y, opcionalmente,
      ``fit_x``, ``fit_y``, ``fit_label`` (para dibujar la curva de ajuste).

    ``fmt`` puede ser ``"png"`` o ``"svg"``. Devuelve los bytes de la imagen.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in series:
        ax.scatter(s["x"], s["y"], label=s["name"], color=s["color"],
                   s=35, zorder=3, edgecolors="white", linewidths=0.5)
        if s.get("fit_x") is not None and s.get("fit_y") is not None:
            ax.plot(s["fit_x"], s["fit_y"], color=s["color"], lw=2,
                    ls="--", label=s.get("fit_label", f"{s['name']} (ajuste)"),
                    zorder=2)
    ax.set_title(title or "Gráfico experimental")
    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")
    ax.grid(alpha=0.3)
    if series:
        ax.legend()
    ax.margins(0.05)  # escalado automático con un pequeño margen
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializa el DataFrame a CSV (UTF-8 con BOM para Excel en español)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def build_zip(
    pngs: Dict[str, bytes], csv_bytes: Optional[bytes] = None
) -> bytes:
    """Empaqueta figuras (nombre→bytes) y, opcionalmente, el CSV en un ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if csv_bytes is not None:
            zf.writestr("datos_tracker_uai.csv", csv_bytes)
        for name, data in pngs.items():
            zf.writestr(name, data)
    return buf.getvalue()
