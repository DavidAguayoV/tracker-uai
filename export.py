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

from matplotlib.ticker import FuncFormatter  # noqa: E402

from core.formato import csv_es  # noqa: E402
from core.kinematics import position_columns, savgol_smooth  # noqa: E402


def _fig_to_png(fig) -> bytes:
    """Convierte una figura de matplotlib a bytes PNG y la cierra."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_position(df: pd.DataFrame) -> bytes:
    """Figura de posición x(t) e y(t).

    Usa la posición suavizada si el estudiante activó el suavizado, igual que
    los gráficos en pantalla, para que la figura del informe sea la misma que
    vio en la app.
    """
    cx, cy, suave = position_columns(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["t"], df[cx], "o-", label="x(t)", ms=3)
    ax.plot(df["t"], df[cy], "s-", label="y(t)", ms=3)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("posición suavizada (m)" if suave else "posición (m)")
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
    cx, cy, suave = position_columns(df)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df[cx], df[cy], "o-", ms=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trayectoria x-y" + (" (suavizada)" if suave else ""))
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
    """Serializa el DataFrame a CSV para Excel en español.

    «;» como separador y coma decimal (UTF-8 con BOM). Antes se usaban la coma
    y el punto: Excel con configuración regional en español abría el archivo
    con todas las columnas metidas en la primera.
    """
    return csv_es(df)


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


def build_zip_archivos(archivos: Dict[str, bytes]) -> bytes:
    """Empaqueta un diccionario nombre→bytes en un ZIP (datos crudos por vídeo)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in archivos.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ==========================================================================
# Laboratorios 3: varios objetos, colisiones, órbita y análisis de grupo
# ==========================================================================
COLORES = ["#0E7490", "#B45309", "#7C3AED", "#BE123C"]


def _coma_ejes(ax) -> None:
    """Coma decimal en las marcas de ambos ejes (convención de los informes)."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def datos_multiobjeto(kdfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une las tablas de varios objetos en una sola, una fila por frame.

    Las columnas de cada objeto llevan su nombre: ``x_m [Carro 1]``, … Un
    frame donde un objeto no tiene punto queda vacío en sus columnas.
    """
    out = None
    for nombre, df in kdfs.items():
        if df is None or df.empty:
            continue
        base = df[["frame", "t"]].reset_index(drop=True)
        resto = df.drop(columns=["frame", "t"]).reset_index(drop=True)
        resto.columns = [f"{c} [{nombre}]" for c in resto.columns]
        d = pd.concat([base, resto], axis=1)
        if out is None:
            out = d
        else:
            out = out.merge(d, on="frame", how="outer", suffixes=("", "_dup"))
            out["t"] = out["t"].fillna(out["t_dup"])
            out = out.drop(columns="t_dup")
    if out is None:
        return pd.DataFrame(columns=["frame", "t"])
    return out.sort_values("frame").reset_index(drop=True)


def fig_velocidad_objetos(kdfs: Dict[str, pd.DataFrame], eje: str = "vx") -> bytes:
    """Velocidad de todos los objetos en un mismo gráfico."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (nombre, df) in enumerate(kdfs.items()):
        if df is None or df.empty or eje not in df:
            continue
        ax.plot(df["t"], df[eje], "o-", ms=3, label=nombre, color=COLORES[i % 4])
    ax.set_xlabel("t (s)")
    ax.set_ylabel(f"{eje} (m/s)")
    ax.set_title("Velocidad de los objetos")
    ax.grid(alpha=0.3)
    ax.legend()
    _coma_ejes(ax)
    return _fig_to_png(fig)


def fig_colision(res, series) -> bytes:
    """Gráfico modelo de la guía: velocidad con signo, tramos y sus promedios.

    ``series`` = [(nombre, t, v_con_signo), ...] (carro 1 primero);
    ``res`` es un :class:`core.colisiones.ResultadoColision`.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for tr in (res.antes, res.despues):
        ax.axvspan(tr.t_ini, tr.t_fin, color="#0E7490", alpha=0.08)
    marcas = ["o", "^"]
    medias = [(res.v1i, res.v1f, "v_{1i}", "v_{1f}"), (res.v2i, res.v2f, "v_{2i}", "v_{2f}")]
    for i, (nombre, t, v) in enumerate(series):
        ax.plot(t, v, marcas[i % 2], ms=3.5, color=COLORES[i % 4], label=nombre)
        if i < 2:
            vi, vf, li, lf = medias[i]
            for vt, tr, lab in ((vi, res.antes, li), (vf, res.despues, lf)):
                if vt is not None and np.isfinite(vt.media):
                    ax.plot([tr.t_ini, tr.t_fin], [vt.media, vt.media], "--",
                            color="#475569", lw=1.2)
                    ax.annotate(f"${lab}$", ((tr.t_ini + tr.t_fin) / 2, vt.media),
                                textcoords="offset points", xytext=(0, 5),
                                ha="center", fontsize=9)
    ax.axvline(res.t_choque, color="#94A3B8", ls=":", lw=1)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Velocidad con signo (m/s)")
    ax.set_title("Velocidad de los carros (tramos promediados sombreados)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    _coma_ejes(ax)
    return _fig_to_png(fig)


def fig_orbita(res, t, x, y, v) -> bytes:
    """Órbita (con el círculo ajustado y la mina) y rapidez cuadro a cuadro."""
    mm = 1000.0
    t, x, y, v = (np.asarray(a, dtype=float) for a in (t, x, y, v))
    sel = (t >= res.t_ini - 1e-9) & (t <= res.t_fin + 1e-9)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                 gridspec_kw={"width_ratios": [1, 1.4]})
    a1.plot(x[~sel] * mm, y[~sel] * mm, ".", color="#CBD5E1", ms=3, label="fuera del tramo")
    a1.plot(x[sel] * mm, y[sel] * mm, ".", color=COLORES[0], ms=3, label="tramo estable")
    c = res.circulo
    th = np.linspace(0, 2 * np.pi, 300)
    a1.plot((c.cx + c.r * np.cos(th)) * mm, (c.cy + c.r * np.sin(th)) * mm, "-",
            color=COLORES[1], lw=1.2, label="círculo ajustado")
    if res.mina is not None:
        a1.plot(res.mina[0] * mm, res.mina[1] * mm, "k*", ms=11, label="mina")
    a1.set_aspect("equal", adjustable="datalim")
    a1.set_xlabel("x (mm)")
    a1.set_ylabel("y (mm)")
    a1.set_title("Órbita del trazador")
    a1.legend(fontsize=8, loc="best")
    a1.grid(alpha=0.3)
    _coma_ejes(a1)

    a2.axvspan(res.t_ini, res.t_fin, color="#0E7490", alpha=0.08)
    a2.plot(t, v * mm, "o-", ms=2.5, lw=0.6, color=COLORES[0])
    a2.axhline(res.rapidez_media * mm, color="#475569", ls="--", lw=1.2)
    a2.set_xlabel("t (s)")
    a2.set_ylabel("rapidez (mm/s)")
    a2.set_title("Rapidez cuadro a cuadro (tramo estable sombreado)")
    a2.grid(alpha=0.3)
    _coma_ejes(a2)
    fig.tight_layout()
    return _fig_to_png(fig)


def fig_grupo_fis101(df: pd.DataFrame) -> bytes:
    """C_R, Δp/p_i y Δp por ensayo, coloreados por bloque."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    col = dict(zip("ABCD", COLORES))
    e = pd.to_numeric(df["ensayo"], errors="coerce")
    bloques = df["bloque"].astype(str).str.upper()
    for ax, c, lab in ((axs[0], "C_R", "$C_R$"), (axs[1], "Δp/p_i (%)", "Δp/p_i (%)"),
                       (axs[2], "Δp (kg·m/s)", "Δp (kg·m/s)")):
        y = pd.to_numeric(df[c], errors="coerce")
        for b in "ABCD":
            m = bloques == b
            if m.any():
                ax.plot(e[m], y[m], "o", color=col[b], label=f"bloque {b}")
        ax.set_xlabel("ensayo")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        _coma_ejes(ax)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def fig_v_I(df: pd.DataFrame, ajustes: dict) -> bytes:
    """v(I) por distancia, con la recta por el origen de cada una."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, b in enumerate(sorted(ajustes)):
        g = df[df["bloque"].astype(str) == b]
        I = pd.to_numeric(g["I (mA)"], errors="coerce")
        v = pd.to_numeric(g["v (mm/s)"], errors="coerce")
        u = pd.to_numeric(g["u_v (mm/s)"], errors="coerce") if "u_v (mm/s)" in g else None
        color = COLORES[i % 4]
        ax.errorbar(I, v, yerr=u, fmt="o", color=color, ms=5, capsize=2, label=b)
        a = ajustes[b]
        if np.isfinite(a.k) and I.notna().any():
            xs = np.linspace(0, float(np.nanmax(I)) * 1.05, 50)
            ax.plot(xs, a.predict(xs), "-", color=color, lw=1.2,
                    label=f"{b}: v = {a.k:.4g}·I,  R² = {a.r2:.4f}".replace(".", ","))
    ax.set_xlabel("Corriente I (mA)")
    ax.set_ylabel("Rapidez v (mm/s)")
    ax.set_title("v(I) por distancia, con ajuste por el origen")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _coma_ejes(ax)
    return _fig_to_png(fig)


def fig_v_V(df: pd.DataFrame) -> bytes:
    """v(V) por distancia, sin ajuste (relación auxiliar de la guía)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, b in enumerate(sorted(df["bloque"].dropna().astype(str).unique())):
        g = df[df["bloque"].astype(str) == b]
        V = pd.to_numeric(g["V (V)"], errors="coerce").to_numpy()
        v = pd.to_numeric(g["v (mm/s)"], errors="coerce").to_numpy()
        ax.plot(V, v, "o", color=COLORES[i % 4], label=b)
    ax.set_xlabel("Voltaje V (V)")
    ax.set_ylabel("Rapidez v (mm/s)")
    ax.set_title("v(V) por distancia (sin ajuste)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _coma_ejes(ax)
    return _fig_to_png(fig)
