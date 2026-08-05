"""Tracker UAI — análisis de video para laboratorios de física.

Interfaz Streamlit. Toda la lógica vive en ``core/`` para poder
testearla sin Streamlit.

Ejecutar con:
    python -m streamlit run app.py

Etapas implementadas:
  1. Video       — carga y navegación frame a frame.
  2. Calibración — escala px→m, origen y eje Y físico (arriba +).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from core.autotrack import (
    csrt_available,
    hsv_range_from_pixel,
    track_color,
    track_csrt,
    track_template,
)
from core.calibration import Calibration, angle_from_points, scale_from_two_points
from core.fitting import (
    MODELS,
    acceleration_from_quadratic,
    fit_conical_pendulum,
    fit_ellipse,
    fit_model,
    period_by_zero_crossings,
    piecewise_linear_fit,
    plateau_value,
    swept_area_segments,
)
from core.kinematics import kinematics_dataframe
from core.tracking import TrackTable, from_dataframe
from core.video import VideoReader, save_uploaded_to_temp
from models.experiment_presets import PRESETS, preset_labels
import export as export_mod

st.set_page_config(page_title="Tracker UAI", page_icon="🎥", layout="wide")

# Ancho máximo (px) al que mostramos el frame para marcar clics / revisar.
# Acotado para que el video no ocupe toda la pantalla en layout ancho.
DISPLAY_WIDTH = 680


# --------------------------------------------------------------------------
# Estado de sesión
# --------------------------------------------------------------------------
def _init_state() -> None:
    """Inicializa las claves de ``st.session_state`` que usa la app."""
    st.session_state.setdefault("video_path", None)
    st.session_state.setdefault("reader", None)
    st.session_state.setdefault("frame_idx", 0)
    st.session_state.setdefault("fps_capture", None)
    # Intervalo de procesamiento (frames inicial/final)
    st.session_state.setdefault("proc_start", 0)
    st.session_state.setdefault("proc_end", 0)
    st.session_state.setdefault("_last_interval", None)
    # Calibración
    st.session_state.setdefault("calib_p1", None)      # (x_px, y_px)
    st.session_state.setdefault("calib_p2", None)
    st.session_state.setdefault("calib_origin", None)
    st.session_state.setdefault("calib_target", "Punto 1 (escala)")
    st.session_state.setdefault("calib_real_dist", 1.0)
    st.session_state.setdefault("calib_xdir", None)    # punto que define +X
    st.session_state.setdefault("calib_use_rotation", False)
    st.session_state.setdefault("calibration", None)   # objeto Calibration
    st.session_state.setdefault("_calib_last_click", None)
    # Tracking
    st.session_state.setdefault("track", TrackTable())
    st.session_state.setdefault("track_step", 1)       # cada cuántos frames marcar
    st.session_state.setdefault("_track_last_click", None)
    st.session_state.setdefault("track_mode", "Manual")
    st.session_state.setdefault("auto_roi", None)      # (x,y,w,h) para plantilla/CSRT
    st.session_state.setdefault("auto_hsv", None)      # (lower, upper) para color
    st.session_state.setdefault("_auto_last_click", None)
    # Cinemática
    st.session_state.setdefault("kin_smooth", False)
    st.session_state.setdefault("kin_window", 7)
    st.session_state.setdefault("kin_poly", 2)
    st.session_state.setdefault("kin_df", None)
    # Ajuste
    st.session_state.setdefault("preset_idx", 0)
    st.session_state.setdefault("fit_result", None)
    # Módulo de gráficos experimentales (independiente del video)
    st.session_state.setdefault("graph_series", None)
    st.session_state.setdefault("graph_title", "Gráfico experimental")
    st.session_state.setdefault("graph_xlabel", "x")
    st.session_state.setdefault("graph_ylabel", "y")
    st.session_state.setdefault("graph_template", "Genérico")
    # Medición geométrica sobre foto (distancias, ángulos, áreas)
    st.session_state.setdefault("meas_image", None)      # imagen RGB (np.ndarray)
    st.session_state.setdefault("meas_mode", "Calibrar (escala)")
    st.session_state.setdefault("meas_points", {})       # modo -> lista de puntos
    st.session_state.setdefault("meas_scale", None)      # m/px
    st.session_state.setdefault("meas_realdist", 0.10)
    st.session_state.setdefault("_meas_last_click", None)
    st.session_state.setdefault("meas_saved", [])        # mediciones guardadas


_init_state()


# --------------------------------------------------------------------------
# Helpers de interfaz
# --------------------------------------------------------------------------
def fit_display(img: np.ndarray) -> np.ndarray:
    """Reduce la imagen a DISPLAY_WIDTH de ancho para acelerar la codificación.

    No afecta el mapeo de coordenadas: el componente reporta el ancho mostrado
    y ``click_to_original`` escala respecto del ancho original del video.
    """
    if img.shape[1] <= DISPLAY_WIDTH:
        return img
    h = int(img.shape[0] * DISPLAY_WIDTH / img.shape[1])
    return cv2.resize(img, (DISPLAY_WIDTH, h), interpolation=cv2.INTER_AREA)


def click_to_original(value: dict, orig_width: int) -> Optional[Tuple[float, float]]:
    """Convierte el clic devuelto por el componente a píxeles de la imagen original.

    El componente entrega x, y en píxeles *mostrados* junto con el ancho
    mostrado. Escalamos por ``orig_width / width_mostrado``.
    """
    if not value or "x" not in value or not value.get("width"):
        return None
    scale = orig_width / float(value["width"])
    return (float(value["x"]) * scale, float(value["y"]) * scale)


def annotate_frame(
    frame: np.ndarray,
    p1: Optional[Tuple[float, float]],
    p2: Optional[Tuple[float, float]],
    origin: Optional[Tuple[float, float]],
    xdir: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Dibuja marcadores de calibración sobre una copia del frame (RGB).

    Si se entrega ``xdir``, los ejes se dibujan rotados según la dirección
    origen→xdir (eje +X); si no, X apunta a la derecha y Y hacia arriba.
    """
    img = frame.copy()
    r = max(4, img.shape[1] // 200)          # radio de puntos según resolución
    th = max(2, img.shape[1] // 400)         # grosor de líneas

    if p1 is not None:
        cv2.circle(img, (int(p1[0]), int(p1[1])), r, (0, 220, 0), -1)
    if p2 is not None:
        cv2.circle(img, (int(p2[0]), int(p2[1])), r, (0, 220, 0), -1)
    if p1 is not None and p2 is not None:
        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (0, 220, 0),
            th,
        )
    if origin is not None:
        ox, oy = int(origin[0]), int(origin[1])
        L = max(20, img.shape[1] // 15)
        # Dirección del eje +X (rotada si hay xdir).
        if xdir is not None:
            ang = np.arctan2(xdir[1] - origin[1], xdir[0] - origin[0])
        else:
            ang = 0.0
        ux, uy = np.cos(ang), np.sin(ang)
        # +Y físico perpendicular, "hacia arriba" en el marco rotado.
        vx, vy = np.sin(ang), -np.cos(ang)
        cv2.arrowedLine(img, (ox, oy), (int(ox + L * ux), int(oy + L * uy)),
                        (230, 40, 40), th, tipLength=0.2)   # X rojo
        cv2.arrowedLine(img, (ox, oy), (int(ox + L * vx), int(oy + L * vy)),
                        (40, 90, 230), th, tipLength=0.2)   # Y azul
        cv2.circle(img, (ox, oy), max(3, r // 2), (255, 255, 0), -1)
        if xdir is not None:
            cv2.circle(img, (int(xdir[0]), int(xdir[1])), max(3, r // 2),
                       (255, 140, 0), -1)
    return img


def annotate_track(
    frame: np.ndarray,
    table: "TrackTable",
    current_idx: int,
) -> np.ndarray:
    """Dibuja los puntos ya marcados y la trayectoria sobre el frame (RGB).

    El punto del frame actual se resalta en amarillo; los demás en cian,
    unidos por una línea que muestra la trayectoria.
    """
    img = frame.copy()
    r = max(4, img.shape[1] // 220)
    th = max(1, img.shape[1] // 600)
    pts = sorted(table.points, key=lambda p: p.frame_idx)

    # Línea de trayectoria.
    for a, b in zip(pts, pts[1:]):
        cv2.line(
            img,
            (int(a.x_px), int(a.y_px)),
            (int(b.x_px), int(b.y_px)),
            (0, 200, 220),
            th,
        )
    # Puntos.
    for p in pts:
        color = (255, 230, 0) if p.frame_idx == current_idx else (0, 200, 220)
        cv2.circle(img, (int(p.x_px), int(p.y_px)), r, color, -1)
    return img


def draw_conical_overlay(img, xpx, ypx, ell, ref, n_segments):
    """Dibuja la elipse ajustada, sus ejes, focos y áreas barridas.

    ``ell`` es un EllipseFit (en píxeles); ``ref`` = (x, y) es el punto desde el
    cual se miden las áreas barridas (centro o un foco).
    """
    img = img.copy()
    th = max(2, img.shape[1] // 400)
    overlay = img.copy()
    rx, ry = int(ref[0]), int(ref[1])
    cx, cy = int(ell.cx), int(ell.cy)
    n = len(xpx) - 1
    colors = [(255, 80, 80), (80, 180, 255), (120, 255, 120), (255, 220, 80),
              (200, 120, 255), (255, 150, 60), (120, 220, 220), (230, 120, 180)]
    # Sectores (áreas barridas) desde el punto de referencia, por tramo.
    for gi, grp in enumerate(np.array_split(np.arange(n), n_segments)):
        col = colors[gi % len(colors)]
        for i in grp:
            tri = np.array([[rx, ry],
                            [int(xpx[i]), int(ypx[i])],
                            [int(xpx[i + 1]), int(ypx[i + 1])]], np.int32)
            cv2.fillPoly(overlay, [tri], col)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # Trayectoria medida.
    pts = np.array([[int(a), int(b)] for a, b in zip(xpx, ypx)], np.int32)
    cv2.polylines(img, [pts], False, (0, 230, 230), th)

    # Elipse ajustada (curva suave) + ejes mayor y menor.
    tt = np.linspace(0, 2 * np.pi, 200)
    ca, sa = np.cos(ell.theta), np.sin(ell.theta)
    ex = ell.cx + ell.a * np.cos(tt) * ca - ell.b * np.sin(tt) * sa
    ey = ell.cy + ell.a * np.cos(tt) * sa + ell.b * np.sin(tt) * ca
    epts = np.array([[int(a), int(b)] for a, b in zip(ex, ey)], np.int32)
    cv2.polylines(img, [epts], True, (255, 255, 255), th)
    # Eje mayor (rojo) y menor (azul).
    ux, uy = ca, sa
    vx, vy = -sa, ca
    cv2.line(img, (int(cx - ell.a * ux), int(cy - ell.a * uy)),
             (int(cx + ell.a * ux), int(cy + ell.a * uy)), (230, 60, 60), th)
    cv2.line(img, (int(cx - ell.b * vx), int(cy - ell.b * vy)),
             (int(cx + ell.b * vx), int(cy + ell.b * vy)), (60, 120, 230), th)

    msz = max(15, img.shape[1] // 45)
    cv2.drawMarker(img, (cx, cy), (255, 255, 0), cv2.MARKER_TILTED_CROSS, msz, th + 1)
    for f in ell.foci:
        cv2.drawMarker(img, (int(f[0]), int(f[1])), (255, 0, 255),
                       cv2.MARKER_STAR, msz, th + 1)
    # Resalta el punto de referencia de las áreas.
    cv2.circle(img, (rx, ry), max(5, msz // 3), (255, 255, 255), th)
    return img


def _invalidate_derived() -> None:
    """Borra todo resultado derivado en caché (al cambiar el intervalo)."""
    for k in ["_ghost_overlay", "export_ready", "export_zip", "export_csv",
              "fit_result", "kin_df"]:
        st.session_state.pop(k, None)


def _sync_interval(ps: int, pe: int) -> None:
    """Detecta cambios de intervalo, borra puntos fuera de rango e invalida caché.

    Preserva calibración, parámetros de detección y demás configuraciones que
    no dependen del rango de frames.
    """
    cur = (int(ps), int(pe))
    if st.session_state.get("_last_interval") != cur:
        st.session_state._last_interval = cur
        _invalidate_derived()
        # Elimina físicamente de la tabla los puntos fuera del nuevo rango.
        table = st.session_state.track
        table.points = [p for p in table.points
                        if cur[0] <= p.frame_idx <= cur[1]]


def render_navigator(reader: VideoReader, key_prefix: str) -> int:
    """Controles de navegación: saltos ±1/±5/±10, frame exacto y slider.

    Usa callbacks con `key` sincronizados a ``frame_idx`` (patrón robusto para
    varios controles que editan el mismo valor). Devuelve el frame actual.
    """
    n = reader.n_frames

    def step(delta: int) -> None:
        st.session_state.frame_idx = int(
            min(n - 1, max(0, st.session_state.frame_idx + delta)))

    specs = [("⏪ −10", -10), ("◀◀ −5", -5), ("◀ −1", -1),
             ("+1 ▶", 1), ("+5 ▶▶", 5), ("+10 ⏩", 10)]
    cols = st.columns(6)
    for col, (label, delta) in zip(cols, specs):
        col.button(label, width="stretch", key=f"{key_prefix}_step{delta}",
                   on_click=step, args=(delta,))

    # Frame exacto (número) + slider, ambos sincronizados con frame_idx.
    nk = f"{key_prefix}_num"
    st.session_state[nk] = int(st.session_state.frame_idx)

    def from_num() -> None:
        st.session_state.frame_idx = int(min(n - 1, max(0, st.session_state[nk])))

    c1, c2 = st.columns([1, 3])
    c1.number_input("Ir al frame", 0, n - 1, key=nk, on_change=from_num,
                    help="Escribe el número de frame para ir directo.")
    if n > 1:
        sk = f"{key_prefix}_sld"
        st.session_state[sk] = int(st.session_state.frame_idx)

        def from_sld() -> None:
            st.session_state.frame_idx = int(st.session_state[sk])

        with c2:
            st.slider("Frame", 0, n - 1, key=sk, on_change=from_sld,
                      label_visibility="collapsed")
    return int(st.session_state.frame_idx)


# --------------------------------------------------------------------------
# Barra lateral: carga de video y opciones
# --------------------------------------------------------------------------
def _load_video(uploaded) -> bool:
    """Guarda el video subido y crea el ``VideoReader``. Devuelve True si tuvo éxito."""
    suffix = "." + uploaded.name.split(".")[-1].lower()
    try:
        with st.spinner("Cargando y verificando el video…"):
            path = save_uploaded_to_temp(uploaded, suffix=suffix)
            reader = VideoReader(path)
    except ValueError as e:
        st.session_state.reader = None
        st.session_state["_load_error"] = str(e)
        return False
    except Exception as e:  # cualquier fallo inesperado de OpenCV
        st.session_state.reader = None
        st.session_state["_load_error"] = (
            f"Error inesperado al procesar el video: {e}"
        )
        return False

    if st.session_state.reader is not None:
        st.session_state.reader.release()
    st.session_state["_load_error"] = None
    st.session_state.video_path = str(path)
    st.session_state.reader = reader
    st.session_state.frame_idx = 0
    st.session_state.proc_start = 0
    st.session_state.proc_end = reader.n_frames - 1
    st.session_state._last_interval = (0, reader.n_frames - 1)
    # Reinicia calibración y tracking al cambiar de video.
    st.session_state.calib_p1 = None
    st.session_state.calib_p2 = None
    st.session_state.calib_origin = None
    st.session_state.calib_xdir = None
    st.session_state.calibration = None
    st.session_state.track = TrackTable()
    return True


with st.sidebar:
    st.header("🎥 Tracker UAI")
    st.caption("Análisis de video para laboratorios de física")

    uploaded = st.file_uploader(
        "Sube tu video",
        type=["mp4", "mov", "avi", "mkv", "m4v", "webm"],
        help="Formatos: mp4, mov, avi, mkv, m4v, webm. Recomendado: H.264. "
        "Graba con la cámara perpendicular al plano del movimiento.",
    )
    if uploaded is not None:
        if st.session_state.get("_uploaded_name") != uploaded.name:
            ok = _load_video(uploaded)
            st.session_state["_uploaded_name"] = uploaded.name if ok else None

    # Mensaje de error de carga (claro, sin congelar la app).
    if st.session_state.get("_load_error"):
        st.error("⚠️ " + st.session_state["_load_error"])

    st.divider()
    etapa = st.radio(
        "Etapa",
        ["1 · Video", "2 · Calibración", "3 · Tracking", "4 · Cinemática",
         "5 · Ajuste", "6 · Exportar", "📊 Gráficos (datos manuales)",
         "📐 Medición (foto)"],
        index=0,
    )

    st.divider()
    with st.expander("⚙️ Opciones avanzadas"):
        usar_fps_manual = st.checkbox(
            "Cámara lenta / fps de captura manual",
            value=st.session_state.fps_capture is not None,
            help="Actívalo si grabaste en cámara lenta (p. ej. 120 o 240 fps).",
        )
        if usar_fps_manual:
            fps_val = st.number_input(
                "fps real de captura",
                min_value=1.0,
                max_value=1000.0,
                value=float(st.session_state.fps_capture or 240.0),
                step=1.0,
            )
            st.session_state.fps_capture = fps_val
        else:
            st.session_state.fps_capture = None


# --------------------------------------------------------------------------
# Pestaña · Gráficos de datos experimentales (independiente del video)
# --------------------------------------------------------------------------
GRAPH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def _new_series(i: int) -> dict:
    """Crea una serie nueva con color automático y una tabla vacía."""
    return {
        "name": f"Serie {i + 1}",
        "color": GRAPH_COLORS[i % len(GRAPH_COLORS)],
        "visible": True,
        "data": pd.DataFrame({"x": [np.nan] * 4, "y": [np.nan] * 4}),
        "fit_type": "Ninguno",
        "fit_visible": True,
    }


def _equation_str(fit) -> str:
    """Ecuación legible del ajuste (lineal o cuadrático)."""
    p = fit.params()
    if fit.model == "linear":
        m = p["m"][0]
        b = p["b"][0]
        return f"y = {m:.4g}·x {'+' if b >= 0 else '−'} {abs(b):.4g}"
    if fit.model == "quadratic":
        a2, a1, a0 = p["a2"][0], p["a1"][0], p["a0"][0]
        return (f"y = {a2:.4g}·x² {'+' if a1 >= 0 else '−'} {abs(a1):.4g}·x "
                f"{'+' if a0 >= 0 else '−'} {abs(a0):.4g}")
    return "—"


G_TEO = 9.794          # valor de referencia de g en Santiago (m/s²)
MEAN_COS = 0.914       # <cos θ> medio para 10–35° (FIS101)

# Plantillas de laboratorio: prefijan título/ejes y fuerzan ajuste lineal.
LAB_TEMPLATES = {
    "Genérico": None,
    "FIS101 · g desde a vs sen θ": {
        "title": "Aceleración vs sen(θ)",
        "x": "sen(θ)", "y": "a (m/s²)",
    },
    "FIS201 · g desde T² vs L": {
        "title": "T² vs largo del péndulo",
        "x": "L (m)", "y": "T² (s²)",
    },
}


def _interpret_lab(template: str, fit) -> None:
    """Muestra la lectura física del ajuste lineal según la plantilla."""
    m, sm = fit.params()["m"]
    b, sb = fit.params()["b"]
    if template.startswith("FIS101"):
        g, sg = m, sm                       # pendiente = g
        mu = -b / (g * MEAN_COS) if g else float("nan")
        st.markdown(
            f"- **g (pendiente)** = {g:.4g} ± {sg:.2g} m/s²\n"
            f"- **Coef. de roce** μ_c = −b∕(g·⟨cos θ⟩) ≈ {mu:.4g} "
            f"(usando ⟨cos θ⟩ = {MEAN_COS})"
        )
    elif template.startswith("FIS201"):
        if m > 0:
            g = 4 * np.pi**2 / m            # pendiente = 4π²/g
            sg = 4 * np.pi**2 * sm / m**2
            d = b / m                        # desfase (largo efectivo extra)
            st.markdown(
                f"- **g** = 4π²∕pendiente = {g:.4g} ± {sg:.2g} m/s²\n"
                f"- **Desfase** d = ordenada∕pendiente = {d:.4g} m "
                f"(largo efectivo extra del péndulo)"
            )
            g, sg = g, sg
        else:
            st.warning("La pendiente debe ser positiva para despejar g.")
            return
    else:
        return
    diff = abs(g - G_TEO) / G_TEO * 100
    st.markdown(
        f"- **Comparación:** g_teórico = {G_TEO} m/s²  →  "
        f"diferencia = **{diff:.2f}\\%**  ·  R² = {fit.r_squared:.5f}"
    )


def stage_graphs() -> None:
    st.subheader("📊 Gráficos de datos experimentales")
    st.caption(
        "Ingresa tus datos a mano, elige un ajuste y descarga el gráfico para "
        "tu informe. No necesitas cargar ningún video. "
        "Ejemplo (péndulo): X = sin(θ), Y = aceleración."
    )

    if st.session_state.graph_series is None:
        st.session_state.graph_series = [_new_series(0)]
    series = st.session_state.graph_series

    # Plantilla de laboratorio (prefija ejes y activa ajuste lineal).
    tpl = st.selectbox(
        "Plantilla de laboratorio", list(LAB_TEMPLATES.keys()),
        index=list(LAB_TEMPLATES.keys()).index(st.session_state.graph_template),
        help="Elige tu laboratorio para etiquetar los ejes y leer g "
        "directamente del ajuste. «Genérico» para uso libre.")
    if tpl != st.session_state.graph_template:
        st.session_state.graph_template = tpl
        cfg = LAB_TEMPLATES[tpl]
        if cfg is not None:
            st.session_state.graph_title = cfg["title"]
            st.session_state.graph_xlabel = cfg["x"]
            st.session_state.graph_ylabel = cfg["y"]
            for s in series:
                s["fit_type"] = "Lineal"
        st.rerun()
    if LAB_TEMPLATES[tpl] is not None:
        ejemplo = ("Ingresa (sen θ, a) de tus 6 ángulos." if tpl.startswith("FIS101")
                   else "Ingresa (L en m, T² en s²) de tus 6 largos.")
        st.info(f"📋 {ejemplo}  El ajuste lineal entrega **g** automáticamente.")

    # Títulos del gráfico y de los ejes.
    c1, c2, c3 = st.columns(3)
    st.session_state.graph_title = c1.text_input(
        "Título del gráfico", st.session_state.graph_title)
    st.session_state.graph_xlabel = c2.text_input(
        "Nombre del eje X", st.session_state.graph_xlabel)
    st.session_state.graph_ylabel = c3.text_input(
        "Nombre del eje Y", st.session_state.graph_ylabel)

    # Agregar / quitar series.
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("➕ Agregar serie", width="stretch"):
        series.append(_new_series(len(series)))
        st.rerun()
    if b2.button("🗑️ Quitar última", width="stretch", disabled=len(series) <= 1):
        series.pop()
        st.rerun()

    # Editor por serie.
    for i, s in enumerate(series):
        with st.expander(f"✏️ {s['name']}", expanded=(len(series) == 1)):
            cc1, cc2, cc3 = st.columns([2, 1, 1])
            s["name"] = cc1.text_input("Nombre", s["name"], key=f"gname{i}")
            s["color"] = cc2.color_picker("Color", s["color"], key=f"gcolor{i}")
            s["visible"] = cc3.checkbox("Mostrar", s["visible"], key=f"gvis{i}")
            s["data"] = st.data_editor(
                s["data"], num_rows="dynamic", key=f"gdata{i}", width="stretch",
                column_config={
                    "x": st.column_config.NumberColumn("X"),
                    "y": st.column_config.NumberColumn("Y"),
                },
            )
            fc1, fc2 = st.columns(2)
            opciones_fit = ["Ninguno", "Lineal", "Cuadrático"]
            s["fit_type"] = fc1.selectbox(
                "Ajuste", opciones_fit,
                index=opciones_fit.index(s["fit_type"]), key=f"gfit{i}")
            s["fit_visible"] = fc2.checkbox(
                "Mostrar ajuste", s["fit_visible"], key=f"gfitvis{i}")

    # Construcción del gráfico y de los ajustes.
    fig = go.Figure()
    export_series = []
    resultados = []
    for s in series:
        if not s["visible"]:
            continue
        df = s["data"]
        x = pd.to_numeric(df["x"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["y"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=s["name"],
                                 marker=dict(color=s["color"], size=9)))
        exp = {"name": s["name"], "color": s["color"], "x": x, "y": y}

        if s["fit_type"] != "Ninguno" and s["fit_visible"]:
            model = "linear" if s["fit_type"] == "Lineal" else "quadratic"
            need = 2 if model == "linear" else 3
            if len(x) >= need:
                try:
                    fit = fit_model(model, x, y)
                    xd = np.linspace(float(np.min(x)), float(np.max(x)), 200)
                    yd = fit.predict(xd)
                    fig.add_trace(go.Scatter(
                        x=xd, y=yd, mode="lines", name=f"{s['name']} (ajuste)",
                        line=dict(color=s["color"], dash="dash")))
                    resultados.append((s["name"], _equation_str(fit),
                                       fit.r_squared, fit))
                    exp["fit_x"] = xd
                    exp["fit_y"] = yd
                    exp["fit_label"] = f"{s['name']} (ajuste)"
                except Exception as e:
                    st.warning(f"No se pudo ajustar «{s['name']}»: {e}")
            else:
                st.info(f"«{s['name']}»: se necesitan al menos {need} puntos "
                        f"para el ajuste {s['fit_type'].lower()}.")
        export_series.append(exp)

    fig.update_layout(
        title=st.session_state.graph_title,
        xaxis_title=st.session_state.graph_xlabel,
        yaxis_title=st.session_state.graph_ylabel,
        height=460, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, width="stretch")

    # Ecuaciones y R².
    if resultados:
        st.markdown("**Resultados de los ajustes**")
        for name, eq, r2, fit in resultados:
            st.markdown(f"- **{name}:**  {eq}   ·   **R² = {r2:.5f}**")
            partes = "  ·  ".join(
                f"{k} = {v:.4g} ± {s:.2g}" for k, (v, s) in fit.params().items())
            st.caption("  " + partes)

        # Interpretación física según la plantilla de laboratorio.
        tpl = st.session_state.graph_template
        if LAB_TEMPLATES.get(tpl):
            lineal = next((f for (_, _, _, f) in resultados if f.model == "linear"),
                          None)
            if lineal is not None:
                st.markdown("**🔬 Interpretación física (resultado del laboratorio)**")
                _interpret_lab(tpl, lineal)

    # Exportar imagen.
    if export_series:
        st.markdown("**Exportar gráfico**")
        try:
            png = export_mod.experimental_scatter_bytes(
                export_series, st.session_state.graph_title,
                st.session_state.graph_xlabel, st.session_state.graph_ylabel, "png")
            svg = export_mod.experimental_scatter_bytes(
                export_series, st.session_state.graph_title,
                st.session_state.graph_xlabel, st.session_state.graph_ylabel, "svg")
            e1, e2 = st.columns(2)
            e1.download_button("⬇️ Descargar PNG", png, "grafico_experimental.png",
                               "image/png", width="stretch")
            e2.download_button("⬇️ Descargar SVG", svg, "grafico_experimental.svg",
                               "image/svg+xml", width="stretch")
        except Exception as e:
            st.warning(f"No se pudo preparar la imagen: {e}")
    else:
        st.info("Escribe algunos datos X e Y en la tabla para ver el gráfico.")


# --------------------------------------------------------------------------
# Pestaña · Medición sobre foto — herramienta geométrica genérica
# (distancias, ángulos y áreas). No calcula magnitudes físicas: solo mide.
# --------------------------------------------------------------------------
MEAS_MODES = ["Calibrar (escala)", "Referencia horizontal", "Distancia",
              "Áng. con horizontal", "Ángulo (3 puntos)", "Área"]
MEAS_CAP = {"Calibrar (escala)": 2, "Referencia horizontal": 2, "Distancia": 2,
            "Áng. con horizontal": 2, "Ángulo (3 puntos)": 3, "Área": None}
MEAS_COLOR = {"Calibrar (escala)": (0, 200, 0), "Referencia horizontal": (120, 120, 120),
              "Distancia": (40, 90, 230), "Áng. con horizontal": (230, 120, 40),
              "Ángulo (3 puntos)": (255, 140, 0), "Área": (170, 90, 220)}


def _put_label(im, text, org, color):
    """Dibuja texto con fondo blanco para que se lea sobre la foto."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, im.shape[1] / 1400)
    tth = max(1, im.shape[1] // 700)
    (tw, thh), _ = cv2.getTextSize(text, font, fs, tth)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(im, (x - 4, y - thh - 6), (x + tw + 4, y + 4), (255, 255, 255), -1)
    cv2.putText(im, text, (x, y), font, fs, color, tth, cv2.LINE_AA)


def _meas_ref_angle(ref_pts):
    """Ángulo (rad) del eje horizontal definido por el usuario (0 = horizontal imagen)."""
    if len(ref_pts) < 2:
        return 0.0
    dx = ref_pts[1][0] - ref_pts[0][0]
    dy = ref_pts[1][1] - ref_pts[0][1]
    return float(np.arctan2(dy, dx))


def _draw_reference(im, ref_pts):
    """Dibuja el eje horizontal de referencia extendido a lo ancho de la imagen."""
    if len(ref_pts) < 2:
        return
    th = max(2, im.shape[1] // 500)
    p0 = np.array(ref_pts[0], float)
    p1 = np.array(ref_pts[1], float)
    d = p1 - p0
    if np.linalg.norm(d) == 0:
        return
    u = d / np.linalg.norm(d)
    diag = int(np.hypot(im.shape[0], im.shape[1]))
    a = (p0 - u * diag).astype(int)
    b = (p0 + u * diag).astype(int)
    cv2.line(im, tuple(a), tuple(b), (150, 150, 150), th, cv2.LINE_AA)
    for p in ref_pts:
        cv2.circle(im, (int(p[0]), int(p[1])), max(4, im.shape[1] // 220),
                   (120, 120, 120), -1)
    _put_label(im, "H (referencia)", (int(p0[0]) + 6, int(p0[1]) - 6), (90, 90, 90))


def _meas_draw_on(im, mode, pts, label=None):
    """Dibuja una medición (in situ) sobre ``im`` y, si se da, su valor."""
    col = MEAS_COLOR[mode]
    r = max(4, im.shape[1] // 220)
    th = max(2, im.shape[1] // 500)
    ipts = [(int(p[0]), int(p[1])) for p in pts]
    anchor = None

    if mode == "Área" and len(ipts) >= 3:
        overlay = im.copy()
        cv2.fillPoly(overlay, [np.array(ipts, np.int32)], col)
        im[:] = cv2.addWeighted(overlay, 0.3, im, 0.7, 0)
        cv2.polylines(im, [np.array(ipts, np.int32)], True, col, th)
        anchor = (int(np.mean([p[0] for p in ipts])),
                  int(np.mean([p[1] for p in ipts])))
    elif mode == "Ángulo (3 puntos)" and len(ipts) >= 2:
        for q in ipts[1:]:
            cv2.line(im, ipts[0], q, col, th)
        anchor = (ipts[0][0] + 8, ipts[0][1] - 8)
    elif mode == "Áng. con horizontal" and len(ipts) >= 2:
        cv2.line(im, ipts[0], ipts[1], col, th)
        anchor = ((ipts[0][0] + ipts[1][0]) // 2, (ipts[0][1] + ipts[1][1]) // 2 - 8)
    elif len(ipts) >= 2:
        for a, b in zip(ipts, ipts[1:]):
            cv2.line(im, a, b, col, th)
        anchor = ((ipts[0][0] + ipts[1][0]) // 2, (ipts[0][1] + ipts[1][1]) // 2 - 8)

    for i, p in enumerate(ipts):
        cv2.circle(im, p, r, col, -1)
        if mode == "Ángulo (3 puntos)" and i == 0:
            cv2.circle(im, p, r + 3, (255, 255, 0), th)

    if label and anchor is not None:
        _put_label(im, label, anchor, col)


def _meas_annotate(img, mode, pts, ref_pts, label=None):
    """Imagen con la referencia y la medición del modo actual (vista interactiva)."""
    im = img.copy()
    _draw_reference(im, ref_pts)
    _meas_draw_on(im, mode, pts, label)
    return im


def _meas_annotate_all(img, points_by_mode, scale, ref_angle):
    """Imagen con TODAS las mediciones y sus valores dibujados (para exportar)."""
    im = img.copy()
    _draw_reference(im, points_by_mode.get("Referencia horizontal", []))
    for m, pl in points_by_mode.items():
        if pl and m != "Referencia horizontal":
            _meas_draw_on(im, m, pl, _meas_value_label(m, pl, scale, ref_angle))
    return im


def _meas_distance_m(pts, scale):
    """Distancia en metros entre los dos puntos, o None."""
    if scale and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return float(np.hypot(x2 - x1, y2 - y1)) * scale
    return None


def _meas_angle_horizontal(pts, ref_angle=0.0):
    """Ángulo agudo (grados) del segmento respecto del eje horizontal (o de referencia)."""
    if len(pts) < 2:
        return None
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    d = np.degrees(np.arctan2(dy, dx) - ref_angle) % 180
    if d > 90:
        d = 180 - d
    return float(d)


def _meas_value_label(mode, pts, scale, ref_angle):
    """Texto del valor de la medición para dibujar sobre la imagen, o None."""
    if mode == "Distancia":
        d = _meas_distance_m(pts, scale)
        return f"{d*100:.1f} cm" if d is not None else None
    if mode == "Áng. con horizontal":
        a = _meas_angle_horizontal(pts, ref_angle)
        return f"{a:.1f}°" if a is not None else None
    if mode == "Ángulo (3 puntos)":
        a = _meas_angle_deg(pts)
        return f"{a:.1f}°" if a is not None else None
    if mode == "Área":
        A = _meas_area_m2(pts, scale)
        return f"{A*1e4:.1f} cm²" if A is not None else None
    return None


def _meas_angle_deg(pts):
    """Ángulo (grados) en el vértice pts[0] entre los brazos a pts[1] y pts[2]."""
    if len(pts) < 3:
        return None
    v = np.array(pts[0], float)
    a = np.array(pts[1], float) - v
    b = np.array(pts[2], float) - v
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    cosang = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
    return float(np.degrees(np.arccos(cosang)))


def _meas_area_m2(pts, scale):
    """Área del polígono (fórmula del zapatero) en m², o None."""
    if not scale or len(pts) < 3:
        return None
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    area_px = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area_px * scale**2


def stage_medicion() -> None:
    st.subheader("📐 Medición sobre una foto")
    st.caption(
        "Herramienta genérica para **medir sobre una imagen fija**: distancias, "
        "ángulos y áreas. Tú tomas las medidas; los cálculos físicos los haces "
        "en tu informe."
    )

    # --- Cargar imagen ---
    up = st.file_uploader("Sube una foto (jpg / png)", type=["jpg", "jpeg", "png"],
                          key="meas_uploader")
    if up is not None and st.session_state.get("_meas_name") != up.name:
        data = np.frombuffer(up.getvalue(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is not None:
            st.session_state.meas_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            st.session_state.meas_points = {}
            st.session_state.meas_scale = None
            st.session_state["_meas_name"] = up.name
            st.rerun()
    if st.session_state.reader is not None and st.button(
            "📸 Usar el frame actual del video en su lugar"):
        st.session_state.meas_image = st.session_state.reader.get_frame(
            int(st.session_state.frame_idx))
        st.session_state.meas_points = {}
        st.session_state.meas_scale = None
        st.rerun()

    img = st.session_state.meas_image
    if img is None:
        st.info("Sube una foto para empezar (o usa un frame del video).")
        return

    W = img.shape[1]
    # --- Herramienta activa ---
    st.session_state.meas_mode = st.radio(
        "Herramienta", MEAS_MODES,
        index=MEAS_MODES.index(st.session_state.meas_mode), horizontal=True)
    mode = st.session_state.meas_mode
    pts = st.session_state.meas_points.setdefault(mode, [])

    ayuda = {
        "Calibrar (escala)": "Marca 2 puntos de una distancia conocida (una regla) "
                             "y escríbela en metros. Necesario para distancias y áreas.",
        "Referencia horizontal": "Marca 2 puntos que definan tu **horizontal** "
                                 "(el eje x del sistema). Los ángulos se medirán "
                                 "respecto de esta línea.",
        "Distancia": "Marca 2 puntos; se mide la distancia entre ellos.",
        "Áng. con horizontal": "Marca 2 puntos a lo largo de la línea (p. ej. el plano "
                               "inclinado); se mide su ángulo con la horizontal.",
        "Ángulo (3 puntos)": "Marca 3 puntos: **primero el vértice**, luego los dos "
                             "extremos.",
        "Área": "Marca los vértices del polígono (mín. 3). Se cierra solo.",
    }[mode]
    st.info("👉 " + ayuda)

    # Eje horizontal de referencia (0 = horizontal de la imagen) y escala.
    ref_pts = st.session_state.meas_points.get("Referencia horizontal", [])
    ref_angle = _meas_ref_angle(ref_pts)
    scale = st.session_state.meas_scale

    disp_w = min(DISPLAY_WIDTH, W)
    cur_label = _meas_value_label(mode, pts, scale, ref_angle)
    annotated = fit_display(_meas_annotate(img, mode, pts, ref_pts, cur_label))
    col_img, col_ctrl = st.columns([3, 2])
    with col_img:
        value = streamlit_image_coordinates(
            annotated, width=disp_w, key="meas_clicker", cursor="crosshair")
        if value is not None and value != st.session_state._meas_last_click:
            st.session_state._meas_last_click = value
            pt = click_to_original(value, W)
            if pt is not None:
                cap = MEAS_CAP[mode]
                if cap is not None and len(pts) >= cap:
                    pts = [pt]                      # reinicia al superar el tope
                else:
                    pts = pts + [pt]
                st.session_state.meas_points[mode] = pts
                st.rerun()

    with col_ctrl:
        if mode == "Calibrar (escala)":
            st.session_state.meas_realdist = st.number_input(
                "Distancia real (m)", min_value=0.0001,
                value=float(st.session_state.meas_realdist), step=0.01,
                format="%.4f")
            if len(pts) == 2:
                try:
                    st.session_state.meas_scale = scale_from_two_points(
                        pts[0], pts[1], float(st.session_state.meas_realdist))
                except ValueError:
                    st.session_state.meas_scale = None
            sc = st.session_state.meas_scale
            st.metric("Escala", f"{sc:.5g} m/px" if sc else "—")

        if st.button("🧹 Limpiar puntos de esta medición", width="stretch"):
            st.session_state.meas_points[mode] = []
            st.rerun()

    # --- Resultado de la medición actual ---
    resultado, valor = None, None
    if mode == "Referencia horizontal":
        if len(ref_pts) == 2:
            rot = np.degrees(ref_angle)
            rot = (rot + 90) % 180 - 90
            st.success(f"Eje horizontal definido (girado {rot:.1f}° respecto de "
                       "la imagen). Los ángulos se medirán respecto de esta línea.")
        else:
            st.info("Marca 2 puntos para fijar tu eje horizontal, o déjalo sin "
                    "definir para usar la horizontal de la imagen.")
    elif mode == "Distancia":
        d = _meas_distance_m(pts, scale)
        if d is not None:
            resultado = f"**Distancia:** {d*100:.2f} cm  ({d:.4g} m)"
            valor = ("Distancia", f"{d*100:.3f} cm")
        elif scale is None:
            st.warning("Primero calibra la escala.")
    elif mode == "Áng. con horizontal":
        ang = _meas_angle_horizontal(pts, ref_angle)
        if ang is not None:
            ref_txt = " (respecto de tu referencia)" if len(ref_pts) == 2 else ""
            resultado = (f"**Ángulo con la horizontal{ref_txt}:** {ang:.2f}°  "
                         f"(con la vertical: {90 - ang:.2f}°)")
            valor = ("Ángulo↔horizontal", f"{ang:.2f}°")
    elif mode == "Ángulo (3 puntos)":
        ang = _meas_angle_deg(pts)
        if ang is not None:
            resultado = f"**Ángulo:** {ang:.2f}°"
            valor = ("Ángulo", f"{ang:.2f}°")
    elif mode == "Área":
        A = _meas_area_m2(pts, scale)
        if A is not None:
            resultado = f"**Área:** {A*1e4:.2f} cm²  ({A:.4g} m²)"
            valor = ("Área", f"{A*1e4:.3f} cm²")
        elif scale is None and len(pts) >= 3:
            st.warning("Primero calibra la escala.")

    if resultado:
        st.divider()
        c1, c2 = st.columns([3, 1])
        c1.markdown("### " + resultado)
        if c2.button("➕ Guardar", width="stretch"):
            st.session_state.meas_saved.append(
                {"Tipo": valor[0], "Valor": valor[1]})
            st.rerun()

    # --- Exportar la foto con las mediciones dibujadas ---
    st.divider()
    export_img = _meas_annotate_all(img, st.session_state.meas_points, scale, ref_angle)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(export_img, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button(
            "⬇️ Descargar foto con las mediciones (PNG)", buf.tobytes(),
            "foto_medida.png", "image/png", width="stretch",
            help="Descarga la imagen con todas las mediciones marcadas, para tu informe.")

    # --- Mediciones guardadas ---
    saved = st.session_state.meas_saved
    if saved:
        st.divider()
        st.markdown("**Mediciones guardadas**")
        st.dataframe(saved, width="stretch", hide_index=True)
        if st.button("🗑️ Borrar guardadas"):
            st.session_state.meas_saved = []
            st.rerun()


# --------------------------------------------------------------------------
# Guardas: hay video cargado
# --------------------------------------------------------------------------
st.title("Tracker UAI")
reader: Optional[VideoReader] = st.session_state.reader

# Módulos independientes: funcionan sin video cargado.
if etapa == "📊 Gráficos (datos manuales)":
    stage_graphs()
    st.stop()
if etapa == "📐 Medición (foto)":
    stage_medicion()
    st.stop()

if reader is None:
    st.info(
        "👈 Sube un video en la barra lateral para comenzar.\n\n"
        "**Recomendaciones de grabación:**\n"
        "- Cámara **perpendicular** al plano del movimiento y fija (trípode).\n"
        "- La regla/objeto de calibración debe estar **en el mismo plano** "
        "que el movimiento.\n"
        "- Buena iluminación y el objeto bien visible en el encuadre.\n\n"
        "💡 ¿Solo quieres graficar datos a mano? Ve a "
        "**📊 Gráficos (datos manuales)** — no necesita video."
    )
    st.stop()

meta = reader.metadata


# --------------------------------------------------------------------------
# Etapa 1 · Video
# --------------------------------------------------------------------------
def stage_video() -> None:
    st.warning(
        "⚠️ Esta versión **no corrige perspectiva**. Los resultados son válidos "
        "si la cámara está perpendicular al plano del movimiento y la calibración "
        "está en ese mismo plano.",
        icon="⚠️",
    )
    for w in reader.warnings:
        st.info("ℹ️ " + w)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", meta.n_frames)
    c2.metric("fps (archivo)", f"{meta.fps:.1f}")
    c3.metric("Resolución", f"{meta.width}×{meta.height}")
    c4.metric("Duración", f"{meta.duration_s:.2f} s")

    if st.session_state.fps_capture:
        st.caption(
            f"⏱️ Usando **fps de captura manual = {st.session_state.fps_capture:.0f}** "
            "para calcular el tiempo (cámara lenta)."
        )

    # Precarga el video una vez → navegación fluida y sin crash de seeks.
    _ensure_full_preload()

    # Reproductor original, opcional (colapsado para no ocupar pantalla).
    with st.expander("▶️ Ver reproductor del video"):
        st.video(st.session_state.video_path)

    st.subheader("Selección de frames")
    idx = render_navigator(reader, "video")
    t = reader.frame_time(idx, fps_capture=st.session_state.fps_capture)
    table: TrackTable = st.session_state.track
    marcado = table.get(idx)

    frame = reader.get_frame(idx)
    # Muestra las detecciones superpuestas para contar frames con precisión.
    disp = annotate_track(frame, table, idx) if len(table) > 0 else frame
    estado = "● con punto marcado" if marcado is not None else "○ sin punto"
    icol, mcol = st.columns([3, 1])
    with icol:
        st.image(fit_display(disp),
                 caption=f"Frame {idx} / {meta.n_frames - 1}  ·  t = {t:.3f} s  ·  "
                 f"{estado}")
    with mcol:
        st.metric("Frame", idx)
        st.metric("Puntos marcados", len(table))
        st.caption("Los puntos ya detectados se ven sobre la imagen, para "
                   "contar y elegir el frame exacto de inicio y fin.")

    # Selección del intervalo de procesamiento.
    st.subheader("Intervalo de procesamiento")
    bcol1, bcol2 = st.columns(2)
    if bcol1.button("Usar frame actual como INICIO", width="stretch",
                    icon=":material/first_page:"):
        st.session_state.proc_start = idx
        if st.session_state.proc_end < idx:
            st.session_state.proc_end = idx
        st.rerun()
    if bcol2.button("Usar frame actual como FINAL", width="stretch",
                    icon=":material/last_page:"):
        st.session_state.proc_end = idx
        if st.session_state.proc_start > idx:
            st.session_state.proc_start = idx
        st.rerun()

    if meta.n_frames > 1:
        lo, hi = st.slider(
            "Frames inicial y final",
            0, meta.n_frames - 1,
            (int(st.session_state.proc_start), int(st.session_state.proc_end)),
        )
        st.session_state.proc_start, st.session_state.proc_end = lo, hi

    ps, pe = st.session_state.proc_start, st.session_state.proc_end
    # Si el intervalo cambió, invalida todo lo derivado del rango anterior.
    prev = st.session_state.get("_last_interval")
    _sync_interval(ps, pe)
    if prev is not None and prev != (int(ps), int(pe)):
        st.warning(
            "Intervalo modificado: se **borraron los puntos fuera del nuevo "
            "rango** y se invalidaron los resultados anteriores (cinemática, "
            "ajustes, gráficos exportados y vista de detecciones). Si detectaste "
            "de forma automática, vuelve a ejecutar el seguimiento en la Etapa 3.",
            icon=":material/sync:",
        )
    t0 = reader.frame_time(ps, fps_capture=st.session_state.fps_capture)
    t1 = reader.frame_time(pe, fps_capture=st.session_state.fps_capture)
    st.success(
        f"Intervalo: frame {ps} → {pe}  ({pe - ps + 1} frames)  ·  "
        f"t = {t0:.3f} s → {t1:.3f} s"
    )


# --------------------------------------------------------------------------
# Etapa 2 · Calibración
# --------------------------------------------------------------------------
def stage_calibration() -> None:
    st.subheader("Calibración espacial")
    _ensure_full_preload()
    st.markdown(
        "Marca en el video: **dos puntos** de una distancia conocida (para la "
        "escala) y el **origen** de coordenadas. El eje Y físico apunta hacia "
        "**arriba**."
    )

    # 1) Elegir frame donde se vea bien la regla de calibración.
    with st.expander("📍 Elegir frame de calibración", expanded=False):
        st.caption(
            "Navega hasta un frame donde la regla/objeto de distancia conocida "
            "se vea claramente."
        )
        render_navigator(reader, "calib")

    idx = int(st.session_state.frame_idx)
    frame = reader.get_frame(idx)

    # Ejes rotados (orientación libre del sistema de referencia).
    st.session_state.calib_use_rotation = st.checkbox(
        "Ejes con orientación libre (marcar dirección +X)",
        value=st.session_state.calib_use_rotation,
        help="Actívalo si el eje X no es horizontal (p. ej. plano inclinado). "
        "Marca un punto en la dirección deseada para +X desde el origen.",
    )

    # 2) ¿Qué se marca con el próximo clic?
    opciones = ["Punto 1 (escala)", "Punto 2 (escala)", "Origen"]
    if st.session_state.calib_use_rotation:
        opciones.append("Dirección +X")
    if st.session_state.calib_target not in opciones:
        st.session_state.calib_target = opciones[0]
    st.session_state.calib_target = st.radio(
        "¿Qué vas a marcar con el próximo clic?",
        opciones,
        horizontal=True,
        index=opciones.index(st.session_state.calib_target),
    )

    # 3) Imagen interactiva con marcadores dibujados.
    annotated = annotate_frame(
        frame,
        st.session_state.calib_p1,
        st.session_state.calib_p2,
        st.session_state.calib_origin,
        st.session_state.calib_xdir if st.session_state.calib_use_rotation else None,
    )
    disp_w = min(DISPLAY_WIDTH, meta.width)
    col_img, col_ctrl = st.columns([3, 2])

    with col_img:
        value = streamlit_image_coordinates(
            annotated,
            width=disp_w,
            key="calib_clicker",
            cursor="crosshair",
        )
        # Procesa un clic nuevo (distinto al anterior).
        if value is not None and value != st.session_state._calib_last_click:
            st.session_state._calib_last_click = value
            pt = click_to_original(value, meta.width)
            if pt is not None:
                target = st.session_state.calib_target
                if target == "Punto 1 (escala)":
                    st.session_state.calib_p1 = pt
                    st.session_state.calib_target = "Punto 2 (escala)"
                elif target == "Punto 2 (escala)":
                    st.session_state.calib_p2 = pt
                    st.session_state.calib_target = "Origen"
                elif target == "Origen":
                    st.session_state.calib_origin = pt
                    if st.session_state.calib_use_rotation:
                        st.session_state.calib_target = "Dirección +X"
                else:
                    st.session_state.calib_xdir = pt
                st.rerun()

    with col_ctrl:
        st.markdown("**Puntos marcados**")
        p1 = st.session_state.calib_p1
        p2 = st.session_state.calib_p2
        org = st.session_state.calib_origin
        st.write(f"🟢 Punto 1: {_fmt_pt(p1)}")
        st.write(f"🟢 Punto 2: {_fmt_pt(p2)}")
        st.write(f"🟡 Origen: {_fmt_pt(org)}")
        if st.session_state.calib_use_rotation:
            st.write(f"🟠 Dirección +X: {_fmt_pt(st.session_state.calib_xdir)}")

        st.session_state.calib_real_dist = st.number_input(
            "Distancia real entre punto 1 y 2 (m)",
            min_value=0.0001,
            value=float(st.session_state.calib_real_dist),
            step=0.01,
            format="%.4f",
        )

        cbtn1, cbtn2 = st.columns(2)
        if cbtn1.button("✅ Aplicar calibración", width="stretch"):
            _apply_calibration()
        if cbtn2.button("🔄 Reiniciar", width="stretch"):
            st.session_state.calib_p1 = None
            st.session_state.calib_p2 = None
            st.session_state.calib_origin = None
            st.session_state.calib_xdir = None
            st.session_state.calibration = None
            st.session_state.calib_target = "Punto 1 (escala)"
            st.rerun()

    # 4) Estado de la calibración.
    calib = st.session_state.calibration
    if calib is not None:
        st.success(f"Calibración aplicada — {calib.summary()}")
        # Ejemplo de conversión: esquina donde está el puntero no; mostramos el origen.
        st.caption(
            "Verificación: la distancia entre los dos puntos marcados equivale "
            f"a {st.session_state.calib_real_dist:.4g} m."
        )
    else:
        st.info(
            "Marca los 3 elementos y pulsa **Aplicar calibración**. "
            "Puedes re-marcar cualquier punto seleccionándolo arriba y "
            "haciendo clic de nuevo."
        )


def _fmt_pt(pt: Optional[Tuple[float, float]]) -> str:
    """Formatea un punto (x, y) en píxeles para mostrarlo, o '—' si falta."""
    if pt is None:
        return "—"
    return f"({pt[0]:.0f}, {pt[1]:.0f}) px"


def _apply_calibration() -> None:
    """Valida los puntos y construye el objeto ``Calibration`` en la sesión."""
    p1 = st.session_state.calib_p1
    p2 = st.session_state.calib_p2
    org = st.session_state.calib_origin
    if p1 is None or p2 is None:
        st.error("Faltan los dos puntos de la escala.")
        return
    if org is None:
        st.error("Falta marcar el origen.")
        return
    try:
        m_per_px = scale_from_two_points(
            p1, p2, float(st.session_state.calib_real_dist)
        )
    except ValueError as e:
        st.error(str(e))
        return
    angle = 0.0
    if st.session_state.calib_use_rotation:
        xdir = st.session_state.calib_xdir
        if xdir is None:
            st.error("Marca el punto de dirección +X o desactiva los ejes rotados.")
            return
        angle = angle_from_points(org, xdir)
    st.session_state.calibration = Calibration(
        m_per_px=m_per_px, origin_px=org, angle_rad=angle
    )
    st.rerun()


# --------------------------------------------------------------------------
# Etapa 3 · Tracking manual
# --------------------------------------------------------------------------
def box_to_original(value: dict, orig_width: int):
    """Convierte un recuadro (arrastre) del componente a (x, y, w, h) en px originales."""
    if not value or "x1" not in value or not value.get("width"):
        return None
    scale = orig_width / float(value["width"])
    x1, y1 = value["x1"] * scale, value["y1"] * scale
    x2, y2 = value["x2"] * scale, value["y2"] * scale
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    return (x, y, w, h)


def stage_tracking() -> None:
    st.subheader("Tracking")

    calib = st.session_state.calibration
    if calib is None:
        st.warning(
            "Aún no aplicaste la calibración. Puedes marcar puntos igual, pero "
            "las columnas x_m, y_m quedarán vacías hasta calibrar (Etapa 2)."
        )

    st.session_state.track_mode = st.radio(
        "Modo de tracking",
        ["Manual", "Automático"],
        horizontal=True,
        index=["Manual", "Automático"].index(st.session_state.track_mode),
    )

    if st.session_state.track_mode == "Manual":
        _tracking_manual()
    else:
        _tracking_auto()

    _render_track_table(calib)


def _ensure_full_preload() -> None:
    """Precarga TODO el video en caché una sola vez (por video).

    Al leer el video completo de forma secuencial (estable) y servir después
    desde el caché, **ninguna navegación hace seek**, lo que elimina el crash
    de FFmpeg (`async_lock`) con .mov/HEVC al saltar muchos frames.
    """
    key = f"_preloaded_full::{st.session_state.video_path}"
    if st.session_state.get(key):
        return
    n = reader.n_frames
    end = min(n - 1, reader._cache_max - 1)
    with st.spinner("Preparando el video para navegación fluida (solo la primera vez)…"):
        bar = st.progress(0.0)
        try:
            reader.preload_range(0, end, 1,
                                 progress_cb=lambda f: bar.progress(min(1.0, f)))
        finally:
            bar.empty()
    st.session_state[key] = True
    if n - 1 > end:
        st.warning(
            f"El video tiene {n} frames; se precargaron los primeros {end + 1}. "
            "Para videos muy largos, recórtalos o baja la resolución para una "
            "navegación 100% estable.")


def _correction_ui(key_prefix: str, advance: bool) -> None:
    """UI compartida: navegar, marcar/corregir con clic, borrar puntos.

    Se usa tanto para el marcado manual (``advance=True``, con auto-avance)
    como para la revisión/corrección tras el tracking automático
    (``advance=False``).
    """
    # Precarga (una vez) para que la navegación lea del caché, sin seeks.
    _ensure_full_preload()
    idx = render_navigator(reader, key_prefix)
    frame = reader.get_frame(idx)
    t = reader.frame_time(idx, fps_capture=st.session_state.fps_capture)
    table: TrackTable = st.session_state.track
    annotated = annotate_track(frame, table, idx)
    disp_w = min(DISPLAY_WIDTH, meta.width)
    last_key = f"_{key_prefix}_last_click"

    col_img, col_info = st.columns([3, 2])
    with col_img:
        value = streamlit_image_coordinates(
            fit_display(annotated), width=disp_w,
            key=f"{key_prefix}_clicker", cursor="crosshair",
        )
        if value is not None and value != st.session_state.get(last_key):
            st.session_state[last_key] = value
            pt = click_to_original(value, meta.width)
            if pt is not None:
                table.add_or_update(idx, t, pt[0], pt[1])
                if advance:
                    st.session_state.frame_idx = min(
                        meta.n_frames - 1, idx + int(st.session_state.track_step)
                    )
                st.rerun()

    with col_info:
        marcado = table.get(idx)
        st.metric("Frame actual", f"{idx}")
        st.metric("t", f"{t:.3f} s")
        st.metric("Puntos marcados", len(table))
        if marcado is not None:
            st.caption(
                f"Este frame tiene un punto en "
                f"({marcado.x_px:.0f}, {marcado.y_px:.0f}) px. "
                "Haz clic para corregir su posición."
            )
        else:
            st.caption("Este frame no tiene punto. Haz clic para agregar uno.")
        if st.button("🗑️ Borrar punto de este frame", width="stretch",
                     key=f"{key_prefix}_del"):
            table.delete(idx)
            st.rerun()


def _tracking_manual() -> None:
    """Marcado manual clic a clic con auto-avance."""
    st.markdown(
        "Haz **clic sobre el objeto** en cada frame. Tras cada clic, el video "
        "avanza automáticamente el número de frames que elijas abajo."
    )
    cfg1, _ = st.columns([1, 3])
    st.session_state.track_step = cfg1.number_input(
        "Avanzar cada N frames",
        min_value=1,
        max_value=max(1, meta.n_frames - 1),
        value=int(st.session_state.track_step),
        step=1,
        help="Marca en cada frame (N=1) o cada N frames para movimientos lentos.",
    )
    _correction_ui("track", advance=True)


def _tracking_auto() -> None:
    """Seguimiento automático por plantilla, color o CSRT."""
    metodos = ["Plantilla", "Color"]
    if csrt_available():
        metodos.append("CSRT")
    else:
        st.caption("💡 CSRT no disponible (instala opencv-contrib-python para habilitarlo).")
    metodo = st.selectbox("Método", metodos)

    # Rango de frames y paso (por defecto, el intervalo elegido en Etapa 1).
    st.caption("Rango inicializado con el **intervalo de procesamiento** (Etapa 1).")
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Frame inicial", 0, meta.n_frames - 1,
                            int(st.session_state.proc_start))
    end = c2.number_input("Frame final", 0, meta.n_frames - 1,
                          int(st.session_state.proc_end))
    step = c3.number_input("Paso (cada N frames)", 1, max(1, meta.n_frames - 1),
                           int(st.session_state.track_step))
    if end <= start:
        st.warning("El frame final debe ser mayor que el inicial.")
        return

    frame = reader.get_frame(int(start))
    disp_w = min(DISPLAY_WIDTH, meta.width)
    table: TrackTable = st.session_state.track

    if metodo in ("Plantilla", "CSRT"):
        st.markdown("**Arrastra un recuadro** sobre el objeto en el frame inicial.")
        disp_frame = frame.copy()
        roi_prev = st.session_state.auto_roi
        if roi_prev:
            rx, ry, rw, rh = (int(v) for v in roi_prev)
            cv2.rectangle(disp_frame, (rx, ry), (rx + rw, ry + rh),
                          (0, 220, 0), max(2, meta.width // 400))
        value = streamlit_image_coordinates(
            disp_frame, width=disp_w, key="auto_box_clicker",
            cursor="crosshair", click_and_drag=True,
        )
        if value is not None and value != st.session_state._auto_last_click:
            st.session_state._auto_last_click = value
            roi = box_to_original(value, meta.width)
            if roi and roi[2] > 3 and roi[3] > 3:
                st.session_state.auto_roi = roi
                st.rerun()
        roi = st.session_state.auto_roi
        if roi:
            st.success(f"Recuadro: x={roi[0]:.0f}, y={roi[1]:.0f}, "
                       f"w={roi[2]:.0f}, h={roi[3]:.0f} px")
        else:
            st.info("Aún no defines el recuadro del objeto.")

        if st.button("▶️ Ejecutar seguimiento", type="primary",
                     disabled=roi is None, width="stretch"):
            _run_autotrack(metodo, int(start), int(end), int(step))

    else:  # Color
        st.markdown("**Haz clic sobre el objeto** para tomar su color.")
        h_tol = st.slider("Tolerancia de tono (H)", 2, 40, 12)
        cc1, cc2 = st.columns(2)
        s_min = cc1.slider("Saturación mínima (S)", 0, 255, 60)
        v_min = cc2.slider("Brillo mínimo (V)", 0, 255, 60)
        value = streamlit_image_coordinates(
            frame, width=disp_w, key="auto_color_clicker", cursor="crosshair",
        )
        if value is not None and value != st.session_state._auto_last_click:
            st.session_state._auto_last_click = value
            pt = click_to_original(value, meta.width)
            if pt is not None:
                st.session_state.auto_hsv = hsv_range_from_pixel(
                    frame, int(pt[0]), int(pt[1]), h_tol, s_min, v_min
                )
                st.rerun()
        hsv = st.session_state.auto_hsv
        if hsv is not None:
            lower, upper = hsv
            # Vista previa de la máscara.
            mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV), lower, upper)
            st.image(mask, caption="Máscara de color (blanco = detectado)",
                     width=disp_w)
            st.caption(f"HSV lower={lower.tolist()}  upper={upper.tolist()}")
        if st.button("▶️ Ejecutar seguimiento", type="primary",
                     disabled=hsv is None, width="stretch"):
            _run_autotrack("Color", int(start), int(end), int(step))

    # --- Revisión y corrección de las detecciones (tras el tracking) ---
    table = st.session_state.track
    if len(table) > 0:
        st.divider()
        st.subheader("Revisión de detecciones")
        st.caption(
            "Revisa la calidad del seguimiento antes de seguir. La imagen "
            "muestra todas las detecciones superpuestas; abajo puedes navegar "
            "frame por frame y corregir puntos mal detectados."
        )
        # El overlay "fantasma" es caro (mediana de frames): se cachea y solo
        # se regenera con un botón, para que navegar al corregir sea fluido.
        if st.button("🔄 Actualizar vista superpuesta", key="refresh_ghost"):
            st.session_state.pop("_ghost_overlay", None)
        if "_ghost_overlay" not in st.session_state:
            frames = [int(f) for f in sorted(p.frame_idx for p in table.points)]
            try:
                with st.spinner("Generando vista de detecciones…"):
                    st.session_state["_ghost_overlay"] = reader.ghost_overlay(frames)
            except Exception as e:
                st.session_state["_ghost_overlay"] = None
                st.caption(f"No se pudo generar la vista superpuesta: {e}")
        ghost = st.session_state.get("_ghost_overlay")
        if ghost is not None:
            st.image(annotate_track(ghost, table, -1), width=DISPLAY_WIDTH,
                     caption="Todas las detecciones superpuestas (cian). "
                     "Usa 'Actualizar' tras corregir.")

        st.markdown("**Corrección frame por frame**")
        _correction_ui("review", advance=False)


def _run_autotrack(metodo: str, start: int, end: int, step: int) -> None:
    """Ejecuta el método elegido, con vista previa en vivo, y vuelca los puntos."""
    table: TrackTable = st.session_state.track
    disp_w = min(DISPLAY_WIDTH, meta.width)
    # Precarga (una vez) para que el tracking lea del caché (evita seeks/crash).
    _ensure_full_preload()
    bar = st.progress(0.0, text="Siguiendo objeto…")
    preview = st.empty()
    fps = st.session_state.fps_capture

    cb = lambda f: bar.progress(min(1.0, f), text="Siguiendo objeto…")
    # Vista previa en vivo (throttle: no en cada frame, para no ralentizar).
    counter = {"n": 0}

    def pv(frame_rgb, x, y, fi):
        counter["n"] += 1
        if counter["n"] % 3 != 0:
            return
        img = frame_rgb.copy()
        rad = max(6, img.shape[1] // 120)
        cv2.circle(img, (int(x), int(y)), rad, (255, 230, 0), 3)
        cv2.drawMarker(img, (int(x), int(y)), (255, 0, 0),
                       cv2.MARKER_CROSS, rad * 2, 2)
        preview.image(img, width=disp_w, caption=f"Detectando… frame {fi}")

    try:
        if metodo == "Plantilla":
            pts = track_template(reader, st.session_state.auto_roi, start, end,
                                 step, fps, progress_cb=cb, preview_cb=pv)
        elif metodo == "CSRT":
            pts = track_csrt(reader, st.session_state.auto_roi, start, end,
                             step, fps, progress_cb=cb, preview_cb=pv)
        else:
            lower, upper = st.session_state.auto_hsv
            pts = track_color(reader, lower, upper, start, end, step, fps,
                              progress_cb=cb, preview_cb=pv)
    except Exception as e:
        bar.empty()
        preview.empty()
        st.error(f"Error en el seguimiento: {e}")
        return
    bar.empty()
    preview.empty()
    for (fi, t, x, y) in pts:
        table.add_or_update(fi, t, x, y)
    if not pts:
        st.warning("No se detectó el objeto en ningún frame. Ajusta los parámetros.")
    else:
        st.success(f"Seguimiento completo: {len(pts)} puntos agregados.")
        st.rerun()


def _render_track_table(calib) -> None:
    """Tabla editable compartida por los modos manual y automático."""
    table: TrackTable = st.session_state.track
    st.divider()
    st.subheader("Tabla de datos")
    if st.button("🧹 Borrar todos los puntos", key="clear_all_table"):
        table.clear()
        st.rerun()
    df = table.to_dataframe(calib)
    if df.empty:
        st.info("Aún no has marcado puntos.")
        return

    st.caption(
        "Puedes **corregir** x_px / y_px directamente en la tabla o **eliminar** "
        "filas (ícono de la izquierda). Las columnas x_m, y_m y t se recalculan solas."
    )
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        disabled=["frame", "t", "x_m", "y_m"],
        column_config={
            "frame": st.column_config.NumberColumn("frame", format="%d"),
            "t": st.column_config.NumberColumn("t (s)", format="%.3f"),
            "x_px": st.column_config.NumberColumn("x_px", format="%.1f"),
            "y_px": st.column_config.NumberColumn("y_px", format="%.1f"),
            "x_m": st.column_config.NumberColumn("x (m)", format="%.4f"),
            "y_m": st.column_config.NumberColumn("y (m)", format="%.4f"),
        },
        key="track_editor",
    )
    # Si el usuario editó/eliminó filas, reconstruye la tabla autoritativa.
    if not _same_editable(df, edited):
        st.session_state.track = from_dataframe(edited)
        st.rerun()


def _same_editable(df_a, df_b) -> bool:
    """Compara solo las columnas editables (x_px, y_px) y el número de filas."""
    if len(df_a) != len(df_b):
        return False
    cols = ["frame", "x_px", "y_px"]
    try:
        a = df_a[cols].reset_index(drop=True)
        b = df_b[cols].reset_index(drop=True)
        return a.equals(b)
    except Exception:
        return False


# --------------------------------------------------------------------------
# Etapa 4 · Cinemática
# --------------------------------------------------------------------------
def _line_fig(x, series: list[tuple], title: str, xlab: str, ylab: str) -> go.Figure:
    """Construye una figura Plotly de líneas+puntos. ``series`` = [(y, nombre), ...]."""
    fig = go.Figure()
    for y, name in series:
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines+markers", name=name)
        )
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def stage_kinematics() -> None:
    st.subheader("Cinemática")

    calib = st.session_state.calibration
    if calib is None:
        st.warning("Necesitas aplicar la calibración (Etapa 2) para calcular en metros.")
        return
    n_int = _points_in_interval()
    if n_int < 2:
        st.info(
            "Marca al menos 2 puntos **dentro del intervalo de procesamiento** "
            "(Etapa 3) para calcular velocidad."
        )
        return

    # Controles de suavizado.
    c1, c2, c3 = st.columns([1.4, 1, 1])
    st.session_state.kin_smooth = c1.checkbox(
        "Suavizar posición (Savitzky-Golay)",
        value=st.session_state.kin_smooth,
        help="Reduce el ruido de marcado, sobre todo en la aceleración.",
    )
    st.session_state.kin_window = c2.number_input(
        "Ventana (impar)",
        min_value=3,
        max_value=max(3, n_int),
        value=int(st.session_state.kin_window),
        step=2,
        disabled=not st.session_state.kin_smooth,
    )
    st.session_state.kin_poly = c3.selectbox(
        "Orden polinomio",
        options=[2, 3],
        index=[2, 3].index(int(st.session_state.kin_poly)),
        disabled=not st.session_state.kin_smooth,
    )

    st.caption(
        f"Analizando **{n_int} puntos** dentro del intervalo "
        f"[{st.session_state.proc_start}, {st.session_state.proc_end}]."
    )
    kdf = _get_kdf()
    # Guarda el resultado para reutilizarlo en export/ajuste.
    st.session_state.kin_df = kdf

    t = kdf["t"].to_numpy()
    xpos = kdf["x_s"].to_numpy() if "x_s" in kdf else kdf["x_m"].to_numpy()
    ypos = kdf["y_s"].to_numpy() if "y_s" in kdf else kdf["y_m"].to_numpy()

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            _line_fig(t, [(xpos, "x(t)"), (ypos, "y(t)")],
                      "Posición vs tiempo", "t (s)", "posición (m)"),
            width="stretch",
        )
        st.plotly_chart(
            _line_fig(t, [(kdf["vx"], "vx(t)"), (kdf["vy"], "vy(t)")],
                      "Velocidad vs tiempo", "t (s)", "velocidad (m/s)"),
            width="stretch",
        )
    with g2:
        # Trayectoria x-y con aspecto igual.
        traj = go.Figure()
        traj.add_trace(
            go.Scatter(x=xpos, y=ypos, mode="lines+markers", name="trayectoria")
        )
        traj.update_layout(
            title="Trayectoria x-y",
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            margin=dict(l=10, r=10, t=40, b=10),
            height=340,
        )
        traj.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(traj, width="stretch")

        st.plotly_chart(
            _line_fig(t, [(kdf["ax"], "ax(t)"), (kdf["ay"], "ay(t)")],
                      "Aceleración vs tiempo", "t (s)", "aceleración (m/s²)"),
            width="stretch",
        )

    with st.expander("📊 Tabla con derivados"):
        st.dataframe(kdf, width="stretch")

    st.caption(
        "Las derivadas usan diferencias finitas centradas (respetan el paso de "
        "tiempo no uniforme). Los extremos son menos precisos por ser de un solo lado."
    )


# --------------------------------------------------------------------------
# Etapa 5 · Ajuste de modelos + presets
# --------------------------------------------------------------------------
# Etiquetas legibles para el ajuste genérico.
MODEL_LABELS = {
    "Lineal (MRU)": "linear",
    "Parabólico (caída / tiro)": "quadratic",
    "Sinusoidal (MAS / péndulo)": "sine",
    "Sinusoidal amortiguado": "damped_sine",
    "Exponencial amortiguado": "damped_exp",
}
SIGNAL_LABELS = {
    "x (m)": "x_m",
    "y (m)": "y_m",
    "rapidez |v| (m/s)": "speed",
    "vx (m/s)": "vx",
    "vy (m/s)": "vy",
}


def _fmt(v: float, s: float) -> str:
    """Formatea 'valor ± sigma'."""
    return f"{v:.4g} ± {s:.2g}"


def _params_table(fit) -> None:
    """Muestra los parámetros del ajuste con incertidumbre y R²."""
    rows = [{"parámetro": n, "valor": v, "± incertidumbre": s}
            for n, (v, s) in fit.params().items()]
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f"R² = {fit.r_squared:.5f}")


def _plot_fit(x, y, fit, xlab: str, ylab: str, title: str) -> None:
    """Grafica datos + curva ajustada."""
    xd = np.linspace(np.nanmin(x), np.nanmax(x), 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="datos"))
    fig.add_trace(go.Scatter(x=xd, y=fit.predict(xd), mode="lines", name="ajuste"))
    fig.update_layout(
        title=title, xaxis_title=xlab, yaxis_title=ylab,
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")


def _get_kdf():
    """Cinemática de los puntos **dentro del intervalo** de procesamiento.

    Filtra por [proc_start, proc_end] para que cambiar el intervalo se
    propague automáticamente a cinemática, ajustes y exportación.
    """
    calib = st.session_state.calibration
    table: TrackTable = st.session_state.track
    df = table.to_dataframe(calib)
    ps, pe = st.session_state.proc_start, st.session_state.proc_end
    if not df.empty:
        df = df[(df["frame"] >= ps) & (df["frame"] <= pe)].reset_index(drop=True)
    return kinematics_dataframe(
        df,
        smooth=st.session_state.kin_smooth,
        window=int(st.session_state.kin_window),
        polyorder=int(st.session_state.kin_poly),
    )


def _points_in_interval() -> int:
    """Número de puntos marcados dentro del intervalo de procesamiento."""
    ps, pe = st.session_state.proc_start, st.session_state.proc_end
    return sum(1 for p in st.session_state.track.points if ps <= p.frame_idx <= pe)


def _signal_picker(kdf, default: str, key: str):
    """Selector de señal; devuelve (etiqueta, arreglo)."""
    labels = list(SIGNAL_LABELS.keys())
    default_label = next(
        (l for l, c in SIGNAL_LABELS.items() if c == default), labels[0]
    )
    label = st.selectbox("Señal a analizar", labels,
                         index=labels.index(default_label), key=key)
    col = SIGNAL_LABELS[label]
    return label, kdf[col].to_numpy()


def stage_ajuste() -> None:
    st.subheader("Ajuste de modelos")

    calib = st.session_state.calibration
    table: TrackTable = st.session_state.track
    if calib is None or _points_in_interval() < 2:
        st.info("Necesitas calibración (Etapa 2) y al menos 2 puntos dentro del "
                "intervalo (Etapa 3).")
        return

    labels = preset_labels()
    st.session_state.preset_idx = st.selectbox(
        "Preset de laboratorio",
        range(len(labels)),
        format_func=lambda i: labels[i],
        index=int(st.session_state.preset_idx),
    )
    preset = PRESETS[st.session_state.preset_idx]
    st.markdown(f"**{preset.descripcion}**")
    if preset.ayuda:
        st.caption("ℹ️ " + preset.ayuda)

    kdf = _get_kdf()
    t = kdf["t"].to_numpy()

    # --- Despacho por flujo de trabajo ---------------------------------
    wf = preset.workflow

    if wf == "fit":
        c1, c2 = st.columns(2)
        with c1:
            model_label = st.selectbox("Modelo", list(MODEL_LABELS.keys()))
        with c2:
            sig_label, y = _signal_picker(kdf, preset.signal or "y_m", "fit_sig")
        model = MODEL_LABELS[model_label]
        _run_generic_fit(t, y, model, sig_label)

    elif wf == "gravity_incline":
        variable = st.radio(
            "Variable a analizar",
            ["Posición x(t) — parábola (a = 2·a₂)",
             "Velocidad v(t) — recta (a = pendiente)"],
            help="Ambas rutas deben dar la misma aceleración; compararlas es una "
            "buena verificación.",
        )
        try:
            if variable.startswith("Posición"):
                x = kdf["x_m"].to_numpy()
                fit = fit_model("quadratic", t, x)
                a, sa = acceleration_from_quadratic(fit)
                st.metric(preset.param_label, _fmt(a, sa))
                _plot_fit(t, x, fit, "t (s)", "x (m)", "x(t) y parábola ajustada")
                _params_table(fit)
            else:
                sig_label, v = _signal_picker(kdf, "vx", "grav_sig")
                fit = fit_model("linear", t, v)
                a, sa = fit.params()["m"]
                st.metric(preset.param_label, _fmt(a, sa))
                _plot_fit(t, v, fit, "t (s)", sig_label, "v(t) y recta ajustada")
                _params_table(fit)
        except Exception as e:
            st.error(f"No se pudo ajustar: {e}")

    elif wf == "terminal_velocity":
        sig_label, v = _signal_picker(kdf, "speed", "term_sig")
        last_n = st.slider("Promediar últimos N puntos (meseta)",
                           2, max(2, len(kdf)), min(5, len(kdf)))
        vt, s = plateau_value(t, v, last_n=last_n)
        st.metric(preset.param_label, _fmt(vt, s))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=v, mode="lines+markers", name=sig_label))
        fig.add_hline(y=vt, line_dash="dash",
                      annotation_text=f"v_t ≈ {vt:.3g} m/s")
        fig.update_layout(title="Lectura de meseta (velocidad terminal)",
                          xaxis_title="t (s)", yaxis_title=sig_label, height=380,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    elif wf == "collision":
        axis_label = st.selectbox("Eje de movimiento", ["x (m)", "y (m)"])
        col = "x_m" if axis_label.startswith("x") else "y_m"
        x = kdf[col].to_numpy()
        t_split = st.slider("Instante del choque (s)",
                            float(t.min()), float(t.max()),
                            float((t.min() + t.max()) / 2), step=float(
                                (t.max() - t.min()) / max(len(t) - 1, 1)) or 0.001)
        try:
            fb, fa = piecewise_linear_fit(t, x, t_split)
            vi, svi = fb.params()["m"]
            vf, svf = fa.params()["m"]
            m1, m2 = st.columns(2)
            m1.metric("v_i (antes)", _fmt(vi, svi))
            m2.metric("v_f (después)", _fmt(vf, svf))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=x, mode="markers", name="datos"))
            tb = t[t <= t_split]
            ta = t[t > t_split]
            fig.add_trace(go.Scatter(x=tb, y=fb.predict(tb), mode="lines",
                                     name="antes"))
            fig.add_trace(go.Scatter(x=ta, y=fa.predict(ta), mode="lines",
                                     name="después"))
            fig.add_vline(x=t_split, line_dash="dot")
            fig.update_layout(title="Colisión: recta por tramos",
                              xaxis_title="t (s)", yaxis_title=axis_label,
                              height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(str(e))

    elif wf == "conical":
        x = kdf["x_m"].to_numpy()
        y = kdf["y_m"].to_numpy()
        xpx = kdf["x_px"].to_numpy()
        ypx = kdf["y_px"].to_numpy()
        frames = [int(f) for f in kdf["frame"].to_numpy()]
        if len(x) < 5:
            st.info("Marca al menos 5 puntos de la órbita para ajustar la elipse.")
            return
        try:
            # Geometría por ajuste de elipse GENERAL (admite rotación).
            ell = fit_ellipse(x, y)          # metros (reporte)
            ell_px = fit_ellipse(xpx, ypx)   # px (dibujo)
            # Período por ajuste sinusoidal (necesita el tiempo).
            _, _, geo = fit_conical_pendulum(t, x, y)

            c = float(np.sqrt(max(ell.a**2 - ell.b**2, 0.0)))
            ang = np.degrees(ell.theta) % 180
            if ang > 90:
                ang -= 180

            m1, m2, m3 = st.columns(3)
            m1.metric("Período T (s)", _fmt(geo.period, geo.period_err))
            m2.metric("Semieje mayor a (m)", f"{ell.a:.4g}")
            m3.metric("Semieje menor b (m)", f"{ell.b:.4g}")
            m4, m5, m6 = st.columns(3)
            m4.metric("Centro-foco c (m)", f"{c:.4g}")
            m5.metric("Excentricidad", f"{ell.eccentricity:.3g}")
            m6.metric("Inclinación elipse", f"{ang:.1f}°")
            st.caption(f"Centro (m): ({ell.cx:.3g}, {ell.cy:.3g})  ·  "
                       f"Foco 1: ({ell.foci[0][0]:.3g}, {ell.foci[0][1]:.3g})  ·  "
                       f"Foco 2: ({ell.foci[1][0]:.3g}, {ell.foci[1][1]:.3g})")

            # 1ª ley: ¿el equilibrio (origen) está en el centro o en un foco?
            D_centro = float(np.hypot(ell.cx, ell.cy))
            D_foco = min(float(np.hypot(*f)) for f in ell.foci)
            st.markdown(
                f"**1ª ley (Kepler):** equilibrio→**centro** = {D_centro*100:.2f} cm; "
                f"equilibrio→**foco más cercano** = {D_foco*100:.2f} cm. "
                "El punto de equilibrio coincide con aquel que sea ≈ 0."
            )

            cA, cB = st.columns(2)
            n_seg = cA.slider("Tramos de igual duración (áreas)",
                              2, min(8, max(2, len(kdf) - 1)), 4)
            ref_lbl = cB.radio("Medir áreas respecto de:",
                               ["Centro", "Foco 1", "Foco 2"], horizontal=True)
            ref_m = {"Centro": (ell.cx, ell.cy), "Foco 1": ell.foci[0],
                     "Foco 2": ell.foci[1]}[ref_lbl]
            ref_px = {"Centro": (ell_px.cx, ell_px.cy), "Foco 1": ell_px.foci[0],
                      "Foco 2": ell_px.foci[1]}[ref_lbl]

            # Overlay: elipse ajustada, ejes, focos y sectores desde la referencia.
            with st.spinner("Generando overlay de la órbita…"):
                ghost = reader.ghost_overlay(frames)
                overlay = draw_conical_overlay(ghost, xpx, ypx, ell_px, ref_px, n_seg)
            st.image(overlay, width=DISPLAY_WIDTH,
                     caption="Órbita y elipse ajustada (blanca) · eje mayor (rojo) · "
                     "eje menor (azul) · ✚ centro · ★ focos · ○ referencia de áreas")

            # Áreas barridas respecto del punto elegido (2ª ley).
            areas = swept_area_segments(x, y, ref_m, n_seg)
            bar = go.Figure(go.Bar(
                x=[f"tramo {i+1}" for i in range(len(areas))], y=areas,
                marker_color=["#ff5050", "#50b4ff", "#78ff78", "#ffdc50",
                              "#c878ff", "#ff963c", "#78dcdc", "#e678b4"]
                [:len(areas)]))
            cv = float(np.std(areas) / np.mean(areas) * 100) if np.mean(areas) else 0
            bar.update_layout(
                title=f"Áreas barridas desde el {ref_lbl.lower()} "
                      f"(variación {cv:.1f}%)",
                yaxis_title="área (m²)", height=320,
                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(bar, width="stretch")
            st.caption("2ª ley: las áreas son ≈ iguales **desde el punto correcto** "
                       "(el centro, para el péndulo). Compara cambiando la referencia. "
                       "Marca los puntos a intervalos de tiempo iguales.")
        except Exception as e:
            st.error(f"No se pudo ajustar la elipse: {e}")

    elif wf == "period_zero":
        sig_label, y = _signal_picker(kdf, preset.signal or "x_m", "pz_sig")
        T, n = period_by_zero_crossings(t, y)
        if np.isnan(T):
            st.warning("No se detectaron suficientes oscilaciones para un cálculo "
                       "confiable. Marca más ciclos o revisa la señal elegida.")
        else:
            st.metric(preset.param_label, f"{T:.4g} s")
            st.caption(f"Contados {n} ciclos completos por cruces ascendentes.")

        # Visualización del movimiento físico (overlay de la oscilación).
        with st.expander("👻 Ver movimiento (overlay de frames)", expanded=True):
            frames = [int(f) for f in kdf["frame"].to_numpy()]
            try:
                with st.spinner("Generando overlay…"):
                    ghost = reader.ghost_overlay(frames)
                st.image(ghost, width=DISPLAY_WIDTH,
                         caption="Oscilación del péndulo (multiexposición)")
            except Exception as e:
                st.caption(f"No se pudo generar el overlay: {e}")

        # Señal con la media y los cruces detectados marcados.
        ymean = float(np.mean(y))
        yc = y - ymean
        cross_t = []
        for i in range(len(yc) - 1):
            if yc[i] <= 0 < yc[i + 1]:
                frac = -yc[i] / (yc[i + 1] - yc[i])
                cross_t.append(t[i] + frac * (t[i + 1] - t[i]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", name=sig_label))
        fig.add_hline(y=ymean, line_dash="dot", annotation_text="media")
        for ct in cross_t:
            fig.add_vline(x=ct, line_dash="dot", line_color="green")
        fig.update_layout(title="Señal y cruces ascendentes (verde)",
                          xaxis_title="t (s)", yaxis_title=sig_label, height=380,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    elif wf == "linear_speed":
        axis_label = st.selectbox("Eje de movimiento", ["x (m)", "y (m)"])
        col = "x_m" if axis_label.startswith("x") else "y_m"
        x = kdf[col].to_numpy()
        unidad = st.radio("Unidad de rapidez", ["m/s", "mm/s"], horizontal=True)
        try:
            fit = fit_model("linear", t, x)
            v, sv = fit.params()["m"]
            factor = 1000.0 if unidad == "mm/s" else 1.0
            st.metric(f"{preset.param_label} ({unidad})",
                      _fmt(v * factor, sv * factor))
            _plot_fit(t, x, fit, "t (s)", axis_label, "Ajuste lineal x(t)")
            _params_table(fit)
        except Exception as e:
            st.error(f"No se pudo ajustar: {e}")


def _run_generic_fit(t, y, model: str, sig_label: str) -> None:
    """Ejecuta y muestra un ajuste genérico."""
    try:
        fit = fit_model(model, t, y)
        st.session_state.fit_result = fit
        _plot_fit(t, y, fit, "t (s)", sig_label, f"Ajuste: {model}")
        _params_table(fit)
        if model == "sine":
            f_v, f_s = fit.params()["f"]
            if f_v != 0:
                st.caption(f"Período T = {1/abs(f_v):.4g} s  ·  "
                           f"ω = {2*np.pi*abs(f_v):.4g} rad/s")
    except Exception as e:
        st.error(f"No se pudo ajustar: {e}")


# --------------------------------------------------------------------------
# Etapa 6 · Exportar
# --------------------------------------------------------------------------
def stage_export() -> None:
    st.subheader("Exportar resultados")

    calib = st.session_state.calibration
    if calib is None or _points_in_interval() < 2:
        st.info("Necesitas calibración (Etapa 2) y al menos 2 puntos dentro del "
                "intervalo (Etapa 3).")
        return

    kdf = _get_kdf()
    n = len(kdf)

    st.markdown("**1) Elige qué exportar**")
    inc_csv = st.checkbox("Datos completos (CSV: crudos + derivados)", value=True)

    st.caption("Figuras (PNG):")
    selected = {}
    cols = st.columns(2)
    for i, label in enumerate(export_mod.FIGURES.keys()):
        # Aceleración y error requieren ≥3 puntos para ser útiles.
        need3 = label in ("Aceleración vs tiempo", "Error de detección")
        disabled = need3 and n < 3
        selected[label] = cols[i % 2].checkbox(
            label, value=not disabled, disabled=disabled,
            help="Requiere al menos 3 puntos." if disabled else None,
        )

    st.markdown("**2) Genera y descarga**")
    if st.button("⚙️ Generar archivos", type="primary"):
        pngs = {}
        try:
            with st.spinner("Generando figuras…"):
                for label, chosen in selected.items():
                    if chosen:
                        fn, fname = export_mod.FIGURES[label]
                        pngs[fname] = fn(kdf)
            csv_bytes = export_mod.dataframe_to_csv_bytes(kdf) if inc_csv else None
            st.session_state["export_zip"] = export_mod.build_zip(pngs, csv_bytes)
            st.session_state["export_csv"] = csv_bytes
            st.session_state["export_ready"] = True
            st.success(f"Listo: {len(pngs)} figura(s)"
                       + (" + CSV" if inc_csv else "") + ".")
        except Exception as e:
            st.error(f"No se pudieron generar los archivos: {e}")
            st.session_state["export_ready"] = False

    if st.session_state.get("export_ready"):
        st.download_button(
            "⬇️ Descargar ZIP (figuras + CSV)",
            data=st.session_state["export_zip"],
            file_name="tracker_uai_export.zip",
            mime="application/zip",
            width="stretch",
        )
        if st.session_state.get("export_csv") is not None:
            st.download_button(
                "⬇️ Descargar solo el CSV",
                data=st.session_state["export_csv"],
                file_name="datos_tracker_uai.csv",
                mime="text/csv",
                width="stretch",
            )

    with st.expander("👁️ Previsualizar datos a exportar"):
        st.dataframe(kdf, width="stretch")


# --------------------------------------------------------------------------
# Router de etapas
# --------------------------------------------------------------------------
if etapa == "1 · Video":
    stage_video()
elif etapa == "2 · Calibración":
    stage_calibration()
elif etapa == "3 · Tracking":
    stage_tracking()
elif etapa == "4 · Cinemática":
    stage_kinematics()
elif etapa == "5 · Ajuste":
    stage_ajuste()
elif etapa == "6 · Exportar":
    stage_export()
