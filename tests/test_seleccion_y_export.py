"""Tests de la selección del objeto y de la coherencia pantalla ↔ exportación."""

import numpy as np
import pandas as pd
import pytest

import export as export_mod
from core.autotrack import box_from_click, default_box_size
from core.kinematics import kinematics_dataframe


# ==========================================================================
# Selección del objeto a seguir (un clic debe producir un recuadro visible)
# ==========================================================================
def test_box_from_click_queda_centrado_en_el_clic():
    """El recuadro se centra en el punto donde se hizo clic."""
    x, y, w, h = box_from_click(100, 80, 40, 640, 480)
    assert (x, y, w, h) == (80.0, 60.0, 40.0, 40.0)
    assert x + w / 2 == pytest.approx(100)
    assert y + h / 2 == pytest.approx(80)


def test_box_from_click_se_recorta_en_los_bordes():
    """Un clic en una esquina no genera un recuadro fuera del frame."""
    for cx, cy in [(0, 0), (639, 479), (5, 470)]:
        x, y, w, h = box_from_click(cx, cy, 60, 640, 480)
        assert x >= 0 and y >= 0
        assert x + w <= 640
        assert y + h <= 480


def test_box_from_click_respeta_tamano_minimo_y_maximo():
    """El lado nunca baja del mínimo ni excede el frame."""
    _, _, w, _ = box_from_click(50, 50, 2, 640, 480)
    assert w >= 8
    _, _, w2, h2 = box_from_click(50, 50, 5000, 640, 480)
    assert w2 <= 640 and h2 <= 480


def test_default_box_size_escala_con_la_resolucion():
    """Un video 4K sugiere un recuadro mayor que uno de baja resolución."""
    assert default_box_size(3840) > default_box_size(640)
    assert default_box_size(320) >= 24


# ==========================================================================
# Coherencia: la figura exportada usa la misma señal que la pantalla
# ==========================================================================
def _df_ruidoso():
    """Trayectoria con ruido, para que suavizar cambie de verdad los datos."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 41)
    x = 2.0 * t + rng.normal(0, 0.02, t.shape)
    y = 0.5 * t + rng.normal(0, 0.02, t.shape)
    return pd.DataFrame({"t": t, "x_m": x, "y_m": y})


def test_figura_de_posicion_cambia_al_suavizar():
    """El PNG exportado debe reflejar el suavizado, no quedarse en los datos crudos.

    Antes la pantalla graficaba x_s y el PNG seguía usando x_m: el estudiante
    pegaba en el informe una figura distinta de la que había visto.
    """
    df = _df_ruidoso()
    crudo = export_mod.fig_position(kinematics_dataframe(df, smooth=False))
    suave = export_mod.fig_position(
        kinematics_dataframe(df, smooth=True, window=7, polyorder=2)
    )
    assert crudo != suave


def test_figura_de_trayectoria_cambia_al_suavizar():
    df = _df_ruidoso()
    crudo = export_mod.fig_trajectory(kinematics_dataframe(df, smooth=False))
    suave = export_mod.fig_trajectory(
        kinematics_dataframe(df, smooth=True, window=7, polyorder=2)
    )
    assert crudo != suave


def test_todas_las_figuras_generan_png_valido():
    """Cada figura del registro produce bytes PNG no vacíos."""
    kdf = kinematics_dataframe(_df_ruidoso())
    for label, (fn, fname) in export_mod.FIGURES.items():
        data = fn(kdf)
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{label} no es PNG"
        assert len(data) > 1000
        assert fname.endswith(".png")


def test_zip_incluye_figuras_y_csv():
    """El ZIP de descarga trae el CSV y cada figura pedida."""
    import io
    import zipfile

    kdf = kinematics_dataframe(_df_ruidoso())
    pngs = {"posicion_t.png": export_mod.fig_position(kdf)}
    data = export_mod.build_zip(pngs, export_mod.dataframe_to_csv_bytes(kdf))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        nombres = set(zf.namelist())
    assert nombres == {"posicion_t.png", "datos_tracker_uai.csv"}
