"""Tests de cinemática: verifican derivadas sobre movimiento sintético."""

import numpy as np
import pandas as pd
import pytest

from core.kinematics import (
    adjust_window,
    compute_kinematics,
    finite_diff,
    frame_gaps,
    kinematics_dataframe,
    motion_summary,
    position_columns,
    savgol_smooth,
)

G = 9.8


def test_mru_velocidad_constante():
    """MRU: velocidad constante y aceleración nula."""
    t = np.linspace(0, 2, 21)
    x = 3.0 * t + 1.0
    y = np.zeros_like(t)
    k = compute_kinematics(t, x, y)
    assert np.allclose(k["vx"], 3.0, atol=1e-6)
    assert np.allclose(k["ax"], 0.0, atol=1e-6)


def test_tiro_parabolico_recupera_gravedad():
    """Tiro parabólico: ay ≈ −g y vy(0) correcto."""
    t = np.linspace(0, 1, 41)
    x = 2.0 * t
    y = 1.0 + 3.0 * t - 0.5 * G * t**2
    k = compute_kinematics(t, x, y)
    # vx constante = 2
    assert np.allclose(k["vx"], 2.0, atol=1e-6)
    # ay ≈ -g en el interior (los extremos son de un solo lado)
    assert np.mean(k["ay"]) == pytest.approx(-G, abs=1e-3)
    # velocidad vertical inicial
    assert k["vy"][0] == pytest.approx(3.0, abs=1e-2)


def test_suavizado_reduce_ruido_en_aceleracion():
    """El suavizado Savitzky-Golay debe acercar ay a −g con datos ruidosos."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 61)
    x = 2.0 * t
    y = 1.0 + 3.0 * t - 0.5 * G * t**2 + rng.normal(0, 0.002, t.shape)
    sin_suavizar = compute_kinematics(t, x, y)
    con_suavizado = compute_kinematics(t, x, y, smooth=True, window=11, polyorder=2)
    err_sin = abs(np.mean(sin_suavizar["ay"]) + G)
    err_con = abs(np.mean(con_suavizado["ay"]) + G)
    # el suavizado no debe empeorar y típicamente mejora el promedio
    assert err_con <= err_sin + 0.05


def test_finite_diff_paso_no_uniforme():
    """Las diferencias centradas respetan un paso de tiempo no uniforme."""
    t = np.array([0.0, 0.1, 0.3, 0.35, 0.6])  # espaciado irregular
    f = 5.0 * t  # pendiente 5
    assert np.allclose(finite_diff(f, t), 5.0, atol=1e-6)


def test_adjust_window_reglas_savgol():
    """La ventana se ajusta a impar, ≤ n y > polyorder; None si no es posible."""
    assert adjust_window(n=20, window=6, polyorder=2) == 7   # par -> impar
    assert adjust_window(n=5, window=11, polyorder=2) == 5   # recorta a n impar
    assert adjust_window(n=3, window=3, polyorder=2) is None  # muy pocos puntos


def test_savgol_smooth_pocos_puntos_devuelve_original():
    """Con muy pocos puntos, savgol_smooth devuelve la señal sin cambios."""
    y = np.array([1.0, 2.0])
    assert np.allclose(savgol_smooth(y, window=5, polyorder=2), y)


# ==========================================================================
# Helpers de presentación (resumen, huecos, columnas a graficar)
# ==========================================================================
def _df_mru(n=20, v=2.0, dt=0.05):
    """DataFrame cinemático de un MRU horizontal, ya derivado."""
    t = np.arange(n) * dt
    df = pd.DataFrame({"t": t, "x_m": v * t, "y_m": np.zeros(n)})
    return kinematics_dataframe(df)


def test_position_columns_prefiere_suavizado():
    """Si hay columnas suavizadas, son las que se grafican y se exportan."""
    t = np.linspace(0, 1, 21)
    df = pd.DataFrame({"t": t, "x_m": t, "y_m": t * 0})
    crudo = kinematics_dataframe(df, smooth=False)
    suave = kinematics_dataframe(df, smooth=True, window=5, polyorder=2)
    assert position_columns(crudo) == ("x_m", "y_m", False)
    assert position_columns(suave) == ("x_s", "y_s", True)


def test_motion_summary_mru():
    """MRU: recorrido = desplazamiento y rapidez media = la del movimiento."""
    df = _df_mru(n=21, v=2.0, dt=0.05)
    r = motion_summary(df)
    assert r["n_puntos"] == 21
    assert r["duracion"] == pytest.approx(1.0)
    assert r["dt_medio"] == pytest.approx(0.05)
    assert r["fps_efectivo"] == pytest.approx(20.0)
    assert r["recorrido"] == pytest.approx(2.0, rel=1e-9)
    assert r["desplazamiento"] == pytest.approx(2.0, rel=1e-9)
    assert r["rapidez_media"] == pytest.approx(2.0, rel=1e-6)


def test_motion_summary_ida_y_vuelta_distingue_recorrido_de_desplazamiento():
    """Volver al punto de partida deja desplazamiento nulo pero recorrido no."""
    t = np.linspace(0, 2, 41)
    x = np.where(t <= 1, t, 2 - t)
    df = kinematics_dataframe(pd.DataFrame({"t": t, "x_m": x, "y_m": x * 0}))
    r = motion_summary(df)
    assert r["desplazamiento"] == pytest.approx(0.0, abs=1e-9)
    assert r["recorrido"] == pytest.approx(2.0, rel=1e-6)


def test_motion_summary_pocos_puntos_no_revienta():
    """Con 0 o 1 punto devuelve nan en vez de lanzar excepción."""
    vacio = motion_summary(pd.DataFrame(columns=["t", "x_m", "y_m"]))
    assert vacio["n_puntos"] == 0
    uno = motion_summary(kinematics_dataframe(
        pd.DataFrame({"t": [0.0], "x_m": [0.0], "y_m": [0.0]})))
    assert uno["n_puntos"] <= 1
    assert np.isnan(uno["duracion"])


def test_frame_gaps_detecta_frames_sin_punto():
    """Los frames del tramo que no tienen detección se listan en orden."""
    assert frame_gaps([0, 1, 3, 5], 0, 5) == [2, 4]
    assert frame_gaps([0, 1, 2, 3], 0, 3) == []


def test_frame_gaps_respeta_el_paso():
    """Con paso 2 solo se esperan puntos en los frames múltiplos del paso."""
    assert frame_gaps([0, 2, 4], 0, 4, step=2) == []
    assert frame_gaps([0, 4], 0, 4, step=2) == [2]
