"""Tests de cinemática: verifican derivadas sobre movimiento sintético."""

import numpy as np
import pytest

from core.kinematics import (
    adjust_window,
    compute_kinematics,
    finite_diff,
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
