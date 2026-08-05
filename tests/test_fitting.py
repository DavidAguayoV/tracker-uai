"""Tests de ajuste de modelos y utilidades de los presets."""

import numpy as np
import pytest

from core.fitting import (
    acceleration_from_quadratic,
    fit_conical_pendulum,
    fit_ellipse,
    fit_model,
    period_by_zero_crossings,
    piecewise_linear_fit,
    plateau_value,
    swept_area_segments,
)


def test_ajuste_lineal_pendiente_e_incertidumbre():
    """Ajuste lineal: recupera la pendiente con incertidumbre pequeña y R²≈1."""
    t = np.linspace(0, 2, 30)
    y = 0.75 * t - 0.2
    fit = fit_model("linear", t, y)
    m, sm = fit.params()["m"]
    assert m == pytest.approx(0.75, abs=1e-6)
    assert sm < 1e-6
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_parabola_recupera_aceleracion():
    """Parábola sintética con a=3: a = 2·a2 debe dar 3 con error esperado."""
    t = np.linspace(0, 1, 40)
    a_real = 3.0
    x = 0.5 * a_real * t**2 + 0.2 * t + 0.1
    fit = fit_model("quadratic", t, x)
    a, sa = acceleration_from_quadratic(fit)
    assert a == pytest.approx(a_real, abs=1e-3)
    assert fit.r_squared > 0.999


def test_parabola_con_ruido_dentro_de_incertidumbre():
    """Con ruido, el valor real cae dentro de ~3σ de la aceleración estimada."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 60)
    a_real = 9.8
    x = 0.5 * a_real * t**2 + rng.normal(0, 0.003, t.shape)
    fit = fit_model("quadratic", t, x)
    a, sa = acceleration_from_quadratic(fit)
    assert abs(a - a_real) < 3 * sa + 0.2


def test_ajuste_sinusoidal_periodo():
    """Sinusoidal: recupera la frecuencia (y por ende el período)."""
    t = np.linspace(0, 3, 300)
    y = 2.0 * np.sin(2 * np.pi * 1.5 * t + 0.5) + 0.3
    fit = fit_model("sine", t, y)
    f, sf = fit.params()["f"]
    assert abs(f) == pytest.approx(1.5, abs=1e-2)


def test_plateau_velocidad_terminal():
    """La meseta promedia el tramo final de una señal que satura."""
    t = np.linspace(0, 2, 100)
    v = 2.0 * np.tanh(5 * t)
    vt, s = plateau_value(t, v, last_n=10)
    assert vt == pytest.approx(2.0, abs=1e-3)
    assert s < 1e-3


def test_piecewise_colision():
    """Recta por tramos: recupera velocidades antes y después del choque."""
    t = np.linspace(0, 2, 41)
    x = np.where(t <= 1.0, 1.0 * t, 1.0 + 0.3 * (t - 1.0))
    fb, fa = piecewise_linear_fit(t, x, 1.0)
    assert fb.params()["m"][0] == pytest.approx(1.0, abs=1e-6)
    assert fa.params()["m"][0] == pytest.approx(0.3, abs=1e-6)


def test_piecewise_tramo_insuficiente_lanza_error():
    """Si un tramo tiene menos de 2 puntos, debe lanzar ValueError."""
    t = np.linspace(0, 1, 10)
    x = t.copy()
    with pytest.raises(ValueError):
        piecewise_linear_fit(t, x, t_split=t[-1] + 1.0)


def test_periodo_por_cruces():
    """Período por cruces por cero de una señal de 2 Hz -> T=0.5 s."""
    t = np.linspace(0, 3, 600)
    y = np.sin(2 * np.pi * 2.0 * t)
    T, n = period_by_zero_crossings(t, y)
    assert T == pytest.approx(0.5, abs=1e-3)
    assert n >= 5


def test_periodo_sin_oscilaciones_devuelve_nan():
    """Sin cruces suficientes, devuelve nan (no fuerza resultado)."""
    t = np.linspace(0, 1, 20)
    y = 2.0 + 0.0 * t
    T, n = period_by_zero_crossings(t, y)
    assert np.isnan(T)


def test_pendulo_conico_geometria():
    """Péndulo cónico: recupera T, semiejes y distancia focal de la elipse."""
    t = np.linspace(0, 2, 200)
    a, b, f = 1.0, 0.5, 1.0
    x = a * np.sin(2 * np.pi * f * t)
    y = b * np.sin(2 * np.pi * f * t + np.pi / 2)
    fx, fy, geo = fit_conical_pendulum(t, x, y)
    assert geo.period == pytest.approx(1.0, abs=1e-2)
    assert geo.semi_major == pytest.approx(1.0, abs=1e-2)
    assert geo.semi_minor == pytest.approx(0.5, abs=1e-2)
    # c = sqrt(a^2 - b^2) = sqrt(0.75) ≈ 0.866
    assert geo.focal_distance == pytest.approx(np.sqrt(0.75), abs=1e-2)


def test_fit_ellipse_elipse_inclinada():
    """Ajuste de elipse general: recupera semiejes, rotación y excentricidad."""
    a, b, th, cx, cy = 2.0, 1.0, np.radians(30), 3.0, -1.0
    t = np.linspace(0, 2 * np.pi, 60)
    xr, yr = a * np.cos(t), b * np.sin(t)
    x = cx + xr * np.cos(th) - yr * np.sin(th)
    y = cy + xr * np.sin(th) + yr * np.cos(th)
    e = fit_ellipse(x, y)
    assert e.cx == pytest.approx(3.0, abs=1e-3)
    assert e.cy == pytest.approx(-1.0, abs=1e-3)
    assert e.a == pytest.approx(2.0, abs=1e-3)
    assert e.b == pytest.approx(1.0, abs=1e-3)
    assert e.eccentricity == pytest.approx(np.sqrt(1 - 0.25), abs=1e-3)
    # la inclinación es una recta (mód. 180°): 30° o 210°≡30°
    assert (np.degrees(e.theta) % 180) == pytest.approx(30.0, abs=1.0)


def test_fit_ellipse_pocos_puntos_error():
    """Con menos de 5 puntos no se puede ajustar una elipse."""
    with pytest.raises(ValueError):
        fit_ellipse(np.array([0, 1, 2.0]), np.array([0, 1, 0.0]))


def test_areas_barridas_kepler_aproximadamente_iguales():
    """Órbita a velocidad angular constante: áreas barridas ≈ iguales."""
    t = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(t)
    y = np.sin(t)
    areas = swept_area_segments(x, y, center=(0.0, 0.0), n_segments=4)
    assert np.std(areas) / np.mean(areas) < 0.02  # variación < 2 %
