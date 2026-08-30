"""Tests de ajuste de modelos y utilidades de los presets."""

import numpy as np
import pytest

from core.fitting import (
    CD_RANGO_ACEPTABLE,
    G_SANTIAGO,
    RHO_AIRE_SANTIAGO,
    acceleration_from_quadratic,
    bar_gyration_radius,
    bifilar_inertia,
    drag_from_terminal_velocity,
    fit_conical_pendulum,
    fit_ellipse,
    fit_model,
    frontal_area,
    period_by_zero_crossings,
    period_timing_uncertainty,
    piecewise_linear_fit,
    plateau_value,
    read_plateau,
    reynolds_number,
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


# ==========================================================================
# FIS101 Lab 2 · arrastre aerodinámico y velocidad terminal
# ==========================================================================
def test_read_plateau_por_ultimos_n_detecta_meseta():
    """En la meseta de una tanh, el tramo final es plano y da v_t."""
    t = np.linspace(0, 2, 100)
    v = 2.13 * np.tanh(G_SANTIAGO * t / 2.13)
    pl = read_plateau(t, v, last_n=10)
    assert pl.value == pytest.approx(2.13, abs=1e-3)
    assert pl.n == 10
    assert pl.t_end == pytest.approx(t[-1])
    assert pl.is_flat


def test_read_plateau_tramo_inicial_no_es_meseta():
    """El tramo inicial de la caída tiene pendiente ≈ g: no es meseta."""
    t = np.linspace(0, 2, 100)
    v = 2.13 * np.tanh(G_SANTIAGO * t / 2.13)
    pl = read_plateau(t, v, t_min=0.0, t_max=0.15)
    assert not pl.is_flat
    assert pl.slope > 5.0          # se parece a la caída libre


def test_read_plateau_por_intervalo_de_tiempo():
    """El modo intervalo promedia solo los puntos dentro de [t_min, t_max]."""
    t = np.linspace(0, 2, 101)
    v = np.where(t < 1.0, 1.0, 3.0)
    pl = read_plateau(t, v, t_min=1.2, t_max=2.0)
    assert pl.value == pytest.approx(3.0)
    assert pl.t_start >= 1.2


def test_read_plateau_ignora_valores_no_finitos():
    """Los NaN de la señal no contaminan el promedio de la meseta."""
    t = np.linspace(0, 1, 20)
    v = np.full(20, 2.0)
    v[5] = np.nan
    pl = read_plateau(t, v, last_n=10)
    assert pl.value == pytest.approx(2.0)
    assert np.isfinite(pl.sigma)


def test_plateau_value_sigue_siendo_compatible():
    """El envoltorio antiguo devuelve (media, sigma) como antes."""
    t = np.linspace(0, 2, 100)
    v = 2.0 * np.tanh(5 * t)
    assert plateau_value(t, v, last_n=10) == pytest.approx(
        (read_plateau(t, v, last_n=10).value, read_plateau(t, v, last_n=10).sigma)
    )


def test_area_frontal_del_filtro():
    """A = π D²/4 para el filtro de 11 cm de la guía."""
    assert frontal_area(0.110) == pytest.approx(9.503e-3, rel=1e-3)


def test_area_frontal_diametro_invalido():
    with pytest.raises(ValueError):
        frontal_area(0.0)


def test_reynolds_del_filtro_es_regimen_cuadratico():
    """El ejemplo de la guía: filtro de 11 cm a 1,2 m/s da Re ~ 8000."""
    re = reynolds_number(1.2, 0.110)
    assert re == pytest.approx(8000, rel=0.05)
    assert re > 1e3                # muy lejos del régimen viscoso


def test_drag_recupera_cd_de_una_caida_sintetica():
    """Con v_t generada desde C_d = 1,3, el cálculo inverso lo recupera."""
    c_d = 1.3
    masa, diametro = 3.04e-3, 0.110
    area = frontal_area(diametro)
    v_t = np.sqrt(2 * masa * G_SANTIAGO / (RHO_AIRE_SANTIAGO * c_d * area))
    dr = drag_from_terminal_velocity(v_t, masa, diametro)
    assert dr.c_d == pytest.approx(c_d, rel=1e-9)
    assert dr.in_reference_range
    assert dr.percent_difference == pytest.approx(0.0, abs=1e-6)


def test_drag_duplica_la_incertidumbre_relativa():
    """C_d ∝ v_t⁻²: la incertidumbre relativa se duplica."""
    dr = drag_from_terminal_velocity(2.13, 3.04e-3, 0.110, v_t_err=0.05)
    assert dr.c_d_err / dr.c_d == pytest.approx(2 * 0.05 / 2.13, rel=1e-9)


def test_drag_detecta_masa_en_gramos_como_fuera_de_rango():
    """Pasar la masa en gramos infla C_d por 1000: debe salir del rango."""
    lo, hi = CD_RANGO_ACEPTABLE
    dr = drag_from_terminal_velocity(2.13, 3.04, 0.110)   # 3,04 "kg"
    assert not dr.in_reference_range
    assert dr.c_d > hi


def test_drag_rechaza_entradas_no_fisicas():
    with pytest.raises(ValueError):
        drag_from_terminal_velocity(0.0, 3.04e-3, 0.110)
    with pytest.raises(ValueError):
        drag_from_terminal_velocity(2.13, -1.0, 0.110)


# ==========================================================================
# FIS201 Lab 2 · péndulo bifilar y momento de inercia
# ==========================================================================
def test_bifilar_recupera_inercia_de_una_barra_uniforme():
    """Genera T desde I = ML²/12 y comprueba que el cálculo inverso lo devuelve."""
    M, L, d, H = 0.050, 0.150, 0.025, 0.550
    inercia = M * L**2 / 12.0
    T = 2 * np.pi * np.sqrt(inercia * H / (M * G_SANTIAGO * d**2))
    bf = bifilar_inertia(T, M, d, H)
    assert bf.inertia == pytest.approx(inercia, rel=1e-9)
    assert bf.gyration_radius == pytest.approx(bar_gyration_radius(L), rel=1e-9)


def test_bifilar_periodo_crece_como_raiz_de_H():
    """A I fijo, duplicar H multiplica el período por √2 (no por 2)."""
    M, d, inercia = 0.050, 0.025, 9.375e-5

    def periodo(H):
        return 2 * np.pi * np.sqrt(inercia * H / (M * G_SANTIAGO * d**2))

    assert periodo(0.80) / periodo(0.40) == pytest.approx(np.sqrt(2), rel=1e-9)
    # Y el cálculo inverso devuelve la misma inercia para ambos largos.
    assert bifilar_inertia(periodo(0.40), M, d, 0.40).inertia == pytest.approx(
        bifilar_inertia(periodo(0.80), M, d, 0.80).inertia, rel=1e-9
    )


def test_bifilar_confundir_d_con_el_largo_da_factor_nueve():
    """Usar L/2 en vez de d (2d=5 cm, L=15 cm) infla I_zz nueve veces."""
    M, H, T = 0.050, 0.550, 2.58
    correcto = bifilar_inertia(T, M, 0.025, H).inertia
    equivocado = bifilar_inertia(T, M, 0.075, H).inertia
    assert equivocado / correcto == pytest.approx(9.0, rel=1e-9)


def test_bifilar_duplica_la_incertidumbre_relativa():
    """I ∝ T²: la incertidumbre relativa se duplica."""
    bf = bifilar_inertia(2.58, 0.050, 0.025, 0.550, period_err=0.04)
    assert bf.inertia_err / bf.inertia == pytest.approx(2 * 0.04 / 2.58, rel=1e-9)


def test_bifilar_rechaza_entradas_no_fisicas():
    with pytest.raises(ValueError):
        bifilar_inertia(0.0, 0.050, 0.025, 0.550)
    with pytest.raises(ValueError):
        bifilar_inertia(2.58, 0.050, 0.0, 0.550)
    with pytest.raises(ValueError):
        bifilar_inertia(2.58, 0.050, 0.025, 0.0)


def test_radio_de_giro_de_barra_uniforme():
    assert bar_gyration_radius(0.150) == pytest.approx(0.150 / np.sqrt(12))


def test_incertidumbre_de_cronometraje_baja_con_los_ciclos():
    """Medir 10 ciclos en vez de 1 divide el error de cronometraje por 10."""
    uno = period_timing_uncertainty(60.0, 1)
    diez = period_timing_uncertainty(60.0, 10)
    assert uno == pytest.approx(2 / 60.0)
    assert diez == pytest.approx(uno / 10)


def test_incertidumbre_de_cronometraje_sin_fps_es_nan():
    assert np.isnan(period_timing_uncertainty(0.0, 10))
    assert np.isnan(period_timing_uncertainty(60.0, 0))
