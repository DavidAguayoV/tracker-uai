"""Tests de la lógica de los Laboratorios 3 (sin vídeo): formato, colisiones,
órbita, tablas por vídeo y análisis de grupo."""

import numpy as np
import pandas as pd
import pytest

from core import formato as F
from core.colisiones import (
    Tramo,
    analizar_colision,
    bloque_de_ensayo,
    calculos_colision,
    detectar_saltos,
    ensayo_desde_nombre,
    resolucion_velocidad,
)
from core.grupo import ajuste_por_origen, ajustes_fis301, resumen_bloques_fis101
from core.lote import (
    COLUMNAS_FIS101,
    COLUMNAS_FIS301,
    avisos_radio_bloque,
    fila_fis301,
    nombre_fis301,
    reemplazar_fila,
)
from core.orbita import ajustar_circulo, analizar_orbita, detectar_arranque


# ==========================================================================
# Formato
# ==========================================================================
def test_coma_decimal_y_cifras():
    assert F.num(0.1234, 2) == "0,12"
    assert F.num(-1.5, 1) == "−1,5"
    assert F.cifras(0.0012345, 2) == "0,0012"
    assert F.num(float("nan")) == "—"


@pytest.mark.parametrize("valor,u,esperado", [
    (0.4523, 0.0123, "0,45 ± 0,01"),
    (0.4523, 0.0096, "0,45 ± 0,01"),     # 0,0096 → 0,01: sube una posición
    (5.037, 0.588, "5,0 ± 0,6"),
    (123.4, 31.0, "120 ± 30"),
    (-0.2667, 0.0225, "−0,27 ± 0,02"),
])
def test_valor_con_incertidumbre_a_una_cifra(valor, u, esperado):
    assert F.valor_inc(valor, u) == esperado


def test_desviacion_muestral_usa_n_menos_1():
    assert F.desviacion([1, 2, 3, 4]) == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert np.isnan(F.desviacion([2.0]))


def test_incertidumbre_es_la_mayor():
    assert F.incertidumbre(0.01, 0.02) == 0.02
    assert F.incertidumbre(0.03, 0.02) == 0.03
    assert F.incertidumbre(float("nan"), 0.02) == 0.02


def test_csv_para_excel_en_espanol_y_su_lectura():
    df = pd.DataFrame({"t (s)": [0.0, 0.0333], "v (m/s)": [0.25, -0.125]})
    datos = F.csv_es(df)
    texto = datos.decode("utf-8-sig")
    assert texto.splitlines()[0] == "t (s);v (m/s)"
    assert "0,0333;-0,125" in texto
    pd.testing.assert_frame_equal(F.leer_csv_es(datos), df)


# ==========================================================================
# Colisiones (con posiciones exactas, sin vídeo)
# ==========================================================================
def _ensayo(v1i, v2i, v1f, v2f, fps=30.0, tc=1.2, dur=2.8, tau=0.1,
            ruido=0.0, semilla=0, rebote_t=None):
    """Posiciones de dos carros con choque suave en tc (y rebote opcional)."""
    rng = np.random.default_rng(semilla)
    t = np.arange(int(dur * fps)) / fps
    g = np.linspace(0, t[-1], 20000)

    def s(z):
        z = np.clip(z, 0, 1)
        return z * z * (3 - 2 * z)

    def x(vi, vf, rebote=False):
        v = vi + (vf - vi) * s((g - tc) / tau + 0.5)
        if rebote and rebote_t is not None:
            v = v - 2 * vf * s((g - rebote_t) / 0.05 + 0.5)
        xg = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(g))])
        return np.interp(t, g, xg) + rng.normal(0, ruido, t.shape)

    return t, x(v1i, v1f) + 0.2, t, x(v2i, v2f, rebote=True) + 0.4


def test_caso_A_elastico_recupera_las_cuatro_velocidades():
    t1, x1, t2, x2 = _ensayo(0.30, 0.0, 0.0, 0.30)
    r = analizar_colision(t1, x1, t2, x2)
    v = r.velocidades
    assert v["v1i"] == pytest.approx(0.30, abs=1e-3)
    assert v["v2i"] == pytest.approx(0.0, abs=1e-3)
    assert v["v1f"] == pytest.approx(0.0, abs=1e-3)
    assert v["v2f"] == pytest.approx(0.30, abs=1e-3)
    assert r.t_choque == pytest.approx(1.2, abs=0.05)
    assert r.signo == 1
    # Ningún tramo toca la interacción magnética (1,15–1,25 s).
    assert r.antes.t_fin < 1.15 and r.despues.t_ini > 1.25


def test_carro_quieto_no_rompe_nada():
    """Bloques A y C: el carro 2 parte en reposo, sin ruido (desviación 0)."""
    t1, x1, t2, x2 = _ensayo(0.25, 0.0, 0.05, 0.30)
    r = analizar_colision(t1, x1, t2, x2)
    assert r.v2i.media == pytest.approx(0.0, abs=1e-6)
    assert r.v2i.desv == pytest.approx(0.0, abs=1e-6)
    c = calculos_colision(0.25, 0.25, **r.velocidades)
    assert np.isfinite(c.C_R)


def test_eje_invertido_se_corrige_al_sentido_del_carro_1():
    t1, x1, t2, x2 = _ensayo(0.30, -0.20, -0.2667, 0.0833)
    r = analizar_colision(t1, -x1, t2, -x2)
    assert r.signo == -1
    assert r.velocidades["v1i"] == pytest.approx(0.30, abs=1e-3)
    assert r.velocidades["v2i"] == pytest.approx(-0.20, abs=1e-3)


def test_segundo_choque_contra_un_tope_se_avisa_y_se_corta():
    t1, x1, t2, x2 = _ensayo(0.30, 0.0, 0.0, 0.30, rebote_t=2.2)
    r = analizar_colision(t1, x1, t2, x2)
    assert any(nombre == "carro 2" and abs(ts - 2.2) < 0.1
               for nombre, ts, _ in r.saltos_extra)
    # El tramo automático termina antes del rebote y v2f no se contamina.
    assert r.despues.t_fin < 2.2
    assert r.velocidades["v2f"] == pytest.approx(0.30, abs=2e-3)
    # Si el estudiante extiende el tramo a mano, se avisa (sin bloquear).
    r2 = analizar_colision(t1, x1, t2, x2, despues=Tramo(1.35, 2.75))
    assert any("segundo choque" in a for a in r2.avisos)


def test_tramo_corto_se_avisa():
    t1, x1, t2, x2 = _ensayo(0.30, 0.0, 0.0, 0.30)
    r = analizar_colision(t1, x1, t2, x2, antes=Tramo(0.8, 1.0))
    assert r.v1i.n < 8
    assert any("v1i" in a and "puntos" in a for a in r.avisos)


def test_tramos_manuales_se_respetan():
    t1, x1, t2, x2 = _ensayo(0.30, 0.0, 0.0, 0.30)
    r = analizar_colision(t1, x1, t2, x2, antes=Tramo(0.2, 0.9), despues=Tramo(1.5, 2.5))
    assert r.antes == Tramo(0.2, 0.9)
    assert r.v1i.n == len([t for t in t1 if 0.2 - 1e-9 <= t <= 0.9 + 1e-9])


def test_ruido_da_desviacion_y_la_incertidumbre_es_la_mayor():
    t1, x1, t2, x2 = _ensayo(0.30, 0.0, 0.0, 0.30, ruido=0.0005, semilla=4)
    res = resolucion_velocidad(0.0015, 1 / 30)
    r = analizar_colision(t1, x1, t2, x2, resolucion=res)
    assert r.v1i.desv > 0
    assert r.v1i.incertidumbre == pytest.approx(max(r.v1i.desv, res))
    assert r.velocidades["v1i"] == pytest.approx(0.30, rel=0.02)


def test_solo_carro_1():
    t1, x1, _, _ = _ensayo(0.30, 0.0, 0.0, 0.30)
    r = analizar_colision(t1, x1)
    assert r.v2i is None and r.v1f.media == pytest.approx(0.0, abs=1e-3)
    assert any("carro 2" in a for a in r.avisos)


def test_calculos_de_la_guia_ejemplo_del_ensayo_7():
    """Datos del ejemplo comentado de la guía (bloque C): p_i = 0,1130; C_R = 0,982."""
    m1, m2 = 0.2512, 0.5017
    v1i, v2i, v1f, v2f = 0.450, 0.0, -0.148, 0.2215
    c = calculos_colision(m1, m2, v1i, v2i, v1f, v2f)
    assert c.p_i == pytest.approx(m1 * v1i)
    assert c.dp == pytest.approx(c.p_f - c.p_i)
    assert c.C_R == pytest.approx(-(v2f - v1f) / (v2i - v1i))
    assert c.K_i == pytest.approx(0.5 * m1 * v1i**2)
    assert not c.p_i_pequeno


def test_p_i_casi_nulo_se_marca():
    """Bloques B y D: momenta opuestos casi se cancelan."""
    c = calculos_colision(0.25, 0.50, 0.30, -0.20, -0.2667, 0.0833)
    assert c.p_i == pytest.approx(-0.025)
    assert c.p_i_pequeno
    assert abs(c.dp_rel) > abs(c.dp / c.escala_p)


def test_detectar_saltos_ignora_la_friccion():
    t = np.arange(60) / 30
    v = 0.3 - 0.02 * t                       # fricción: cae poco a poco
    assert detectar_saltos(t, v, umbral=0.075) == []
    v2 = np.where(t < 1.0, 0.3, -0.3)
    assert len(detectar_saltos(t, v2, umbral=0.075)) == 1


@pytest.mark.parametrize("nombre,n,b", [
    ("ensayo07.mp4", 7, "C"), ("Ensayo_1.MOV", 1, "A"), ("ensayo12.mp4", 12, "D"),
    ("video.mp4", None, None),
])
def test_ensayo_y_bloque_desde_el_nombre(nombre, n, b):
    assert ensayo_desde_nombre(nombre) == n
    assert bloque_de_ensayo(n) == b


# ==========================================================================
# Órbita
# ==========================================================================
def _orbita(R=0.005, v=0.005, n=240, fps=30.0, ruido=0.0, horario=True, semilla=0):
    rng = np.random.default_rng(semilla)
    t = np.arange(n) / fps
    th = (-1 if horario else 1) * v / R * t
    x = 0.03 + R * np.cos(th) + rng.normal(0, ruido, n)
    y = 0.02 + R * np.sin(th) + rng.normal(0, ruido, n)
    return t, x, y


def test_circulo_exacto():
    t, x, y = _orbita()
    c = ajustar_circulo(x[:50], y[:50])     # arco parcial
    assert (c.cx, c.cy, c.r) == pytest.approx((0.03, 0.02, 0.005), abs=1e-9)


def test_orbita_rapidez_radio_mina_y_sentido():
    t, x, y = _orbita(ruido=2e-6, semilla=1)
    r = analizar_orbita(t, x, y, mina=(0.03, 0.02))
    assert r.rapidez_media == pytest.approx(0.005, rel=0.02)
    assert r.radio_medio == pytest.approx(0.005, rel=0.01)
    assert r.dist_mina_media == pytest.approx(0.005, rel=0.01)
    assert r.sentido == "horario"
    assert r.vueltas > 1
    assert r.rapidez_desv > 0
    assert r.avisos == []


def test_orbita_menos_de_una_vuelta_avisa():
    t, x, y = _orbita(n=100)
    r = analizar_orbita(t, x, y)
    assert r.vueltas < 1
    assert any("vuelta" in a for a in r.avisos)


def test_orbita_con_mina_equivocada_avisa():
    t, x, y = _orbita()
    r = analizar_orbita(t, x, y, mina=(0.05, 0.02))
    assert any("mina" in a for a in r.avisos)


def test_arranque_se_descarta():
    t = np.arange(300) / 30
    v = 0.004 * np.clip(t / 2.0, 0, 1)
    t0 = detectar_arranque(t, v)
    assert 1.6 < t0 < 2.1


# ==========================================================================
# Tablas por vídeo
# ==========================================================================
@pytest.mark.parametrize("nombre,b,V", [
    ("D0_10V.mp4", "D0", 10.0), ("d2_24v.MOV", "D2", 24.0), ("D1-18V.mp4", "D1", 18.0),
    ("grupo3_D1_12V_b.mp4", "D1", 12.0), ("video.mp4", None, None),
])
def test_bloque_y_voltaje_desde_el_nombre(nombre, b, V):
    assert nombre_fis301(nombre) == (b, V)


def test_fila_fis301_en_mm_y_reemplazo():
    t, x, y = _orbita()
    res = analizar_orbita(t, x, y)
    fila = fila_fis301("D0_20V.mp4", res, corriente_mA=42.0)
    assert fila["bloque"] == "D0" and fila["V (V)"] == 20.0
    assert fila["v (mm/s)"] == pytest.approx(5.0, rel=0.01)
    assert fila["r (mm)"] == pytest.approx(5.0, rel=0.01)
    df = pd.DataFrame(columns=COLUMNAS_FIS301)
    df = reemplazar_fila(df, fila)
    df = reemplazar_fila(df, dict(fila, **{"I (mA)": 43.0}))
    assert len(df) == 1 and df.loc[0, "I (mA)"] == 43.0
    assert list(df.columns) == COLUMNAS_FIS301


def test_radio_distinto_en_el_bloque_se_marca():
    df = pd.DataFrame({"bloque": ["D0"] * 4, "r (mm)": [5.0, 5.2, 4.9, 8.0]})
    av = avisos_radio_bloque(df)
    assert av.iloc[3] != "" and (av.iloc[:3] == "").all()


# ==========================================================================
# Análisis de grupo
# ==========================================================================
def test_ajuste_por_origen_como_excel():
    I = np.array([15.1, 28.6, 42.3, 55.0, 68.4])
    v = np.array([1.36, 2.55, 3.75, 4.87, 6.05])
    a = ajuste_por_origen(I, v)
    k = np.sum(I * v) / np.sum(I * I)
    assert a.k == pytest.approx(k)
    assert a.k == pytest.approx(0.0884, abs=5e-4)       # figura de la guía: 0,0883
    res = v - k * I
    assert a.k_err == pytest.approx(np.sqrt(np.sum(res**2) / 4 / np.sum(I * I)))
    assert a.r2 == pytest.approx(1 - np.sum(res**2) / np.sum(v * v))
    assert a.n == 5


def test_ajustes_por_bloque_no_mezclan_distancias():
    df = pd.DataFrame({
        "bloque": ["D0"] * 3 + ["D1"] * 3,
        "I (mA)": [10, 20, 30, 10, 20, 30],
        "v (mm/s)": [1, 2, 3, 0.5, 1.0, 1.5],
    })
    aj = ajustes_fis301(df)
    assert aj["D0"].k == pytest.approx(0.1)
    assert aj["D1"].k == pytest.approx(0.05)
    assert aj["D0"].r2 == pytest.approx(1.0)


def test_resumen_por_bloque_fis101():
    filas = []
    for i, (b, cr, rel) in enumerate([("A", 0.98, -1.0), ("A", 0.96, -2.0), ("A", 0.97, -1.5),
                                       ("B", 0.90, 10.0), ("B", 0.94, -20.0)]):
        filas.append({"ensayo": i + 1, "bloque": b, "C_R": cr, "Δp/p_i (%)": rel,
                      "Δp (kg·m/s)": 0.001 * rel})
    df = pd.DataFrame(filas)
    r = resumen_bloques_fis101(df)
    assert list(r["bloque"]) == ["A", "B"]
    a = r[r["bloque"] == "A"].iloc[0]
    assert a["C_R medio"] == pytest.approx(0.97)
    assert a["s C_R"] == pytest.approx(0.01)
    assert a["n"] == 3
    assert "Δp medio (kg·m/s)" in r.columns


def test_columnas_fis101_tienen_las_magnitudes_de_la_guia():
    for c in ["v1i (m/s)", "v2i (m/s)", "v1f (m/s)", "v2f (m/s)", "p_i (kg·m/s)",
              "p_f (kg·m/s)", "Δp (kg·m/s)", "Δp/p_i (%)", "K_i (J)", "K_f (J)", "C_R"]:
        assert c in COLUMNAS_FIS101
