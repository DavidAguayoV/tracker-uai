"""Prueba de integración del flujo guiado: la app se renderiza y se navega.

Usa ``streamlit.testing.v1.AppTest``, que ejecuta ``app.py`` headless. Cubre lo
que los tests de ``core/`` no pueden ver: que cada etapa se dibuje sin lanzar
excepciones y que los botones "Continuar" cambien de etapa de verdad.
"""

import pathlib

import pytest

from streamlit.testing.v1 import AppTest

from core.calibration import Calibration
from core.tracking import TrackTable
from core.video import VideoReader

RAIZ = pathlib.Path(__file__).resolve().parent.parent
VIDEO = RAIZ / "manual" / "ejemplos" / "carro.mp4"

pytestmark = pytest.mark.skipif(
    not VIDEO.exists(), reason="falta el video de ejemplo manual/ejemplos/carro.mp4"
)


def _app(etapa="1 · Video", con_calib=True, n_puntos=20):
    """App con un video, una calibración y una trayectoria sintética cargados."""
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120)
    at.session_state["video_path"] = str(VIDEO)
    at.session_state["reader"] = VideoReader(str(VIDEO))
    at.session_state["proc_start"] = 0
    at.session_state["proc_end"] = 40
    at.session_state["_last_interval"] = (0, 40)
    at.session_state["etapa"] = etapa

    tabla = TrackTable()
    for i in range(n_puntos):
        tabla.add_or_update(i, i / 30.0, 200 + 12 * i, 600 - 3 * i)
    at.session_state["track"] = tabla

    if con_calib:
        at.session_state["calibration"] = Calibration(
            m_per_px=0.002, origin_px=(100.0, 900.0), angle_rad=0.0
        )
        at.session_state["calib_p1"] = (100.0, 900.0)
        at.session_state["calib_p2"] = (600.0, 900.0)
        at.session_state["calib_origin"] = (100.0, 900.0)
    return at


def _sin_excepciones(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize(
    "etapa",
    ["1 · Video", "2 · Calibración", "3 · Tracking", "4 · Cinemática",
     "5 · Ajuste", "6 · Exportar"],
)
def test_cada_etapa_se_renderiza_sin_errores(etapa):
    """Ninguna de las seis etapas lanza excepción al dibujarse."""
    at = _app(etapa=etapa).run()
    _sin_excepciones(at)


def test_botones_continuar_avanzan_de_etapa():
    """El botón 'Continuar' del Paso 1 deja la app en el Paso 2.

    Es el punto delicado del flujo guiado: Streamlit prohíbe escribir en la
    clave de un widget ya instanciado, así que la navegación pasa por la clave
    intermedia ``_goto``.
    """
    at = _app(etapa="1 · Video").run()
    _sin_excepciones(at)
    boton = [b for b in at.button if b.key == "nav_video"]
    assert boton, "no se encontró el botón de avance del Paso 1"
    at = boton[0].click().run()
    _sin_excepciones(at)
    assert at.session_state["etapa"] == "2 · Calibración"
    assert "_goto" not in at.session_state, "el destino debe consumirse"


def test_cadena_completa_de_avance():
    """Se puede recorrer del Paso 1 al 6 solo con los botones de avance."""
    esperado = [
        ("nav_video", "2 · Calibración"),
        ("nav_calib", "3 · Tracking"),
        ("nav_track", "4 · Cinemática"),
        ("nav_kin", "5 · Ajuste"),
        ("nav_ajuste", "6 · Exportar"),
    ]
    at = _app(etapa="1 · Video").run()
    for key, destino in esperado:
        botones = [b for b in at.button if b.key == key]
        assert botones, f"falta el botón {key} en {at.session_state['etapa']}"
        at = botones[0].click().run()
        _sin_excepciones(at)
        assert at.session_state["etapa"] == destino


def test_calibracion_se_aplica_sola_al_completar_los_puntos():
    """Con los tres puntos marcados no hace falta pulsar ningún botón."""
    at = _app(etapa="2 · Calibración", con_calib=False)
    at.session_state["calib_p1"] = (100.0, 900.0)
    at.session_state["calib_p2"] = (600.0, 900.0)
    at.session_state["calib_origin"] = (100.0, 900.0)
    at.session_state["calib_real_dist"] = 0.5
    at = at.run()
    _sin_excepciones(at)
    calib = at.session_state["calibration"]
    assert calib is not None
    # 0,5 m repartidos en 500 px.
    assert calib.m_per_px == pytest.approx(0.001, rel=1e-6)


def test_tracking_arranca_en_modo_automatico():
    """El modo por defecto es el automático, no el marcado manual."""
    at = _app(etapa="3 · Tracking").run()
    _sin_excepciones(at)
    assert at.session_state["track_mode"] == "Automático"


def test_cinematica_muestra_el_resumen_del_movimiento():
    """La etapa de cinemática entrega métricas, no solo gráficos."""
    at = _app(etapa="4 · Cinemática").run()
    _sin_excepciones(at)
    etiquetas = [m.label for m in at.metric]
    for esperada in ["Puntos analizados", "Distancia recorrida",
                     "Rapidez media", "Rapidez máxima"]:
        assert esperada in etiquetas, f"falta la métrica {esperada}"


def test_exportar_ofrece_descargas_de_figuras_y_datos():
    """Se ven las figuras y cada una tiene su botón de descarga."""
    at = _app(etapa="6 · Exportar").run()
    _sin_excepciones(at)
    nombres = [d.label for d in at.download_button]
    assert any("posicion_t.png" in n for n in nombres), nombres
    assert any("CSV" in n for n in nombres), nombres
    assert any("ZIP" in n for n in nombres), nombres


def test_sin_video_la_app_no_revienta():
    """Sin video cargado la app muestra la bienvenida, no una excepción."""
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    _sin_excepciones(at)
    assert at.session_state["etapa"] == "1 · Video"


@pytest.mark.parametrize("etapa", ["📊 Gráficos (datos manuales)",
                                   "📐 Medición (foto)"])
def test_modulos_independientes_funcionan_sin_video(etapa):
    """Gráficos y Medición no requieren video ni rompen el bloque de progreso."""
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120)
    at.session_state["etapa"] = etapa
    at = at.run()
    _sin_excepciones(at)


def test_navegacion_no_usa_rerun_intermedio():
    """Un clic de avance deja la app en la etapa nueva en una sola pasada.

    Si la navegación volviera a usar ``st.rerun()`` dentro del clic, la etapa
    vieja se dibujaría primero y sus widgets quedarían huérfanos.
    """
    at = _app(etapa="3 · Tracking").run()
    _sin_excepciones(at)
    at = [b for b in at.button if b.key == "nav_track"][0].click().run()
    _sin_excepciones(at)
    assert at.session_state["etapa"] == "4 · Cinemática"
    etiquetas = [m.label for m in at.metric]
    assert "Puntos analizados" in etiquetas, "no se dibujó la etapa nueva"
