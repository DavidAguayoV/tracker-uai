"""Integración (AppTest) de los flujos de los Laboratorios 3.

Se usan los vídeos sintéticos de ``tests/sinteticos.py``: la app corre
headless con el vídeo cargado, la calibración aplicada y los objetos ya
seguidos (o se ejecuta el seguimiento desde la propia interfaz).
"""

import pathlib

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.autotrack import box_from_click
from core.calibration import Calibration
from core.lote import COLUMNAS_FIS301, tabla_fis301_vacia
from core.seguimiento import muestra_color, seguir_color, seguir_plantilla
from core.tracking import TrackTable
from core.video import VideoReader
from models.experiment_presets import PRESETS
from tests.sinteticos import CASO_A, Orbita, video_colision, video_orbita

RAIZ = pathlib.Path(__file__).resolve().parent.parent
IDX = {p.key: i for i, p in enumerate(PRESETS)}
ETAPA_LOTE = "🌀 Lote FIS301 (24 videos)"
ETAPA_GRUPO = "📈 Análisis de grupo (Lab 3)"


def _sin_excepciones(at):
    assert not at.exception, [e.value for e in at.exception]


def _tabla(pts):
    t = TrackTable()
    for fi, tt, x, y in pts:
        t.add_or_update(fi, tt, x, y)
    return t


@pytest.fixture(scope="module")
def colision(tmp_path_factory):
    d = tmp_path_factory.mktemp("col")
    ruta, datos = video_colision(CASO_A, d / "ensayo01.mp4")
    r = VideoReader(ruta)
    r.preload_range(0, r.n_frames - 1)
    f0 = r.get_frame(0)
    pts1, _ = seguir_color(r, muestra_color(f0, datos["px1"][0], datos["y"]),
                           0, r.n_frames - 1, semilla=(datos["px1"][0], datos["y"]))
    exc = {p[0]: (p[2], p[3]) for p in pts1}
    pts2, _ = seguir_color(r, muestra_color(f0, datos["px2"][0], datos["y"]),
                           0, r.n_frames - 1, semilla=(datos["px2"][0], datos["y"]),
                           excluir=exc)
    r.release()
    return ruta, datos, pts1, pts2


@pytest.fixture(scope="module")
def orbita(tmp_path_factory):
    d = tmp_path_factory.mktemp("orb")
    o = Orbita()
    ruta, datos = video_orbita(o, d / "D0_20V.mp4")
    r = VideoReader(ruta)
    r.preload_range(0, r.n_frames - 1)
    m = muestra_color(r.get_frame(0), datos["cx"][0], datos["cy"][0])
    pts, _ = seguir_color(r, m, 0, r.n_frames - 1, semilla=(datos["cx"][0], datos["cy"][0]))
    r.release()
    return ruta, datos, pts, m, o


def _app_colision(colision, etapa="5 · Ajuste", con_puntos=True):
    ruta, datos, pts1, pts2 = colision
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180)
    reader = VideoReader(ruta)
    ss = at.session_state
    ss["video_path"] = str(ruta)
    ss["reader"] = reader
    ss["_uploaded_name"] = "ensayo01.mp4"
    ss["proc_start"] = 0
    ss["proc_end"] = reader.n_frames - 1
    ss["_last_interval"] = (0, reader.n_frames - 1)
    ss["preset_idx"] = IDX["fis101_lab3_colisiones"]
    ss["calibration"] = Calibration(m_per_px=CASO_A.escala, origin_px=(0.0, 300.0))
    t1 = _tabla(pts1) if con_puntos else TrackTable()
    t2 = _tabla(pts2) if con_puntos else TrackTable()
    ss["objetos"] = {"Carro 1": t1, "Carro 2": t2}
    ss["objeto_activo"] = "Carro 1"
    ss["track"] = t1
    ss["m1_g"] = 250.0
    ss["m2_g"] = 250.0
    ss["etapa"] = etapa
    return at


def _app_orbita(orbita, etapa="5 · Ajuste"):
    ruta, datos, pts, m, o = orbita
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180)
    reader = VideoReader(ruta)
    ss = at.session_state
    ss["video_path"] = str(ruta)
    ss["reader"] = reader
    ss["_uploaded_name"] = "D0_20V.mp4"
    ss["proc_start"] = 0
    ss["proc_end"] = reader.n_frames - 1
    ss["_last_interval"] = (0, reader.n_frames - 1)
    ss["preset_idx"] = IDX["fis301_lab3_lorentz"]
    ss["calibration"] = Calibration(m_per_px=o.escala, origin_px=(0.0, 480.0))
    ss["calib_size"] = (reader.width, reader.height)
    t = _tabla(pts)
    ss["objetos"] = {"Objeto 1": t}
    ss["objeto_activo"] = "Objeto 1"
    ss["track"] = t
    ss["mina_px"] = datos["mina"]
    ss["muestra_trazador"] = m
    ss["etapa"] = etapa
    return at


def _metricas(at):
    return {m.label: m.value for m in at.metric}


# ==========================================================================
# FIS101 · Lab 3
# ==========================================================================
def test_fis101_paso5_entrega_las_cuatro_velocidades_y_la_guia(colision):
    at = _app_colision(colision).run()
    _sin_excepciones(at)
    m = _metricas(at)
    assert m["v1i (m/s)"].startswith("0,30 ±"), m
    assert m["v2i (m/s)"].startswith("0,00 ±"), m
    assert m["v1f (m/s)"].startswith("0,00 ±"), m
    assert m["v2f (m/s)"].startswith("0,30 ±"), m
    for etiqueta in ["p_i (kg·m/s)", "p_f (kg·m/s)", "Δp = p_f − p_i (kg·m/s)",
                     "Δp/p_i (%)", "K_i (J)", "K_f (J)", "C_R"]:
        assert etiqueta in m, etiqueta
    assert float(m["C_R"].replace(",", ".")) == pytest.approx(1.0, abs=0.02)


def test_fis101_guardar_ensayo_y_analisis_de_grupo(colision):
    at = _app_colision(colision).run()
    _sin_excepciones(at)
    at = at.button(key="ens_guardar").click().run()
    _sin_excepciones(at)
    tabla = at.session_state["tabla_fis101"]
    assert len(tabla) == 1
    fila = tabla.iloc[0]
    assert fila["ensayo"] == 1 and fila["bloque"] == "A"
    assert fila["C_R"] == pytest.approx(1.0, abs=0.02)
    assert fila["Δp (kg·m/s)"] == pytest.approx(0.0, abs=0.002)
    # Pestaña de grupo: muestra la tabla por bloque sin redactar conclusiones.
    at.session_state["etapa"] = ETAPA_GRUPO
    at = at.run()
    _sin_excepciones(at)
    assert len(at.dataframe) >= 2
    nombres = [d.label for d in at.download_button]
    assert any("Resumen por bloque" in n for n in nombres), nombres


def test_fis101_sin_carro_2_no_revienta(colision):
    at = _app_colision(colision)
    at.session_state["objetos"]["Carro 2"] = TrackTable()
    at = at.run()
    _sin_excepciones(at)
    assert any("Carro 2" in w.value for w in at.warning)


def test_fis101_tramos_a_mano(colision):
    at = _app_colision(colision).run()
    at = at.checkbox(key="col_manual").check().run()
    _sin_excepciones(at)
    sliders = [s for s in at.slider if s.key and s.key.startswith("col_antes")]
    assert sliders
    at = sliders[0].set_value((0.2, 0.8)).run()
    _sin_excepciones(at)
    assert any("0,200–0,800" in c.value for c in at.caption), [c.value for c in at.caption][:5]


def test_fis101_seguimiento_de_dos_carros_desde_la_interfaz(colision):
    """Paso 3: se sigue el carro 1, se cambia al carro 2 y se sigue (misma marca)."""
    ruta, datos, _, _ = colision
    at = _app_colision(colision, etapa="3 · Tracking", con_puntos=False)
    at.session_state["auto_roi"] = box_from_click(datos["px1"][0], datos["y"], 30, 1280, 480)
    at = at.run()
    _sin_excepciones(at)
    boton = [b for b in at.button if b.label.startswith("▶️ Ejecutar")]
    at = boton[0].click().run()
    _sin_excepciones(at)
    n = len(datos["t"])
    assert len(at.session_state["objetos"]["Carro 1"].points) == n
    # Cambia al carro 2 con el selector del Paso 3.
    at = at.radio(key="obj_sel_p3").set_value("Carro 2").run()
    _sin_excepciones(at)
    assert at.session_state["objeto_activo"] == "Carro 2"
    assert len(at.session_state["track"].points) == 0
    at.session_state["auto_roi"] = box_from_click(datos["px2"][0], datos["y"], 30, 1280, 480)
    at = at.run()
    boton = [b for b in at.button if b.label.startswith("▶️ Ejecutar")]
    at = boton[0].click().run()
    _sin_excepciones(at)
    c1 = at.session_state["objetos"]["Carro 1"].points
    c2 = at.session_state["objetos"]["Carro 2"].points
    assert len(c2) == n
    # El carro 2 no se confundió con el carro 1 (marcas del mismo color).
    for p in c2:
        assert abs(p.x_px - datos["px2"][p.frame_idx]) < 2.0


def test_fis101_exportar_incluye_ambos_carros(colision):
    at = _app_colision(colision, etapa="6 · Exportar").run()
    _sin_excepciones(at)
    nombres = [d.label for d in at.download_button]
    assert any("velocidad_objetos.png" in n for n in nombres), nombres


def test_elegir_el_lab_en_la_barra_lateral_nombra_los_carros():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    _sin_excepciones(at)
    at = at.selectbox(key="lab_sel").set_value(IDX["fis101_lab3_colisiones"]).run()
    _sin_excepciones(at)
    assert list(at.session_state["objetos"]) == ["Carro 1", "Carro 2"]
    assert at.session_state["mantener_calib"] is True


# ==========================================================================
# FIS301 · Lab 3
# ==========================================================================
def test_fis301_paso5_rapidez_radio_y_mina(orbita):
    at = _app_orbita(orbita).run()
    _sin_excepciones(at)
    m = _metricas(at)
    assert "±" in m["v (mm/s)"], m
    v = float(m["v (mm/s)"].split("±")[0].replace(",", "."))
    assert v == pytest.approx(5.0, rel=0.1)       # redondeado a la cifra de u
    assert m["Radio de la órbita r (mm)"].startswith("5,0")
    assert m["Distancia a la mina (mm)"].startswith("5,0")


def test_fis301_agregar_video_al_lote_y_ajuste_de_grupo(orbita):
    at = _app_orbita(orbita).run()
    _sin_excepciones(at)
    at = at.number_input(key="orb_I::D0_20V.mp4").set_value(42.0).run()
    at = at.button(key="orb_guardar").click().run()
    _sin_excepciones(at)
    tabla = at.session_state["tabla_fis301"]
    assert len(tabla) == 1
    fila = tabla.iloc[0]
    assert (fila["bloque"], fila["V (V)"], fila["I (mA)"]) == ("D0", 20.0, 42.0)
    assert fila["v (mm/s)"] == pytest.approx(5.0, rel=0.02)
    assert fila["r (mm)"] == pytest.approx(5.0, rel=0.02)
    assert "D0_20V.mp4" in at.session_state["lote_crudos"]


def _tabla_lote():
    filas = []
    for b, k in (("D0", 0.09), ("D1", 0.07), ("D2", 0.05)):
        for V, I in ((10, 15.0), (14, 30.0), (18, 45.0), (22, 60.0)):
            filas.append({"archivo": f"{b}_{V}V.mp4", "bloque": b, "V (V)": float(V),
                          "I (mA)": I, "v (mm/s)": k * I, "s_v (mm/s)": 0.3,
                          "u_v (mm/s)": 0.4, "r (mm)": 5.0, "s_r (mm)": 0.1,
                          "d_mina (mm)": 5.0, "N": 200, "vueltas": 1.2,
                          "t_ini (s)": 0.1, "t_fin (s)": 7.9, "avisos": ""})
    return pd.DataFrame(filas)[COLUMNAS_FIS301]


def test_fis301_etapa_lote_muestra_la_tabla_editable(orbita):
    at = _app_orbita(orbita, etapa=ETAPA_LOTE)
    at.session_state["tabla_fis301"] = _tabla_lote()
    at = at.run()
    _sin_excepciones(at)
    nombres = [d.label for d in at.download_button]
    assert any("Tabla resumen" in n for n in nombres), nombres
    assert any("ZIP" in n for n in nombres), nombres


def test_fis301_etapa_lote_sin_calibracion_no_revienta():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120)
    at.session_state["etapa"] = ETAPA_LOTE
    at = at.run()
    _sin_excepciones(at)
    assert any("calibración" in w.value for w in at.warning)


def test_fis301_analisis_de_grupo_ajusta_por_el_origen():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120)
    at.session_state["tabla_fis301"] = _tabla_lote()
    at.session_state["preset_idx"] = IDX["fis301_lab3_lorentz"]
    at.session_state["etapa"] = ETAPA_GRUPO
    at = at.run()
    _sin_excepciones(at)
    nombres = [d.label for d in at.download_button]
    assert any("v(I)" in n for n in nombres) and any("v(V)" in n for n in nombres)
    tablas = [d.value for d in at.dataframe]
    ajustes = next(t for t in tablas if "k ± error" in t.columns)
    assert list(ajustes["bloque"]) == ["D0", "D1", "D2"]
    assert ajustes["k ((mm/s)/mA)"].iloc[0] == "0,0900"


def test_grupo_sin_datos_no_revienta():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120)
    at.session_state["etapa"] = ETAPA_GRUPO
    at = at.run()
    _sin_excepciones(at)


def test_flujo_de_un_objeto_no_activa_el_modo_robusto():
    """Labs 1 y 2: un solo objeto, sin selector de objetos ni seguimiento robusto."""
    video = RAIZ / "manual" / "ejemplos" / "carro.mp4"
    if not video.exists():
        pytest.skip("falta carro.mp4")
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120)
    at.session_state["video_path"] = str(video)
    at.session_state["reader"] = VideoReader(str(video))
    at.session_state["proc_start"] = 0
    at.session_state["proc_end"] = 40
    at.session_state["_last_interval"] = (0, 40)
    at.session_state["etapa"] = "3 · Tracking"
    at = at.run()
    _sin_excepciones(at)
    assert not [r for r in at.radio if r.key == "obj_sel_p3"]
    assert not any("robusto" in c.value for c in at.caption)


# ==========================================================================
# Subidas de archivos (lote, CSV y cámara fija)
# ==========================================================================
def test_fis301_lote_procesa_los_videos_subidos(orbita, tmp_path):
    ruta, datos, pts, m, o = orbita
    o2 = Orbita(rapidez=0.003, radio=0.004, horario=False)
    ruta2, _ = video_orbita(o2, tmp_path / "D1_12V.mp4")
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    at.session_state["calibration"] = Calibration(m_per_px=o.escala, origin_px=(0.0, 480.0))
    at.session_state["calib_size"] = (640, 480)
    at.session_state["muestra_trazador"] = m
    at.session_state["mina_px"] = datos["mina"]
    at.session_state["etapa"] = ETAPA_LOTE
    at = at.run()
    _sin_excepciones(at)
    up = [u for u in at.get("file_uploader") if u.key == "lote_up"][0]
    up.set_value([(p.name, p.read_bytes(), "video/mp4") for p in (ruta, ruta2)])
    at = at.run()
    _sin_excepciones(at)
    at = at.button(key="lote_go").click().run()
    _sin_excepciones(at)
    tabla = at.session_state["tabla_fis301"]
    assert list(tabla["archivo"]) == ["D0_20V.mp4", "D1_12V.mp4"]
    assert list(tabla["V (V)"]) == [20.0, 12.0]
    assert tabla["v (mm/s)"].iloc[0] == pytest.approx(5.0, rel=0.02)
    assert tabla["v (mm/s)"].iloc[1] == pytest.approx(3.0, rel=0.02)
    assert tabla["r (mm)"].iloc[1] == pytest.approx(4.0, rel=0.02)
    assert set(at.session_state["lote_crudos"]) == {"D0_20V.mp4", "D1_12V.mp4"}


def test_importar_la_tabla_del_lote_desde_csv():
    from core.formato import csv_es

    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120)
    at.session_state["etapa"] = ETAPA_LOTE
    at = at.run()
    up = [u for u in at.get("file_uploader") if u.key == "lote_csv_up"][0]
    up.set_value(("fis301_lab3_resumen.csv", csv_es(_tabla_lote()), "text/csv"))
    at = at.run()
    _sin_excepciones(at)
    tabla = at.session_state["tabla_fis301"]
    assert len(tabla) == 12
    assert tabla["I (mA)"].iloc[0] == pytest.approx(15.0)


def test_camara_fija_conserva_la_calibracion_al_subir_otro_video(colision):
    ruta, _, _, _ = colision
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180)
    cal = Calibration(m_per_px=0.0015, origin_px=(0.0, 300.0))
    at.session_state["calibration"] = cal
    at.session_state["calib_p1"] = (0.0, 300.0)
    at.session_state["calib_p2"] = (100.0, 300.0)
    at.session_state["calib_origin"] = (0.0, 300.0)
    at.session_state["calib_size"] = (1280, 480)
    at.session_state["mantener_calib"] = True
    at = at.run()
    up = [u for u in at.get("file_uploader") if u.label == "Sube tu video"][0]
    up.set_value(("ensayo02.mp4", ruta.read_bytes(), "video/mp4"))
    at = at.run()
    _sin_excepciones(at)
    assert at.session_state["reader"] is not None
    assert at.session_state["calibration"] is cal


def test_sin_camara_fija_la_calibracion_se_reinicia(colision):
    ruta, _, _, _ = colision
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180)
    at.session_state["calibration"] = Calibration(m_per_px=0.0015, origin_px=(0.0, 300.0))
    at = at.run()
    up = [u for u in at.get("file_uploader") if u.label == "Sube tu video"][0]
    up.set_value(("ensayo02.mp4", ruta.read_bytes(), "video/mp4"))
    at = at.run()
    _sin_excepciones(at)
    assert at.session_state["calibration"] is None
