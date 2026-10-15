"""Verificación de extremo a extremo con vídeos sintéticos de resultado conocido.

Vídeo → seguimiento robusto → análisis, sin intervención manual salvo el
«clic» inicial sobre cada objeto. Tolerancia: 2 % sobre cada velocidad (o
2 % de la velocidad del carro 1 cuando el valor exacto es 0) y sobre C_R,
la rapidez y el radio de la órbita.

Para ver la tabla de valores esperados vs. obtenidos:
    python tests/verificar_lab3.py
"""

import numpy as np
import pytest

from core.autotrack import box_from_click
from core.calibration import Calibration
from core.colisiones import analizar_colision, calculos_colision, resolucion_velocidad
from core.orbita import analizar_orbita
from core.seguimiento import muestra_color, seguir_color, seguir_plantilla
from core.video import VideoReader
from tests.sinteticos import CASO_A, CASO_D, Orbita, video_colision, video_orbita

TOL = 0.02


def seguir_carros(ruta, datos, metodo):
    """Sigue los dos carros (mismo color de marca) y devuelve t, x (m) de cada uno."""
    r = VideoReader(ruta)
    r.preload_range(0, r.n_frames - 1)
    f0 = r.get_frame(0)
    excluir, salida = None, []
    for clave in ("px1", "px2"):
        x0, y0 = datos[clave][0], datos["y"]
        if metodo == "color":
            pts, info = seguir_color(r, muestra_color(f0, x0, y0), 0, r.n_frames - 1,
                                     semilla=(x0, y0), excluir=excluir)
        else:
            roi = box_from_click(x0, y0, 30, r.width, r.height)
            pts, info = seguir_plantilla(r, roi, 0, r.n_frames - 1, excluir=excluir)
        excluir = {p[0]: (p[2], p[3]) for p in pts}
        salida.append(pts)
    r.release()
    return salida


def analizar_ensayo(caso, ruta, datos, metodo):
    pts1, pts2 = seguir_carros(ruta, datos, metodo)
    esc = caso.escala
    t1 = np.array([p[1] for p in pts1]); x1 = np.array([p[2] for p in pts1]) * esc
    t2 = np.array([p[1] for p in pts2]); x2 = np.array([p[2] for p in pts2]) * esc
    res = analizar_colision(t1, x1, t2, x2,
                            resolucion=resolucion_velocidad(esc, 1 / caso.fps))
    calc = calculos_colision(caso.m1, caso.m2, **res.velocidades)
    return res, calc, (pts1, pts2)


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("lab3")
    return {
        "A": video_colision(CASO_A, d / "ensayo01.mp4"),
        "D": video_colision(CASO_D, d / "ensayo10.mp4"),
        "Dinv": video_colision(CASO_D, d / "ensayo11.mp4", invertir=True),
        "orb": video_orbita(Orbita(), d / "D0_20V.mp4"),
        "orb_arr": video_orbita(Orbita(t_arranque=1.5, horario=False, rapidez=0.003,
                                       radio=0.004, duracion=10.0), d / "D1_12V.mp4"),
    }


@pytest.mark.parametrize("clave,caso", [("A", CASO_A), ("D", CASO_D), ("Dinv", CASO_D)])
@pytest.mark.parametrize("metodo", ["color", "plantilla"])
def test_colision_sintetica(videos, clave, caso, metodo):
    ruta, datos = videos[clave]
    res, calc, (p1, p2) = analizar_ensayo(caso, ruta, datos, metodo)
    # El seguimiento no confunde los carros (misma marca) ni pierde cuadros.
    n = len(datos["t"])
    assert len(p1) == n and len(p2) == n
    for pts, clave_px in ((p1, "px1"), (p2, "px2")):
        err = max(abs(p[2] - datos[clave_px][p[0]]) for p in pts)
        assert err < 2.0, f"el seguimiento se desvió {err:.1f} px"
    exactos = {"v1i": caso.v1i, "v2i": caso.v2i, "v1f": caso.v1f, "v2f": caso.v2f}
    escala_v = abs(caso.v1i)
    for k, ex in exactos.items():
        assert res.velocidades[k] == pytest.approx(ex, abs=TOL * max(abs(ex), escala_v)), k
    assert calc.C_R == pytest.approx(caso.e, rel=TOL)
    assert res.t_choque == pytest.approx(caso.t_choque, abs=0.05)
    assert res.signo == (-1 if clave == "Dinv" else 1)
    assert not any("segundo choque" in a for a in res.avisos)


@pytest.mark.parametrize("clave", ["orb", "orb_arr"])
@pytest.mark.parametrize("metodo", ["color", "plantilla"])
def test_orbita_sintetica(videos, clave, metodo):
    ruta, datos = videos[clave]
    o = Orbita() if clave == "orb" else Orbita(t_arranque=1.5, horario=False,
                                              rapidez=0.003, radio=0.004, duracion=10.0)
    r = VideoReader(ruta)
    r.preload_range(0, r.n_frames - 1)
    f0 = r.get_frame(0)
    x0, y0 = datos["cx"][0], datos["cy"][0]
    if metodo == "color":
        pts, info = seguir_color(r, muestra_color(f0, x0, y0), 0, r.n_frames - 1,
                                 semilla=(x0, y0))
    else:
        roi = box_from_click(x0, y0, 16, r.width, r.height)
        pts, info = seguir_plantilla(r, roi, 0, r.n_frames - 1)
    # Los cuadros en que el trazador no se ve quedan sin punto (no se inventan).
    assert set(o.perdidos) <= set(info.perdidos)
    err = max(np.hypot(p[2] - datos["cx"][p[0]], p[3] - datos["cy"][p[0]]) for p in pts)
    assert err < 1.5, f"el seguimiento se desvió {err:.1f} px (¿saltó a la mina?)"

    cal = Calibration(m_per_px=o.escala, origin_px=(0.0, float(r.height)))
    t = np.array([p[1] for p in pts])
    x, y = cal.to_meters(np.array([p[2] for p in pts]), np.array([p[3] for p in pts]))
    res = analizar_orbita(t, x, y, mina=cal.to_meters(*datos["mina"]),
                          resolucion=resolucion_velocidad(o.escala, 1 / o.fps))
    assert res.rapidez_media == pytest.approx(o.rapidez, rel=TOL)
    assert res.radio_medio == pytest.approx(o.radio, rel=TOL)
    assert res.dist_mina_media == pytest.approx(o.radio, rel=TOL)
    assert res.sentido == ("horario" if o.horario else "antihorario")
    if o.t_arranque:
        assert res.t_ini > 0.8 * o.t_arranque
    r.release()


def test_lote_fis301_procesa_varios_videos_con_la_misma_calibracion(videos, tmp_path):
    """Modo por lotes: una calibración, color y mina para todos los vídeos."""
    from core.lote import fila_fis301, ordenar_fis301, reemplazar_fila, tabla_fis301_vacia
    from core.procesar import procesar_video_fis301

    o = Orbita()
    cal = Calibration(m_per_px=o.escala, origin_px=(0.0, 480.0))
    ruta, datos = videos["orb"]
    r = VideoReader(ruta)
    m = muestra_color(r.get_frame(0), datos["cx"][0], datos["cy"][0])
    r.release()
    tabla = tabla_fis301_vacia()
    for clave in ("orb", "orb_arr"):
        ruta_k, _ = videos[clave]
        res, pts, info = procesar_video_fis301(ruta_k, cal, m, mina_px=datos["mina"])
        tabla = reemplazar_fila(tabla, fila_fis301(ruta_k.name, res))
    tabla = ordenar_fis301(tabla)
    assert list(tabla["bloque"]) == ["D0", "D1"]
    assert list(tabla["V (V)"]) == [20.0, 12.0]
    assert tabla.loc[0, "v (mm/s)"] == pytest.approx(5.0, rel=TOL)
    assert tabla.loc[1, "v (mm/s)"] == pytest.approx(3.0, rel=TOL)
