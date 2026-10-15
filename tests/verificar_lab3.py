"""Tabla de verificación de los Laboratorios 3 con vídeos sintéticos.

Genera los vídeos de ``tests/sinteticos.py`` en una carpeta temporal, los
procesa igual que la app (seguimiento robusto → análisis) e imprime, en
Markdown, los valores esperados frente a los obtenidos.

    python tests/verificar_lab3.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

from core.autotrack import box_from_click  # noqa: E402
from core.calibration import Calibration  # noqa: E402
from core.colisiones import resolucion_velocidad  # noqa: E402
from core.orbita import analizar_orbita  # noqa: E402
from core.seguimiento import muestra_color, seguir_color, seguir_plantilla  # noqa: E402
from core.video import VideoReader  # noqa: E402
from tests.sinteticos import CASO_A, CASO_D, Orbita, video_colision, video_orbita  # noqa: E402
from tests.test_lab3_videos import analizar_ensayo  # noqa: E402


def _err(obt, esp, ref):
    """Error relativo (%), respecto de ``ref`` cuando el valor exacto es 0."""
    base = abs(esp) if esp != 0 else abs(ref)
    return 100.0 * abs(obt - esp) / base


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows (cp1252)
    except Exception:
        pass
    d = Path(tempfile.mkdtemp(prefix="lab3_"))
    print("### FIS101 — dos carros con la misma marca de color\n")
    print("| Caso | Método | Magnitud | Esperado | Obtenido | Error (%) |")
    print("|---|---|---|---|---|---|")
    casos = [("A (elástico, e = 1)", CASO_A, False, "ensayo01.mp4"),
             ("D (inelástico, e = 0,7)", CASO_D, False, "ensayo10.mp4"),
             ("D con el eje invertido", CASO_D, True, "ensayo11.mp4")]
    peor = 0.0
    for nombre, caso, inv, archivo in casos:
        ruta, datos = video_colision(caso, d / archivo, invertir=inv)
        for metodo in ("color", "plantilla"):
            res, calc, _ = analizar_ensayo(caso, ruta, datos, metodo)
            v = res.velocidades
            filas = [("v1i (m/s)", caso.v1i, v["v1i"]), ("v2i (m/s)", caso.v2i, v["v2i"]),
                     ("v1f (m/s)", caso.v1f, v["v1f"]), ("v2f (m/s)", caso.v2f, v["v2f"]),
                     ("C_R", caso.e, calc.C_R)]
            for mag, esp, obt in filas:
                e = _err(obt, esp, caso.v1i)
                peor = max(peor, e)
                print(f"| {nombre} | {metodo} | {mag} | {esp:+.4f} | {obt:+.4f} | {e:.2f} |"
                      .replace(".", ","))
    print(f"\nPeor error FIS101: {peor:.2f} %".replace(".", ","))

    print("\n### FIS301 — trazador en órbita (ruido, burbujas y cuadros perdidos)\n")
    print("| Caso | Método | Magnitud | Esperado | Obtenido | Error (%) |")
    print("|---|---|---|---|---|---|")
    orbitas = [("R = 5 mm, v = 5 mm/s, horario", Orbita(), "D0_20V.mp4"),
               ("R = 4 mm, v = 3 mm/s, antihorario, arranque 1,5 s",
                Orbita(t_arranque=1.5, horario=False, rapidez=0.003, radio=0.004,
                       duracion=10.0), "D1_12V.mp4")]
    peor = 0.0
    for nombre, o, archivo in orbitas:
        ruta, datos = video_orbita(o, d / archivo)
        for metodo in ("color", "plantilla"):
            r = VideoReader(ruta)
            r.preload_range(0, r.n_frames - 1)
            x0, y0 = datos["cx"][0], datos["cy"][0]
            if metodo == "color":
                pts, info = seguir_color(r, muestra_color(r.get_frame(0), x0, y0),
                                         0, r.n_frames - 1, semilla=(x0, y0))
            else:
                pts, info = seguir_plantilla(r, box_from_click(x0, y0, 16, r.width, r.height),
                                             0, r.n_frames - 1)
            r.release()
            cal = Calibration(m_per_px=o.escala, origin_px=(0.0, 480.0))
            t = np.array([p[1] for p in pts])
            x, y = cal.to_meters(np.array([p[2] for p in pts]), np.array([p[3] for p in pts]))
            res = analizar_orbita(t, x, y, mina=cal.to_meters(*datos["mina"]),
                                  resolucion=resolucion_velocidad(o.escala, 1 / o.fps))
            filas = [("v (mm/s)", o.rapidez * 1e3, res.rapidez_media * 1e3),
                     ("r (mm)", o.radio * 1e3, res.radio_medio * 1e3),
                     ("d_mina (mm)", o.radio * 1e3, res.dist_mina_media * 1e3)]
            for mag, esp, obt in filas:
                e = _err(obt, esp, esp)
                peor = max(peor, e)
                print(f"| {nombre} | {metodo} | {mag} | {esp:.3f} | {obt:.3f} | {e:.2f} |"
                      .replace(".", ","))
            print(f"| {nombre} | {metodo} | s_v / resolución (mm/s) | — | "
                  f"{res.rapidez_desv * 1e3:.3f} / {res.resolucion * 1e3:.3f} | — |"
                  .replace(".", ","))
            print(f"| {nombre} | {metodo} | cuadros sin punto | {len(o.perdidos)} ocultos | "
                  f"{info.n_perdidos} | — |")
            print(f"| {nombre} | {metodo} | inicio del tramo estable (s) | "
                  f"{'≥ 1,2' if o.t_arranque else '≈ 0'} | {res.t_ini:.2f} | — |"
                  .replace(".", ","))
    print(f"\nPeor error FIS301: {peor:.2f} %".replace(".", ","))


if __name__ == "__main__":
    main()
