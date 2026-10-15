"""Procesamiento completo de un vídeo de FIS301 para el modo por lotes.

Con la cámara fija, la calibración, el color del trazador y la posición de la
mina son los mismos en los 24 vídeos de la sesión. Cada vídeo se procesa
entero sin clics: en el primer cuadro se toma la mancha del color del
trazador más cercana a la mina, y desde ahí se sigue con
:func:`core.seguimiento.seguir_color`.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.calibration import Calibration
from core.colisiones import resolucion_velocidad
from core.orbita import ResultadoOrbita, analizar_orbita
from core.seguimiento import InfoSeguimiento, MuestraColor, seguir_color
from core.video import VideoReader


def procesar_video_fis301(
    ruta,
    calibracion: Calibration,
    muestra: MuestraColor,
    mina_px: Optional[Tuple[float, float]] = None,
    fps_capture: Optional[float] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    tam_calibracion: Optional[Tuple[int, int]] = None,
) -> Tuple[ResultadoOrbita, List[tuple], InfoSeguimiento]:
    """Sigue el trazador en todo el vídeo y analiza su órbita.

    Devuelve ``(resultado, puntos, info)``; ``puntos`` son las tuplas
    ``(frame, t, x_px, y_px)`` del seguimiento (los datos crudos).
    Lanza ``ValueError`` si el trazador no se encuentra o si el vídeo no
    tiene la resolución ``tam_calibracion`` (ancho, alto) con que se calibró.
    """
    reader = VideoReader(str(Path(ruta)))
    try:
        if tam_calibracion and (reader.width, reader.height) != tuple(tam_calibracion):
            raise ValueError(
                f"{Path(ruta).name} mide {reader.width}×{reader.height} px y la "
                f"calibración se hizo en un video de {tam_calibracion[0]}×"
                f"{tam_calibracion[1]} px: no se puede reutilizar."
            )
        reader.preload_range(0, reader.n_frames - 1)
        pts, info = seguir_color(
            reader, muestra, 0, reader.n_frames - 1, 1, fps_capture,
            semilla=mina_px, progress_cb=progress_cb,
        )
        fps = fps_capture or reader.fps
    finally:
        reader.release()
    if len(pts) < 5:
        raise ValueError(
            f"No se encontró el trazador en {Path(ruta).name} "
            f"({len(pts)} cuadros con el color elegido)."
        )
    t = np.array([p[1] for p in pts])
    x, y = calibracion.to_meters(np.array([p[2] for p in pts]), np.array([p[3] for p in pts]))
    mina = calibracion.to_meters(*mina_px) if mina_px is not None else None
    res = analizar_orbita(
        t, x, y, mina=mina,
        resolucion=resolucion_velocidad(calibracion.m_per_px, 1.0 / fps if fps else 0.0),
    )
    if info.n_perdidos > 0.2 * info.n_cuadros:
        res.avisos.append(
            f"El trazador no se detectó en {info.n_perdidos} de {info.n_cuadros} "
            "cuadros: revisa el color elegido o la iluminación."
        )
    return res, pts, info
