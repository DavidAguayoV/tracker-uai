"""Lectura de video y navegación frame a frame con OpenCV.

Este módulo es independiente de Streamlit: se puede usar y testear
por separado. La interfaz (app.py) se encarga de guardar el video
subido a un archivo temporal y de mostrar los frames.
"""

from __future__ import annotations

import os

# Fuerza a FFmpeg a decodificar en un solo hilo. Evita el crash nativo
# "Assertion fctx->async_lock failed ... pthread_frame.c" que ocurre al leer
# algunos .mov (a menudo HEVC/H.265) con decodificación multihilo en Windows.
# Debe definirse ANTES de crear cualquier cv2.VideoCapture.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Metadatos básicos del video."""

    path: str
    n_frames: int
    fps: float
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        """Duración aproximada en segundos según el fps del contenedor."""
        if self.fps > 0:
            return self.n_frames / self.fps
        return 0.0


def save_uploaded_to_temp(uploaded: Any, suffix: str = ".mp4") -> Path:
    """Guarda un video subido en un archivo temporal, por bloques.

    OpenCV necesita una ruta en disco (no acepta el buffer en memoria
    que entrega ``st.file_uploader``), por eso escribimos a un temporal.
    Acepta un objeto tipo archivo (con ``.read()``) o bytes crudos. La
    copia por bloques evita cargar videos grandes completos en memoria.

    Devuelve la ruta al archivo temporal creado.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        if hasattr(uploaded, "read"):
            try:
                uploaded.seek(0)
            except Exception:
                pass
            shutil.copyfileobj(uploaded, tmp, length=4 * 1024 * 1024)
        else:
            tmp.write(uploaded)
    finally:
        tmp.flush()
        tmp.close()
    return Path(tmp.name)


class VideoReader:
    """Envoltorio simple sobre ``cv2.VideoCapture``.

    Mantiene la captura abierta para navegar por frames sin reabrir el
    archivo en cada movimiento del slider. Los frames se devuelven en
    formato RGB (OpenCV usa BGR internamente).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.warnings: list[str] = []
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(
                "No se pudo abrir el video. Puede estar dañado o en un formato "
                "no soportado por OpenCV. Formatos recomendados: mp4/mov/avi/mkv "
                "con códec H.264."
            )
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._last_idx: int = -1
        # Caché de frames (JPEG en memoria) para evitar seeks aleatorios, que
        # hacen caer a FFmpeg con algunos .mov/HEVC.
        self._cache: dict[int, bytes] = {}
        self._cache_order: list[int] = []
        self._cache_max: int = 2000
        self._ts_cache: dict[int, float] = {}   # timestamp (ms) por frame

        # Verifica que el primer frame sea decodificable (detecta códecs no
        # soportados, p. ej. HEVC/H.265 de algunos iPhone).
        ok, _ = self.cap.read()
        if not ok:
            self.release()
            raise ValueError(
                "El video se abrió pero no se pudo decodificar (posible códec "
                "HEVC/H.265 no soportado). Solución: convierte el video a H.264, "
                "por ejemplo con:  ffmpeg -i entrada.mov -c:v libx264 salida.mp4"
            )
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._last_idx = -1

        # Sanea metadatos poco fiables.
        if self.fps <= 0 or self.fps > 1000:
            self.fps = 30.0
            self.warnings.append(
                "El fps del archivo no es fiable; se asume 30 fps. Usa el fps "
                "de captura manual si conoces el valor real."
            )
        if self.n_frames <= 0:
            # Algunos contenedores no reportan el total; se estima por duración.
            dur_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.n_frames = max(1, int(self.fps * (dur_ms / 1000.0))) if dur_ms else 0
            if self.n_frames <= 0:
                self.release()
                raise ValueError(
                    "No se pudo determinar la cantidad de frames del video. "
                    "Reencódalo a mp4/H.264 e inténtalo de nuevo."
                )
            self.warnings.append(
                "El contenedor no reporta el total de frames; se estimó por "
                "duración y podría ser aproximado."
            )

    @property
    def metadata(self) -> VideoMetadata:
        """Devuelve los metadatos del video."""
        return VideoMetadata(
            path=self.path,
            n_frames=self.n_frames,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )

    def _clamp(self, idx: int) -> int:
        """Restringe el índice al rango válido de frames."""
        if self.n_frames > 0:
            return int(max(0, min(idx, self.n_frames - 1)))
        return max(0, int(idx))

    def _cache_put(self, idx: int, frame_rgb: np.ndarray) -> None:
        """Guarda un frame en el caché (comprimido JPEG) con límite LRU."""
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        if idx not in self._cache:
            self._cache_order.append(idx)
        self._cache[idx] = buf.tobytes()
        while len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def _cache_get(self, idx: int) -> Optional[np.ndarray]:
        """Devuelve un frame del caché como RGB, o None si no está."""
        data = self._cache.get(idx)
        if data is None:
            return None
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def preload_range(self, start: int, end: int, step: int = 1,
                      progress_cb=None) -> int:
        """Precarga frames [start, end] en caché con una lectura secuencial.

        La lectura secuencial es estable (evita los seeks aleatorios que hacen
        caer a FFmpeg con algunos .mov/HEVC). Devuelve cuántos frames cargó.
        """
        start = self._clamp(start)
        end = self._clamp(end)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        self._last_idx = start - 1
        i = start
        loaded = 0
        total = max(end - start + 1, 1)
        while i <= end and len(self._cache_order) < self._cache_max:
            ts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))  # timestamp del frame i
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            self._last_idx = i
            if (i - start) % step == 0:
                self._cache_put(i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self._ts_cache[i] = ts
                loaded += 1
            if progress_cb:
                progress_cb((i - start + 1) / total)
            i += 1
        return loaded

    def get_frame(self, idx: int) -> np.ndarray:
        """Devuelve el frame ``idx`` como arreglo RGB (alto, ancho, 3).

        Primero busca en caché (evita seeks aleatorios inestables); si no está,
        lo lee del video y lo guarda en caché.
        """
        idx = self._clamp(idx)
        cached = self._cache_get(idx)
        if cached is not None:
            return cached
        # Evita re-seek si pedimos el siguiente frame consecutivo.
        if idx != self._last_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # Reintenta con un seek explícito antes de rendirse.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
        self._last_idx = idx
        if not ok or frame is None:
            raise ValueError(
                f"No se pudo leer el frame {idx}. El video puede estar truncado "
                "o el frame final mal indexado."
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._cache_put(idx, rgb)
        return rgb

    def get_timestamp_ms(self, idx: int) -> float:
        """Timestamp real del frame en milisegundos (según el contenedor).

        Usa ``CAP_PROP_POS_MSEC``, más robusto que ``frame/fps`` cuando el
        video tiene frame rate variable (típico en celulares).
        """
        idx = self._clamp(idx)
        if idx in self._ts_cache:          # evita seek si ya está en caché
            return self._ts_cache[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.cap.grab()
        self._last_idx = idx
        ts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        self._ts_cache[idx] = ts
        return ts

    def frame_time(self, idx: int, fps_capture: Optional[float] = None) -> float:
        """Tiempo físico del frame en segundos.

        Reglas (según decisiones del proyecto):
          * Si se entrega ``fps_capture`` (p. ej. cámara lenta a 240 fps),
            el tiempo es ``idx / fps_capture``. Esto corrige el caso en que
            el video se reproduce a otra velocidad que la de captura.
          * Si no, se usa el timestamp real del contenedor.
          * Si el timestamp no es válido (0 o negativo en frames > 0), se
            cae a ``idx / fps`` como último recurso.
        """
        idx = self._clamp(idx)
        if fps_capture and fps_capture > 0:
            return idx / fps_capture
        ts_ms = self.get_timestamp_ms(idx)
        if ts_ms > 0 or idx == 0:
            return ts_ms / 1000.0
        if self.fps > 0:
            return idx / self.fps
        return 0.0

    def ghost_overlay(
        self,
        indices: list[int],
        diff_thresh: int = 30,
        max_frames: int = 40,
    ) -> np.ndarray:
        """Compone una imagen multiexposición ("fantasma") de varios frames.

        Estima el fondo como la mediana de los frames muestreados y pega
        encima los píxeles del objeto en movimiento (los que difieren del
        fondo) en cada frame. El resultado muestra la trayectoria completa
        del objeto sobre una sola imagen.
        """
        if not indices:
            raise ValueError("Se necesita al menos un frame para el overlay.")
        # Submuestrea para no procesar cientos de frames.
        if len(indices) > max_frames:
            sel = np.linspace(0, len(indices) - 1, max_frames).astype(int)
            indices = [indices[i] for i in sel]
        frames = [self.get_frame(i) for i in indices]
        stack = np.stack(frames).astype(np.float32)
        bg = np.median(stack, axis=0).astype(np.uint8)
        result = bg.copy()
        for f in frames:
            diff = np.abs(f.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
            mask = diff > diff_thresh
            result[mask] = f[mask]
        return result

    def release(self) -> None:
        """Libera la captura de OpenCV."""
        if self.cap is not None:
            self.cap.release()

    def __del__(self) -> None:  # pragma: no cover - limpieza best-effort
        try:
            self.release()
        except Exception:
            pass
