"""Órbita del trazador alrededor de una mina (FIS301, Laboratorio 3).

El trazador de látex gira alrededor de una mina de grafito vertical, que en
la vista cenital es un punto. La guía pide, para cada vídeo:

  * la **rapidez** (magnitud de la velocidad) cuadro a cuadro;
  * su **promedio en el tramo de giro estable**, descartando el arranque, y su
    **desviación estándar cuadro a cuadro** ``s_v``, que es la incertidumbre
    de ``v`` (o la resolución, la mayor);
  * el **radio de la órbita**, que debe ser parecido entre puntos.

Aquí el radio sale de ajustar un círculo a la trayectoria (y, si el
estudiante marca la mina, también de la distancia de cada punto a ella).
Como control independiente se da la rapidez ``ω·R`` a partir del ángulo
barrido, que no sufre el sesgo del ruido sobre ``|v|``.

Todas las magnitudes van en unidades SI (m, s); la interfaz las muestra en mm.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.formato import desviacion, incertidumbre
from core.kinematics import finite_diff

#: Vueltas mínimas que pide la guía en cada vídeo.
VUELTAS_MIN = 1.0
#: Dispersión relativa del radio a partir de la cual se avisa.
RADIO_DISPERSION_AVISO = 0.15


@dataclass
class Circulo:
    """Círculo ajustado: centro (cx, cy), radio r y residuo radial rms."""

    cx: float
    cy: float
    r: float
    rms: float


def ajustar_circulo(x: np.ndarray, y: np.ndarray) -> Circulo:
    """Ajuste de un círculo por mínimos cuadrados geométricos.

    Parte del ajuste algebraico de Kåsa (lineal, siempre tiene solución) y lo
    refina con Gauss-Newton sobre las distancias radiales, que es el ajuste
    correcto cuando la órbita no es una vuelta completa.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para ajustar un círculo.")

    # Kåsa: x² + y² = 2a·x + 2b·y + c
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    rhs = x**2 + y**2
    (a, b, c), *_ = np.linalg.lstsq(A, rhs, rcond=None)
    r = float(np.sqrt(max(c + a * a + b * b, 0.0)))

    for _ in range(50):
        dx, dy = x - a, y - b
        d = np.hypot(dx, dy)
        d = np.where(d == 0, 1e-15, d)
        res = d - r
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])
        paso, *_ = np.linalg.lstsq(J, -res, rcond=None)
        a, b, r = a + paso[0], b + paso[1], r + paso[2]
        if np.max(np.abs(paso)) < 1e-12 * max(1.0, abs(r)):
            break
    d = np.hypot(x - a, y - b)
    return Circulo(float(a), float(b), float(abs(r)), float(np.sqrt(np.mean((d - r) ** 2))))


def rapidez(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rapidez cuadro a cuadro |v| por diferencias centradas (paso no uniforme)."""
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return np.full(len(t), np.nan)
    return np.hypot(finite_diff(x, t), finite_diff(y, t))


def detectar_arranque(t: np.ndarray, v: np.ndarray, fraccion: float = 0.90) -> float:
    """Instante en que termina el arranque: la rapidez llega a su régimen.

    Usa una mediana móvil (robusta a cuadros sueltos) y busca el primer
    instante en que alcanza ``fraccion`` del nivel final (mediana de la
    segunda mitad del vídeo). Si el giro ya era estable desde el inicio,
    devuelve el segundo punto: el primero se calcula con una diferencia de un
    solo lado y es menos preciso.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(t)
    if n < 4:
        return float(t[0]) if n else float("nan")
    nivel = float(np.nanmedian(v[n // 2:]))
    w = max(3, n // 15)
    med = np.array([np.nanmedian(v[max(0, i - w // 2): i + w // 2 + 1]) for i in range(n)])
    idx = 1
    if nivel > 0:
        cumple = np.nonzero(med >= fraccion * nivel)[0]
        if len(cumple):
            idx = max(1, int(cumple[0]))
    return float(t[min(idx, n - 1)])


@dataclass
class ResultadoOrbita:
    """Resultado del análisis de un vídeo de FIS301 (unidades SI)."""

    t_ini: float
    t_fin: float
    n: int
    rapidez_media: float
    rapidez_desv: float
    resolucion: float
    circulo: Circulo
    radio_medio: float           # distancia media de los puntos al centro ajustado
    radio_desv: float
    vueltas: float
    rapidez_angular: float       # ω·R (control, desde el ángulo barrido)
    sentido: str                 # "horario" o "antihorario", visto desde arriba
    mina: Optional[Tuple[float, float]] = None
    dist_mina_media: float = float("nan")
    dist_mina_desv: float = float("nan")
    centro_a_mina: float = float("nan")
    avisos: List[str] = field(default_factory=list)

    @property
    def incertidumbre(self) -> float:
        """La desviación estándar cuadro a cuadro o la resolución, la mayor."""
        return incertidumbre(self.rapidez_desv, self.resolucion)


def analizar_orbita(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mina: Optional[Tuple[float, float]] = None,
    t_ini: Optional[float] = None,
    t_fin: Optional[float] = None,
    resolucion: float = 0.0,
) -> ResultadoOrbita:
    """Analiza la órbita del trazador en el tramo de giro estable.

    ``x``, ``y`` en metros (vista cenital, eje y hacia arriba); ``mina`` en las
    mismas coordenadas. Si no se da ``t_ini``, el arranque se detecta solo.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    orden = np.argsort(t, kind="stable")
    t, x, y = t[orden], x[orden], y[orden]
    _, unicos = np.unique(t, return_index=True)
    t, x, y = t[unicos], x[unicos], y[unicos]
    if len(t) < 5:
        raise ValueError(
            f"Hay solo {len(t)} puntos del trazador; se necesitan al menos 5."
        )

    v = rapidez(t, x, y)
    if t_ini is None:
        t_ini = detectar_arranque(t, v)
    if t_fin is None:
        t_fin = float(t[-2]) if len(t) > 4 else float(t[-1])
    sel = (t >= t_ini - 1e-9) & (t <= t_fin + 1e-9)
    if sel.sum() < 3:
        raise ValueError("El tramo de giro estable tiene menos de 3 puntos.")

    ts, xs, ys, vs = t[sel], x[sel], y[sel], v[sel]
    circ = ajustar_circulo(xs, ys)
    d = np.hypot(xs - circ.cx, ys - circ.cy)

    # Ángulo barrido alrededor del centro ajustado: vueltas, sentido y ω·R.
    ang = np.unwrap(np.arctan2(ys - circ.cy, xs - circ.cx))
    barrido = float(ang[-1] - ang[0])
    vueltas = abs(barrido) / (2 * np.pi)
    omega = barrido / (ts[-1] - ts[0]) if ts[-1] > ts[0] else float("nan")
    if len(ts) >= 3:
        omega = float(np.polyfit(ts - ts[0], ang, 1)[0])
    sentido = "antihorario" if omega > 0 else "horario"

    res = ResultadoOrbita(
        t_ini=float(ts[0]),
        t_fin=float(ts[-1]),
        n=int(len(ts)),
        rapidez_media=float(np.mean(vs)),
        rapidez_desv=desviacion(vs),
        resolucion=float(resolucion),
        circulo=circ,
        radio_medio=float(np.mean(d)),
        radio_desv=desviacion(d),
        vueltas=float(vueltas),
        rapidez_angular=float(abs(omega) * circ.r),
        sentido=sentido,
    )

    if mina is not None:
        mx, my = float(mina[0]), float(mina[1])
        dm = np.hypot(xs - mx, ys - my)
        res.mina = (mx, my)
        res.dist_mina_media = float(np.mean(dm))
        res.dist_mina_desv = desviacion(dm)
        res.centro_a_mina = float(np.hypot(circ.cx - mx, circ.cy - my))
        if res.centro_a_mina > 0.5 * circ.r:
            res.avisos.append(
                "El centro de la órbita está lejos de la mina marcada "
                f"({res.centro_a_mina * 1000:.1f} mm, más de la mitad del "
                "radio). ¿Marcaste la mina correcta?"
            )

    if vueltas < VUELTAS_MIN:
        res.avisos.append(
            f"En el tramo analizado el trazador da {vueltas:.2f} vueltas; la "
            "guía pide al menos una vuelta completa."
        )
    if res.radio_medio > 0 and res.radio_desv / res.radio_medio > RADIO_DISPERSION_AVISO:
        res.avisos.append(
            "El radio de la órbita cambia más de un "
            f"{RADIO_DISPERSION_AVISO * 100:.0f} % entre puntos: el trazador no gira "
            "a distancia constante de la mina (o el seguimiento saltó)."
        )
    if (np.isfinite(res.rapidez_angular) and res.rapidez_angular > 0
            and res.rapidez_media > 1.05 * res.rapidez_angular):
        res.avisos.append(
            "La rapidez media cuadro a cuadro supera en más de un 5 % a la que "
            "sale del ángulo barrido (ω·R): el ruido del seguimiento la está "
            "inflando. Revisa el seguimiento o graba con más resolución."
        )
    if res.n < 10:
        res.avisos.append(
            f"Solo hay {res.n} puntos en el tramo estable: la desviación "
            "estándar es poco confiable."
        )
    return res
