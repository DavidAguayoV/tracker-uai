"""Colisiones entre dos carros (FIS101, Laboratorio 3).

Del vídeo salen las posiciones de los dos carros a lo largo del riel. Este
módulo hace lo que la guía pide hacer con *Tracker*:

  1. Derivar la velocidad cuadro a cuadro (diferencias centradas).
  2. Detectar el **instante del choque**, que es el salto de la velocidad.
  3. Separar los **tramos de velocidad constante** antes y después,
     descartando un par de puntos junto al salto.
  4. Promediar cada tramo: ``v1i``, ``v2i``, ``v1f`` y ``v2f``, cada uno con la
     desviación estándar de sus puntos.

Además calcula, si se dan las masas, las magnitudes de la guía:
``p_i``, ``p_f``, ``Δp``, ``Δp/p_i``, ``K_i``, ``K_f`` y ``C_R``.

**Convención de signos** (la de la guía): eje x paralelo al riel y positivo en
el sentido en que parte el carro 1. Si en el vídeo el carro 1 avanza hacia −x,
todas las velocidades se multiplican por −1 (``ResultadoColision.signo``).

El choque se detecta sobre la **velocidad relativa** ``v2 − v1``: es la que
siempre salta más (de ``u_i`` a ``−C_R·u_i``), también cuando un carro está
quieto o cuando con masas iguales los carros intercambian velocidades.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.formato import desviacion, incertidumbre
from core.kinematics import finite_diff

#: Puntos que se descartan a cada lado del salto, además de la transición.
N_DESCARTE = 2
#: Puntos mínimos que la guía considera razonables para promediar un tramo.
N_MIN_TRAMO = 8
#: Ventana (en puntos) para buscar saltos secundarios (choque con un tope).
VENTANA_SALTO = 3

#: Bloques del plan de ensayos de la guía (Tabla «Plan de ensayos»).
BLOQUES = {
    "A": "masas iguales · carro 2 en reposo",
    "B": "masas iguales · carro 2 lanzado en sentido contrario",
    "C": "masas distintas · carro 2 en reposo",
    "D": "masas distintas · carro 2 lanzado en sentido contrario",
}


# ==========================================================================
# Estructuras de resultado
# ==========================================================================
@dataclass
class Tramo:
    """Intervalo de tiempo [t_ini, t_fin] (s), ambos extremos incluidos."""

    t_ini: float
    t_fin: float

    def contiene(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (t >= self.t_ini - 1e-9) & (t <= self.t_fin + 1e-9)


@dataclass
class VelocidadTramo:
    """Promedio de la velocidad en un tramo, con la dispersión de sus puntos."""

    media: float
    desv: float
    n: int
    resolucion: float = 0.0

    @property
    def incertidumbre(self) -> float:
        """La desviación estándar o la resolución, la mayor (sin redondear)."""
        return incertidumbre(self.desv, self.resolucion)


@dataclass
class ResultadoColision:
    """Resultado del análisis de un ensayo."""

    t_choque: float
    antes: Tramo
    despues: Tramo
    v1i: VelocidadTramo
    v1f: VelocidadTramo
    v2i: Optional[VelocidadTramo]
    v2f: Optional[VelocidadTramo]
    signo: int = 1
    transicion: Tramo = None
    saltos_extra: List[Tuple[str, float, bool]] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)

    @property
    def velocidades(self) -> dict:
        """Las cuatro velocidades con signo (m/s); ``nan`` si falta el carro 2."""
        nan = float("nan")
        return {
            "v1i": self.v1i.media,
            "v2i": self.v2i.media if self.v2i else nan,
            "v1f": self.v1f.media,
            "v2f": self.v2f.media if self.v2f else nan,
        }


# ==========================================================================
# Utilidades numéricas
# ==========================================================================
def resolucion_velocidad(m_per_px: float, dt: float) -> float:
    """Resolución de una velocidad cuadro a cuadro (m/s).

    Mover un punto en un píxel cambia la velocidad por diferencias centradas
    en ``escala / (2·Δt)``. Es la «resolución de la calibración» llevada a
    velocidad.
    """
    if not (m_per_px and dt) or m_per_px <= 0 or dt <= 0:
        return 0.0
    return float(m_per_px / (2.0 * dt))


def _ruido(v: np.ndarray) -> float:
    """Ruido punto a punto de una señal, robusto frente a saltos (MAD de diferencias)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return 0.0
    d = np.diff(v)
    mad = float(np.median(np.abs(d - np.median(d))))
    return 1.4826 * mad / np.sqrt(2.0)


def _mejor_corte(u: np.ndarray, minimo: int = 3) -> int:
    """Índice del primer punto del segundo tramo en el mejor ajuste a dos escalones."""
    n = len(u)
    if n < 2 * minimo:
        raise ValueError(
            f"Hay solo {n} puntos: se necesitan al menos {2 * minimo} para "
            "separar un tramo antes y otro después del choque."
        )
    c1 = np.cumsum(u)
    c2 = np.cumsum(u * u)
    mejor_k, mejor_sse = minimo, np.inf
    for k in range(minimo, n - minimo + 1):
        sl, ql = c1[k - 1], c2[k - 1]
        sr, qr = c1[-1] - sl, c2[-1] - ql
        sse = (ql - sl * sl / k) + (qr - sr * sr / (n - k))
        if sse < mejor_sse:
            mejor_k, mejor_sse = k, sse
    return int(mejor_k)


def detectar_saltos(
    t: np.ndarray,
    v: np.ndarray,
    umbral: float,
    ventana: int = VENTANA_SALTO,
) -> List[float]:
    """Instantes donde la velocidad cambia de golpe más que ``umbral``.

    Compara el promedio de ``ventana`` puntos a cada lado de cada instante. Un
    choque con un tope invierte la velocidad y produce un salto grande; la
    fricción la cambia poco a poco y no lo produce.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    w = int(max(1, ventana))
    n = len(v)
    if n < 2 * w + 1 or not np.isfinite(umbral) or umbral <= 0:
        return []
    d = np.full(n, 0.0)
    for i in range(w, n - w + 1):
        d[i] = np.mean(v[i:i + w]) - np.mean(v[i - w:i])
    sobre = np.abs(d) > umbral
    saltos = []
    i = 0
    while i < n:
        if sobre[i]:
            j = i
            while j + 1 < n and sobre[j + 1]:
                j += 1
            k = i + int(np.argmax(np.abs(d[i:j + 1])))
            saltos.append(float(0.5 * (t[k - 1] + t[k])) if k > 0 else float(t[k]))
            i = j + 1
        else:
            i += 1
    return saltos


def promedio_tramo(
    t: np.ndarray, v: np.ndarray, tramo: Tramo, resolucion: float = 0.0
) -> VelocidadTramo:
    """Promedio y desviación estándar (1/(n−1)) de ``v`` dentro del tramo."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    sel = tramo.contiene(t) & np.isfinite(v)
    vals = v[sel]
    if len(vals) == 0:
        return VelocidadTramo(float("nan"), float("nan"), 0, resolucion)
    return VelocidadTramo(
        media=float(np.mean(vals)),
        desv=desviacion(vals),
        n=int(len(vals)),
        resolucion=float(resolucion),
    )


def _limpiar(t, x) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    orden = np.argsort(t, kind="stable")
    t, x = t[orden], x[orden]
    # Tiempos repetidos darían velocidades infinitas.
    _, unicos = np.unique(t, return_index=True)
    return t[unicos], x[unicos]


def velocidad(t, x) -> Tuple[np.ndarray, np.ndarray]:
    """Tiempo y velocidad cuadro a cuadro (diferencias centradas) de una posición."""
    t, x = _limpiar(t, x)
    if len(t) < 2:
        return t, np.full(len(t), np.nan)
    return t, finite_diff(x, t)


# ==========================================================================
# Análisis de un ensayo
# ==========================================================================
def analizar_colision(
    t1: np.ndarray,
    x1: np.ndarray,
    t2: Optional[np.ndarray] = None,
    x2: Optional[np.ndarray] = None,
    antes: Optional[Tramo] = None,
    despues: Optional[Tramo] = None,
    n_descartar: int = N_DESCARTE,
    resolucion: float = 0.0,
    n_min: int = N_MIN_TRAMO,
) -> ResultadoColision:
    """Analiza un ensayo: choque, tramos y las cuatro velocidades con signo.

    Parámetros
    ----------
    t1, x1 : tiempo (s) y posición a lo largo del riel (m) del carro 1.
    t2, x2 : lo mismo para el carro 2 (opcional: sin él solo hay v1i y v1f).
    antes, despues : tramos elegidos a mano. Si faltan, se proponen solos.
    n_descartar : puntos extra que se descartan a cada lado del salto.
    resolucion : resolución de una velocidad cuadro a cuadro (m/s), ver
        :func:`resolucion_velocidad`. Entra en la incertidumbre.
    n_min : bajo este número de puntos en un tramo se avisa.
    """
    ta, va = velocidad(t1, x1)
    hay2 = t2 is not None and x2 is not None and len(np.asarray(t2)) >= 2
    if hay2:
        tb, vb = velocidad(t2, x2)
    else:
        tb = vb = None
    if len(ta) < 6:
        raise ValueError(
            "El carro 1 tiene muy pocos puntos para analizar el choque "
            f"({len(ta)}). Síguelo en todo el tramo del vídeo."
        )

    # --- Señal para detectar el choque: velocidad relativa en una malla común
    if hay2 and len(tb) >= 2:
        lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
        sel = (ta >= lo) & (ta <= hi)
        tc = ta[sel]
        if len(tc) >= 6:
            u = np.interp(tc, tb, vb) - va[sel]
        else:
            tc, u = ta, -va
    else:
        tc, u = ta, -va

    k = _mejor_corte(u)
    L = float(np.median(u[:k]))
    R = float(np.median(u[k:]))
    salto = abs(R - L)
    sigma = _ruido(u)
    tol = max(4.0 * sigma, 0.1 * salto, 1e-12)
    a = k - 1
    while a > 0 and abs(u[a] - L) > tol:
        a -= 1
    b = k
    while b < len(u) - 1 and abs(u[b] - R) > tol:
        b += 1
    t_choque = float(0.5 * (tc[k - 1] + tc[k]))
    transicion = Tramo(float(tc[a]), float(tc[b]))

    # --- Tramos propuestos ---------------------------------------------------
    nd = int(max(0, n_descartar))
    ia = max(0, a - nd)
    ib = min(len(tc) - 1, b + nd)
    auto_antes = Tramo(float(tc[0]), float(tc[ia]))
    auto_despues = Tramo(float(tc[ib]), float(tc[-1]))

    # Escala de velocidades del ensayo, para los umbrales de saltos secundarios.
    escala = max(abs(L), abs(R), 1e-9)
    dt_med = float(np.median(np.diff(tc))) if len(tc) > 1 else 0.0
    margen = (VENTANA_SALTO + nd) * dt_med
    zona_choque = Tramo(transicion.t_ini - margen, transicion.t_fin + margen)

    carros = [("carro 1", ta, va)]
    if hay2:
        carros.append(("carro 2", tb, vb))
    extra: List[Tuple[str, float]] = []
    for nombre, tt, vv in carros:
        umbral = max(6.0 * _ruido(vv) * np.sqrt(2.0 / VENTANA_SALTO), 0.25 * escala)
        for ts in detectar_saltos(tt, vv, umbral):
            if not zona_choque.contiene(np.array([ts]))[0]:
                extra.append((nombre, ts))

    # El tramo automático se corta antes de un segundo choque (con un tope) y
    # después de un empujón previo, para no promediar dos velocidades distintas.
    for _, ts in extra:
        if ts > t_choque and ts - margen > auto_despues.t_ini and ts < auto_despues.t_fin:
            auto_despues = Tramo(auto_despues.t_ini, float(ts - margen))
        if ts < t_choque and ts + margen < auto_antes.t_fin and ts > auto_antes.t_ini:
            auto_antes = Tramo(float(ts + margen), auto_antes.t_fin)

    antes = antes or auto_antes
    despues = despues or auto_despues

    # --- Promedios -----------------------------------------------------------
    v1i = promedio_tramo(ta, va, antes, resolucion)
    v1f = promedio_tramo(ta, va, despues, resolucion)
    v2i = promedio_tramo(tb, vb, antes, resolucion) if hay2 else None
    v2f = promedio_tramo(tb, vb, despues, resolucion) if hay2 else None

    # --- Convención de signos: + en el sentido en que parte el carro 1 ------
    signo = 1
    if np.isfinite(v1i.media) and v1i.media < 0:
        signo = -1
        for vt in (v1i, v1f, v2i, v2f):
            if vt is not None:
                vt.media = -vt.media

    # --- Avisos (no bloquean) -----------------------------------------------
    avisos: List[str] = []
    saltos_extra = []
    for nombre, ts in extra:
        dentro = antes.contiene(np.array([ts]))[0] or despues.contiene(np.array([ts]))[0]
        saltos_extra.append((nombre, float(ts), bool(dentro)))
        if dentro:
            avisos.append(
                f"Posible segundo choque del {nombre} en t ≈ {ts:.2f} s, "
                "dentro de un tramo promediado. La guía pide un único choque "
                "en el intervalo analizado: acorta el tramo para dejarlo fuera."
            )
    for etiqueta, vt in (("v1i", v1i), ("v1f", v1f), ("v2i", v2i), ("v2f", v2f)):
        if vt is None:
            continue
        if vt.n == 0:
            avisos.append(f"El tramo de {etiqueta} no tiene puntos: revisa sus límites.")
        elif vt.n < n_min:
            avisos.append(
                f"El tramo de {etiqueta} tiene solo {vt.n} puntos (se recomiendan "
                f"al menos {n_min}). El promedio es poco confiable."
            )
    if not hay2:
        avisos.append("Falta el carro 2: solo se calcularon v1i y v1f.")
    elif v1i.n and v2i is not None and v2i.n and np.isfinite(v1i.media):
        if abs(v1i.media) < 1e-9:
            avisos.append("El carro 1 parece quieto antes del choque.")

    return ResultadoColision(
        t_choque=t_choque,
        antes=antes,
        despues=despues,
        v1i=v1i,
        v1f=v1f,
        v2i=v2i,
        v2f=v2f,
        signo=signo,
        transicion=transicion,
        saltos_extra=saltos_extra,
        avisos=avisos,
    )


# ==========================================================================
# Magnitudes de la guía
# ==========================================================================
@dataclass
class CalculosColision:
    """Magnitudes derivadas de un ensayo (unidades SI)."""

    p_i: float
    p_f: float
    dp: float
    dp_rel: float          # Δp / p_i, como fracción (×100 para %)
    K_i: float
    K_f: float
    C_R: float
    escala_p: float        # |m1·v1i| + |m2·v2i|

    @property
    def p_i_pequeno(self) -> bool:
        """``True`` si p_i casi se anula (bloques B y D): Δp/p_i no es confiable."""
        if not np.isfinite(self.p_i) or not np.isfinite(self.escala_p):
            return False
        return abs(self.p_i) < 0.25 * self.escala_p


def calculos_colision(
    m1: float, m2: float, v1i: float, v2i: float, v1f: float, v2f: float
) -> CalculosColision:
    """Ecuaciones de la guía, con masas en kg y velocidades con signo en m/s.

    ``p = m1·v1 + m2·v2``, ``K = ½m1·v1² + ½m2·v2²``,
    ``C_R = −(v2f − v1f)/(v2i − v1i)``, ``Δp = p_f − p_i``.
    """
    nan = float("nan")
    p_i = m1 * v1i + m2 * v2i
    p_f = m1 * v1f + m2 * v2f
    dp = p_f - p_i
    dp_rel = dp / p_i if np.isfinite(p_i) and p_i != 0 else nan
    K_i = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
    K_f = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
    den = v2i - v1i
    C_R = -(v2f - v1f) / den if np.isfinite(den) and den != 0 else nan
    return CalculosColision(
        p_i=float(p_i), p_f=float(p_f), dp=float(dp), dp_rel=float(dp_rel),
        K_i=float(K_i), K_f=float(K_f), C_R=float(C_R),
        escala_p=float(abs(m1 * v1i) + abs(m2 * v2i)),
    )


# ==========================================================================
# Nombres de archivo del plan de ensayos
# ==========================================================================
def ensayo_desde_nombre(nombre: str) -> Optional[int]:
    """Número de ensayo leído de un nombre como ``ensayo07.mp4`` (o ``None``)."""
    m = re.search(r"ensayo[\s_-]*0*(\d{1,2})", str(nombre), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def bloque_de_ensayo(n: Optional[int]) -> Optional[str]:
    """Bloque del plan (A: 1–3, B: 4–6, C: 7–9, D: 10–12) o ``None``."""
    if n is None or not 1 <= int(n) <= 12:
        return None
    return "ABCD"[(int(n) - 1) // 3]
