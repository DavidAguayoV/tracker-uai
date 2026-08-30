"""Ajuste de modelos con incertidumbres y utilidades para los presets.

Los ajustes usan ``scipy.optimize.curve_fit``; las incertidumbres de
cada parámetro salen de la raíz de la diagonal de la matriz de
covarianza. Además incluye utilidades que necesitan los laboratorios:
lectura de meseta (velocidad terminal), ajuste lineal por tramos
(colisiones), período por cruces por cero y geometría de elipse
(péndulo cónico).

Para los laboratorios 2 hay utilidades de control por video, que
verifican en el acto el orden de magnitud del resultado:
``drag_from_terminal_velocity`` (FIS101: C_d de una sola caída) y
``bifilar_inertia`` (FIS201: I_zz de un solo largo de hilo). El
resultado que se informa en ambos casos es el del ajuste lineal del
grupo sobre todas las configuraciones, que se hace fuera de la app.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ==========================================================================
# Constantes de las guías de laboratorio (Santiago)
# ==========================================================================
#: Aceleración de gravedad local adoptada en las guías 2026 (m/s²).
G_SANTIAGO = 9.794

#: Densidad del aire en Santiago, 570 m de altitud y 20 °C (kg/m³).
RHO_AIRE_SANTIAGO = 1.12

#: Viscosidad dinámica del aire (Pa·s).
MU_AIRE = 1.81e-5

#: Rango de C_d considerado consistente con la referencia en FIS101 Lab 2.
CD_REFERENCIA = 1.3
CD_RANGO_ACEPTABLE = (1.1, 1.4)


# ==========================================================================
# Modelos base (firma compatible con curve_fit: f(t, *params))
# ==========================================================================
def model_linear(t, m, b):
    """Recta: x = m·t + b."""
    return m * t + b


def model_quadratic(t, a2, a1, a0):
    """Parábola: x = a2·t² + a1·t + a0."""
    return a2 * t**2 + a1 * t + a0


def model_sine(t, A, f, phi, C):
    """Sinusoidal: x = A·sin(2πf·t + φ) + C."""
    return A * np.sin(2 * np.pi * f * t + phi) + C


def model_damped_sine(t, A, gamma, f, phi, C):
    """Sinusoidal amortiguado: x = A·e^(−γt)·sin(2πf·t + φ) + C."""
    return A * np.exp(-gamma * t) * np.sin(2 * np.pi * f * t + phi) + C


def model_damped_exp(t, A, tau, C):
    """Exponencial amortiguado: x = A·e^(−t/τ) + C."""
    return A * np.exp(-t / tau) + C


def model_terminal_velocity(t, v_t, g):
    """Caída con arrastre cuadrático: v(t) = v_t·tanh(g·t / v_t)."""
    return v_t * np.tanh(g * t / v_t)


# ==========================================================================
# Estimaciones iniciales
# ==========================================================================
def _guess_frequency(t: np.ndarray, y: np.ndarray) -> float:
    """Estima la frecuencia dominante (Hz) por FFT, remuestreando a paso uniforme."""
    n = len(t)
    if n < 4:
        return 1.0
    t_uni = np.linspace(t[0], t[-1], n)
    y_uni = np.interp(t_uni, t, y)
    y_uni = y_uni - np.mean(y_uni)
    dt = (t[-1] - t[0]) / (n - 1)
    if dt <= 0:
        return 1.0
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(y_uni))
    if len(amp) > 1:
        k = 1 + int(np.argmax(amp[1:]))  # ignora componente DC
        if freqs[k] > 0:
            return float(freqs[k])
    span = t[-1] - t[0]
    return 1.0 / span if span > 0 else 1.0


def _guess(model: str, t: np.ndarray, y: np.ndarray) -> List[float]:
    """Devuelve un p0 razonable para cada modelo."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if model == "linear":
        m, b = np.polyfit(t, y, 1)
        return [m, b]
    if model == "quadratic":
        a2, a1, a0 = np.polyfit(t, y, 2)
        return [a2, a1, a0]
    if model in ("sine", "damped_sine"):
        A = np.std(y) * np.sqrt(2) or 1.0
        C = float(np.mean(y))
        f = _guess_frequency(t, y)
        if model == "sine":
            return [A, f, 0.0, C]
        span = t[-1] - t[0]
        gamma = 0.5 / span if span > 0 else 0.1
        return [A, gamma, f, 0.0, C]
    if model == "damped_exp":
        C = float(y[-1])
        A = float(y[0] - y[-1]) or 1.0
        span = t[-1] - t[0]
        tau = span / 3 if span > 0 else 1.0
        return [A, tau, C]
    if model == "terminal_velocity":
        v_t = float(np.max(np.abs(y))) or 1.0
        return [v_t, 9.8]
    raise ValueError(f"Modelo desconocido: {model}")


# Registro: nombre -> (función, nombres de parámetros)
MODELS: Dict[str, Tuple[Callable, List[str]]] = {
    "linear": (model_linear, ["m", "b"]),
    "quadratic": (model_quadratic, ["a2", "a1", "a0"]),
    "sine": (model_sine, ["A", "f", "phi", "C"]),
    "damped_sine": (model_damped_sine, ["A", "gamma", "f", "phi", "C"]),
    "damped_exp": (model_damped_exp, ["A", "tau", "C"]),
    "terminal_velocity": (model_terminal_velocity, ["v_t", "g"]),
}


# ==========================================================================
# Resultado del ajuste
# ==========================================================================
@dataclass
class FitResult:
    """Resultado de un ajuste: parámetros, incertidumbres y bondad."""

    model: str
    param_names: List[str]
    popt: np.ndarray
    perr: np.ndarray
    r_squared: float
    func: Callable

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Evalúa el modelo ajustado en ``t``."""
        return self.func(np.asarray(t, dtype=float), *self.popt)

    def params(self) -> Dict[str, Tuple[float, float]]:
        """Diccionario {nombre: (valor, sigma)}."""
        return {
            name: (float(v), float(s))
            for name, v, s in zip(self.param_names, self.popt, self.perr)
        }


def _r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de determinación R²."""
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_model(
    model: str,
    t: np.ndarray,
    y: np.ndarray,
    p0: Optional[List[float]] = None,
) -> FitResult:
    """Ajusta ``model`` a los datos (t, y) y devuelve un ``FitResult``.

    Las incertidumbres son la raíz de la diagonal de la matriz de
    covarianza. Lanza ``ValueError`` si hay muy pocos puntos.
    """
    if model not in MODELS:
        raise ValueError(f"Modelo desconocido: {model}")
    func, names = MODELS[model]
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < len(names):
        raise ValueError(
            f"Se necesitan al menos {len(names)} puntos para ajustar '{model}'."
        )
    if p0 is None:
        p0 = _guess(model, t, y)
    popt, pcov = curve_fit(func, t, y, p0=p0, maxfev=20000)
    perr = np.sqrt(np.abs(np.diag(pcov)))
    r2 = _r_squared(y, func(t, *popt))
    return FitResult(model, names, popt, perr, r2, func)


# ==========================================================================
# Utilidades para presets específicos
# ==========================================================================
def acceleration_from_quadratic(fit: FitResult) -> Tuple[float, float]:
    """Aceleración a = 2·a2 con su incertidumbre, desde un ajuste parabólico."""
    if fit.model != "quadratic":
        raise ValueError("Se esperaba un ajuste parabólico.")
    a2, s_a2 = fit.params()["a2"]
    return 2.0 * a2, 2.0 * s_a2


@dataclass
class PlateauResult:
    """Lectura de la meseta de una señal que satura (velocidad terminal)."""

    value: float          # promedio del tramo
    sigma: float          # desviación estándar muestral del tramo
    n: int                # puntos promediados
    t_start: float
    t_end: float
    slope: float          # pendiente del tramo (unidades de la señal / s)
    slope_err: float

    @property
    def relative_drift(self) -> float:
        """Cambio relativo de la señal a lo largo del tramo, según la recta ajustada."""
        if not np.isfinite(self.slope) or self.value == 0:
            return float("nan")
        return float(abs(self.slope) * (self.t_end - self.t_start) / abs(self.value))

    @property
    def is_flat(self) -> bool:
        """``True`` si el tramo es meseta: la señal ya no crece de forma sistemática.

        Se aplican dos criterios, y basta con que se cumpla uno:

        * **Físico**: la señal cambia menos de un 1 % a lo largo del tramo.
          Manda sobre el estadístico porque una deriva despreciable no deja de
          serlo por estar bien medida.
        * **Estadístico**: la pendiente es compatible con cero a 2σ.

        Sin el criterio físico, datos muy limpios darían siempre "no es meseta":
        con ruido casi nulo, ``slope_err`` se achica más rápido que la deriva
        residual y cualquier curvatura ínfima resulta significativa.
        """
        drift = self.relative_drift
        if np.isfinite(drift) and drift < 0.01:
            return True
        if not np.isfinite(self.slope) or not np.isfinite(self.slope_err):
            return True
        return abs(self.slope) <= 2.0 * self.slope_err


def read_plateau(
    t: np.ndarray,
    v: np.ndarray,
    last_n: Optional[int] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> PlateauResult:
    """Lee la meseta de ``v(t)`` promediando un tramo final.

    El tramo se elige de dos formas excluyentes: con ``last_n`` (los últimos
    N puntos, como pide la guía del Lab 2 de FIS101: 8 a 15 cuadros) o con un
    intervalo de tiempo ``[t_min, t_max]``, útil cuando el video sigue después
    de que el cuerpo toca el suelo.

    Además del promedio y su desviación estándar, ajusta una recta al tramo:
    su pendiente permite comprobar que efectivamente es meseta y no un punto
    intermedio de la curva.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    if len(t) == 0:
        nan = float("nan")
        return PlateauResult(nan, nan, 0, nan, nan, nan, nan)

    if t_min is not None or t_max is not None:
        lo = -np.inf if t_min is None else float(t_min)
        hi = np.inf if t_max is None else float(t_max)
        sel = (t >= lo) & (t <= hi)
        if not sel.any():
            sel = np.zeros(len(t), dtype=bool)
            sel[-1] = True
    else:
        n = len(t) if last_n is None else int(max(1, min(last_n, len(t))))
        sel = np.zeros(len(t), dtype=bool)
        sel[-n:] = True

    ts, vs = t[sel], v[sel]
    media = float(np.mean(vs))
    sigma = float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0

    # Pendiente del tramo por mínimos cuadrados, con su error estándar.
    # Se calcula a mano en vez de con curve_fit porque un tramo perfectamente
    # plano tiene residuos nulos y la covarianza queda indefinida (OptimizeWarning).
    slope = slope_err = float("nan")
    if len(vs) >= 3 and np.ptp(ts) > 0:
        dt = ts - np.mean(ts)
        s_xx = float(np.sum(dt**2))
        slope = float(np.sum(dt * (vs - np.mean(vs))) / s_xx)
        residuos = vs - (np.mean(vs) + slope * dt)
        var_res = float(np.sum(residuos**2) / (len(vs) - 2))
        slope_err = float(np.sqrt(var_res / s_xx))

    return PlateauResult(
        value=media,
        sigma=sigma,
        n=int(len(vs)),
        t_start=float(ts[0]),
        t_end=float(ts[-1]),
        slope=float(slope),
        slope_err=float(slope_err),
    )


def plateau_value(
    t: np.ndarray, v: np.ndarray, last_n: int = 5
) -> Tuple[float, float]:
    """Promedia los últimos ``last_n`` puntos de una señal (p. ej. velocidad terminal).

    Devuelve (media, desviación estándar). Envoltorio simple sobre
    :func:`read_plateau`, que entrega además el diagnóstico del tramo.
    """
    res = read_plateau(t, v, last_n=last_n)
    return res.value, res.sigma


# --------------------------------------------------------------------------
# FIS101 Lab 2 · arrastre aerodinámico y velocidad terminal
# --------------------------------------------------------------------------
def frontal_area(diameter: float) -> float:
    """Área frontal de un cuerpo circular: ``A = π D² / 4`` (D en metros)."""
    d = float(diameter)
    if d <= 0:
        raise ValueError("El diámetro debe ser positivo.")
    return float(np.pi * d**2 / 4.0)


def reynolds_number(
    v: float,
    length: float,
    rho: float = RHO_AIRE_SANTIAGO,
    mu: float = MU_AIRE,
) -> float:
    """Número de Reynolds ``Re = ρ v D / μ`` (adimensional).

    Sirve para justificar el régimen: el arrastre lineal exige ``Re ≲ 1`` y el
    cuadrático domina desde ``Re ~ 10³``.
    """
    if mu <= 0:
        raise ValueError("La viscosidad debe ser positiva.")
    return float(rho * abs(v) * abs(length) / mu)


@dataclass
class DragResult:
    """Coeficiente de arrastre estimado a partir de una sola caída."""

    c_d: float
    c_d_err: float
    area: float
    reynolds: float

    @property
    def in_reference_range(self) -> bool:
        """``True`` si ``C_d`` cae en el rango de referencia de la guía."""
        lo, hi = CD_RANGO_ACEPTABLE
        return bool(np.isfinite(self.c_d) and lo <= self.c_d <= hi)

    @property
    def percent_difference(self) -> float:
        """Diferencia porcentual respecto del valor tabulado de referencia."""
        return float(abs(self.c_d - CD_REFERENCIA) / CD_REFERENCIA * 100.0)


def drag_from_terminal_velocity(
    v_t: float,
    mass: float,
    diameter: float,
    v_t_err: float = 0.0,
    rho: float = RHO_AIRE_SANTIAGO,
    g: float = G_SANTIAGO,
    mu: float = MU_AIRE,
) -> DragResult:
    """Coeficiente de arrastre de **una sola caída**, como control de magnitud.

    Invierte ``v_t = sqrt(2 m g / (ρ C_d A))`` con ``A = π D² / 4``:

    .. math:: C_d = \\frac{2\\,m\\,g}{\\rho\\,A\\,v_t^{2}}

    Como ``C_d ∝ v_t^{-2}``, la incertidumbre relativa se duplica:
    ``ΔC_d/C_d = 2·Δv_t/v_t``.

    Este **no** es el resultado que se informa: la guía pide el ajuste lineal
    de ``v_t²`` contra ``m`` sobre las cinco configuraciones del bloque A. Sirve
    para detectar en el momento un problema de calibración o de lectura de la
    meseta, comparando con el rango 1,1–1,4.
    """
    v_t = float(v_t)
    mass = float(mass)
    if v_t <= 0:
        raise ValueError("La velocidad terminal debe ser positiva.")
    if mass <= 0:
        raise ValueError("La masa debe ser positiva.")
    area = frontal_area(diameter)
    c_d = 2.0 * mass * g / (rho * area * v_t**2)
    c_d_err = 2.0 * c_d * abs(v_t_err) / v_t
    return DragResult(
        c_d=float(c_d),
        c_d_err=float(c_d_err),
        area=area,
        reynolds=reynolds_number(v_t, diameter, rho=rho, mu=mu),
    )


# --------------------------------------------------------------------------
# FIS201 Lab 2 · péndulo bifilar y momento de inercia
# --------------------------------------------------------------------------
def bar_gyration_radius(length: float) -> float:
    """Radio de giro teórico de una barra uniforme: ``k = L / √12``."""
    return float(abs(length) / np.sqrt(12.0))


@dataclass
class BifilarResult:
    """Momento de inercia estimado a partir de un solo largo de hilo."""

    inertia: float          # I_zz (kg·m²)
    inertia_err: float
    gyration_radius: float  # k = sqrt(I/M) (m)


def bifilar_inertia(
    period: float,
    mass: float,
    half_separation: float,
    wire_length: float,
    period_err: float = 0.0,
    g: float = G_SANTIAGO,
) -> BifilarResult:
    """Momento de inercia de **un solo largo de hilo**, como control de magnitud.

    Invierte el período del péndulo bifilar
    ``T = 2π sqrt(I H / (M g d²))``:

    .. math:: I_{zz} = \\frac{T^{2} M g d^{2}}{4\\pi^{2} H}

    ``half_separation`` es ``d``, la **semiseparación entre los hilos**, no el
    largo del cuerpo: en el montaje de la guía los hilos se amarran a 5 cm
    sobre una barra de 15 cm, y confundir ambos da un momento de inercia nueve
    veces mayor. ``wire_length`` es ``H``, el largo de los hilos.

    Como ``I ∝ T²``, la incertidumbre relativa se duplica:
    ``ΔI/I = 2·ΔT/T``.

    El resultado que se informa es el del ajuste de ``T²`` contra ``H`` sobre
    los seis largos; esto sirve para verificar el orden de magnitud en el acto.
    """
    period = float(period)
    mass = float(mass)
    d = float(half_separation)
    H = float(wire_length)
    if period <= 0:
        raise ValueError("El período debe ser positivo.")
    if mass <= 0:
        raise ValueError("La masa debe ser positiva.")
    if d <= 0:
        raise ValueError("La semiseparación de los hilos debe ser positiva.")
    if H <= 0:
        raise ValueError("El largo de los hilos debe ser positivo.")

    inertia = period**2 * mass * g * d**2 / (4.0 * np.pi**2 * H)
    inertia_err = 2.0 * inertia * abs(period_err) / period
    return BifilarResult(
        inertia=float(inertia),
        inertia_err=float(inertia_err),
        gyration_radius=float(np.sqrt(inertia / mass)),
    )


def period_timing_uncertainty(
    fps: float, n_cycles: int, frame_error: float = 2.0
) -> float:
    """Incertidumbre de cronometraje del período, en segundos.

    El error de identificar cada cruce es de un par de cuadros y es el mismo se
    midan uno o diez períodos; al repartirlo entre ``n_cycles`` ciclos se
    reduce en esa proporción. Por eso la guía exige contar diez oscilaciones y
    no una.
    """
    if fps <= 0 or n_cycles <= 0:
        return float("nan")
    return float(frame_error / fps / n_cycles)


def piecewise_linear_fit(
    t: np.ndarray, x: np.ndarray, t_split: float
) -> Tuple[FitResult, FitResult]:
    """Ajusta una recta antes y otra después de ``t_split`` (colisiones).

    Devuelve (ajuste_antes, ajuste_despues). La pendiente ``m`` de cada
    uno es la velocidad del tramo. Lanza ``ValueError`` si algún tramo
    tiene menos de 2 puntos.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    before = t <= t_split
    after = t > t_split
    if before.sum() < 2 or after.sum() < 2:
        raise ValueError(
            "Cada tramo (antes/después del choque) necesita al menos 2 puntos. "
            "Ajusta el instante de separación."
        )
    fit_before = fit_model("linear", t[before], x[before])
    fit_after = fit_model("linear", t[after], x[after])
    return fit_before, fit_after


def period_by_zero_crossings(
    t: np.ndarray, y: np.ndarray
) -> Tuple[float, int]:
    """Período por cruces por cero ascendentes de la señal centrada.

    Cuenta cruces en el mismo sentido (de − a +) y estima T como el
    tiempo total entre el primer y el último cruce, dividido por el
    número de ciclos. Devuelve (período, n_ciclos). ``nan`` si no hay
    al menos dos cruces.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float) - np.mean(y)
    crossings = []
    for i in range(len(y) - 1):
        if y[i] <= 0 < y[i + 1]:  # cruce ascendente
            # Interpolación lineal para el instante del cruce.
            frac = -y[i] / (y[i + 1] - y[i])
            crossings.append(t[i] + frac * (t[i + 1] - t[i]))
    if len(crossings) < 2:
        return float("nan"), 0
    n_cycles = len(crossings) - 1
    period = (crossings[-1] - crossings[0]) / n_cycles
    return float(period), n_cycles


@dataclass
class EllipseGeometry:
    """Geometría de la órbita del péndulo cónico."""

    period: float
    period_err: float
    a: float          # semieje asociado a x
    b: float          # semieje asociado a y
    center: Tuple[float, float]
    semi_major: float
    semi_minor: float
    focal_distance: float          # distancia centro-foco (c)
    foci: Tuple[Tuple[float, float], Tuple[float, float]]


def fit_conical_pendulum(
    t: np.ndarray, x: np.ndarray, y: np.ndarray
) -> Tuple[FitResult, FitResult, EllipseGeometry]:
    """Ajusta x(t) e y(t) como sinusoides y deriva la geometría de la elipse.

    Modelo: x = A·sin(2πf·t + φx) + Cx ; y = B·sin(2πf·t + φy) + Cy.
    Devuelve (ajuste_x, ajuste_y, geometría). El período se promedia de
    ambos ajustes.
    """
    fx = fit_model("sine", t, x)
    fy = fit_model("sine", t, y)
    px, py = fx.params(), fy.params()

    A = abs(px["A"][0])
    B = abs(py["A"][0])
    Cx, Cy = px["C"][0], py["C"][0]

    fxv, fxs = px["f"]
    fyv, fys = py["f"]
    f_mean = 0.5 * (abs(fxv) + abs(fyv))
    period = 1.0 / f_mean if f_mean > 0 else float("nan")
    # Propagación de T = 1/f (promedio de ambos ejes).
    f_err = 0.5 * np.hypot(fxs, fys)
    period_err = period**2 * f_err if np.isfinite(period) else float("nan")

    semi_major = max(A, B)
    semi_minor = min(A, B)
    c = float(np.sqrt(max(semi_major**2 - semi_minor**2, 0.0)))
    # Focos a lo largo del eje mayor.
    if A >= B:
        foci = ((Cx - c, Cy), (Cx + c, Cy))
    else:
        foci = ((Cx, Cy - c), (Cx, Cy + c))

    geo = EllipseGeometry(
        period=period,
        period_err=period_err,
        a=A,
        b=B,
        center=(Cx, Cy),
        semi_major=semi_major,
        semi_minor=semi_minor,
        focal_distance=c,
        foci=foci,
    )
    return fx, fy, geo


@dataclass
class EllipseFit:
    """Elipse ajustada a una nube de puntos (permite rotación)."""

    cx: float
    cy: float
    a: float           # semieje mayor
    b: float           # semieje menor
    theta: float       # ángulo de rotación del eje mayor (rad)
    foci: Tuple[Tuple[float, float], Tuple[float, float]]
    eccentricity: float


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> EllipseFit:
    """Ajusta una elipse general (Halir–Flusser) a los puntos (x, y).

    A diferencia de ajustar x(t) e y(t) por separado, este método reconstruye
    correctamente elipses **inclinadas** (ejes no alineados con x/y). Devuelve
    centro, semiejes, ángulo de rotación, focos y excentricidad.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 5:
        raise ValueError("Se necesitan al menos 5 puntos para ajustar una elipse.")

    # Centrar mejora la estabilidad numérica.
    mx, my = float(np.mean(x)), float(np.mean(y))
    xc, yc = x - mx, y - my

    D1 = np.column_stack([xc**2, xc * yc, yc**2])
    D2 = np.column_stack([xc, yc, np.ones_like(xc)])
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    C1 = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
    T = -np.linalg.solve(S3, S2.T)
    M = np.linalg.solve(C1, S1 + S2 @ T)
    _, evec = np.linalg.eig(M)
    cond = 4 * evec[0, :] * evec[2, :] - evec[1, :] ** 2
    valid = np.where(cond > 0)[0]
    if len(valid) == 0:
        raise ValueError("No se pudo ajustar una elipse a los datos.")
    a1 = np.real(evec[:, valid[0]])
    coef = np.concatenate([a1, T @ a1])          # [A, B, C, D, E, F] centrados
    A, B, Cc, Dd, Ee, Ff = coef

    M2 = np.array([[A, B / 2], [B / 2, Cc]])
    x0, y0 = np.linalg.solve(M2, [-Dd / 2, -Ee / 2])
    Fp = A * x0**2 + B * x0 * y0 + Cc * y0**2 + Dd * x0 + Ee * y0 + Ff
    evals, evecs = np.linalg.eigh(M2)
    lengths = np.sqrt(np.maximum(-Fp / evals, 0.0))
    order = np.argsort(lengths)[::-1]            # mayor primero
    a_len = float(lengths[order[0]])
    b_len = float(lengths[order[1]])
    major = evecs[:, order[0]]
    theta = float(np.arctan2(major[1], major[0]))

    cx, cy = x0 + mx, y0 + my
    c = float(np.sqrt(max(a_len**2 - b_len**2, 0.0)))
    ux, uy = np.cos(theta), np.sin(theta)
    foci = ((cx - c * ux, cy - c * uy), (cx + c * ux, cy + c * uy))
    ecc = c / a_len if a_len > 0 else 0.0
    return EllipseFit(float(cx), float(cy), a_len, b_len, theta, foci, ecc)


def swept_area_segments(
    x: np.ndarray, y: np.ndarray, center: Tuple[float, float], n_segments: int
) -> np.ndarray:
    """Áreas barridas respecto de ``center`` en ``n_segments`` tramos de igual nº de pasos.

    Usa la fórmula del área de sector (½·|r_i × r_{i+1}|) acumulada por
    tramo. Sirve para verificar la 2ª ley de Kepler (áreas iguales en
    tiempos iguales, si los frames están equiespaciados en el tiempo).
    Devuelve un arreglo con el área de cada tramo.
    """
    x = np.asarray(x, dtype=float) - center[0]
    y = np.asarray(y, dtype=float) - center[1]
    # Área de cada paso: ½ (x_i·y_{i+1} − x_{i+1}·y_i).
    dA = 0.5 * np.abs(x[:-1] * y[1:] - x[1:] * y[:-1])
    splits = np.array_split(dA, n_segments)
    return np.array([float(np.sum(s)) for s in splits])
