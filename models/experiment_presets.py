"""Presets de laboratorio para el ajuste de modelos.

Cada preset describe un flujo de análisis (``workflow``) y valores por
defecto (señal a ajustar, modelo, etiqueta del parámetro físico). La
interfaz (app.py) lee estos presets para armar los controles y ejecutar
la utilidad correspondiente de ``core/fitting.py``.

Los flujos disponibles (``workflow``):
  * "fit"               — ajuste genérico de un modelo a una señal.
  * "gravity_incline"   — parábola en x(t); reporta a = 2·a2.
  * "terminal_velocity" — lee la meseta de v(t); reporta v_t.
  * "collision"         — recta por tramos; reporta v_i y v_f.
  * "conical"           — péndulo cónico: elipse, período, focos, áreas.
  * "period_zero"       — período por cruces por cero de una señal.
  * "linear_speed"      — recta en x(t) o y(t); la pendiente es la rapidez.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Preset:
    """Descripción de un preset de análisis."""

    key: str
    curso: str
    nombre: str
    workflow: str
    descripcion: str
    signal: Optional[str] = None        # "x_m", "y_m", "speed", ...
    model: Optional[str] = None         # nombre en core.fitting.MODELS
    param_label: str = ""               # parámetro físico que entrega
    ayuda: str = ""                     # nota para el estudiante


PRESETS: List[Preset] = [
    Preset(
        key="generico",
        curso="—",
        nombre="Genérico / manual",
        workflow="fit",
        descripcion="Elige tú la señal y el modelo a ajustar.",
        signal="y_m",
        model="linear",
        param_label="parámetros del modelo",
        ayuda="Ajuste libre: sirve para MRU, tiro parabólico, MAS, etc.",
    ),
    Preset(
        key="fis101_lab1_gravedad",
        curso="FIS101",
        nombre="Lab 1 — Gravedad en plano inclinado (Galileo)",
        workflow="gravity_incline",
        descripcion="Parábola en x(t); la aceleración es a = 2·(coef. cuadrático).",
        signal="x_m",
        model="quadratic",
        param_label="a (m/s²)",
        ayuda="Trackea la posición del carro a lo largo del riel. Un video por "
        "lanzamiento. El paso de a vs sin(θ) para obtener g es análisis externo.",
    ),
    Preset(
        key="fis101_lab2_terminal",
        curso="FIS101",
        nombre="Lab 2 — Velocidad terminal (arrastre)",
        workflow="terminal_velocity",
        descripcion="Lee la meseta de v(t): promedio de los últimos N puntos.",
        signal="speed",
        model="terminal_velocity",
        param_label="v_t (m/s)",
        ayuda="Trackea la caída vertical del filtro de café. v(t) satura hacia "
        "una meseta; promedia el tramo donde ya no crece.",
    ),
    Preset(
        key="fis101_lab3_colisiones",
        curso="FIS101",
        nombre="Lab 3 — Colisiones (momentum)",
        workflow="collision",
        descripcion="Recta por tramos: v_i antes y v_f después del choque.",
        signal="x_m",
        model="linear",
        param_label="v_i, v_f (m/s)",
        ayuda="Trackea un carro por sesión. Marca el instante del choque para "
        "separar los tramos. Repite para el otro carro (multi-objeto es v2).",
    ),
    Preset(
        key="fis201_lab1_conico",
        curso="FIS201",
        nombre="Lab 1 — Péndulo cónico (Kepler)",
        workflow="conical",
        descripcion="Elipse: x=A·sin, y=B·sin; período, centro, focos y áreas.",
        signal=None,
        model="sine",
        param_label="T, semiejes, focos",
        ayuda="Trackea la plomada vista desde arriba (órbita elíptica). Marca "
        "los puntos a intervalos de tiempo iguales para verificar áreas iguales.",
    ),
    Preset(
        key="fis201_lab2_bifilar",
        curso="FIS201",
        nombre="Lab 2 — Péndulo bifilar (torsión)",
        workflow="period_zero",
        descripcion="Período por cruces por cero (≈10 oscilaciones).",
        signal="x_m",
        model="sine",
        param_label="T (s)",
        ayuda="Trackea un punto de la barra en vista cenital; su oscilación "
        "sirve para contar cruces. El período se obtiene contando ciclos.",
    ),
    Preset(
        key="fis301_lab3_lorentz",
        curso="FIS301",
        nombre="Lab 3 — Fuerza de Lorentz / MHD",
        workflow="linear_speed",
        descripcion="Recta en x(t) o y(t); la pendiente es la rapidez del flujo.",
        signal="x_m",
        model="linear",
        param_label="v (rapidez del flujo)",
        ayuda="Trackea el trazador de látex flotante, que se mueve a rapidez "
        "aproximadamente constante.",
    ),
]

PRESETS_BY_KEY = {p.key: p for p in PRESETS}


def preset_labels() -> List[str]:
    """Etiquetas legibles para el selector de la interfaz."""
    out = []
    for p in PRESETS:
        if p.curso == "—":
            out.append(p.nombre)
        else:
            out.append(f"{p.curso} · {p.nombre}")
    return out
