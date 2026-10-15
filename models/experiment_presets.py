"""Presets de laboratorio para el ajuste de modelos.

Cada preset describe un flujo de análisis (``workflow``) y valores por
defecto (señal a ajustar, modelo, etiqueta del parámetro físico). La
interfaz (app.py) lee estos presets para armar los controles y ejecutar
la utilidad correspondiente de ``core/fitting.py``.

Los flujos disponibles (``workflow``):
  * "fit"               — ajuste genérico de un modelo a una señal.
  * "gravity_incline"   — parábola en x(t); reporta a = 2·a2.
  * "terminal_velocity" — lee la meseta de v(t); reporta v_t.
  * "collision"         — recta por tramos; reporta v_i y v_f (un carro).
  * "colision_2carros"  — FIS101 Lab 3: dos carros, tramos y C_R.
  * "orbita"            — FIS301 Lab 3: rapidez y radio de la órbita.
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
        nombre="Lab 2 — Arrastre aerodinámico y velocidad terminal",
        workflow="terminal_velocity",
        descripcion="Lee v_t como el promedio de la meseta de v(t) (8 a 15 "
        "cuadros) y controla el orden de magnitud de C_d.",
        signal="speed",
        model="terminal_velocity",
        param_label="v_t (m/s)",
        ayuda="Trackea la caída del filtro de café desde la suelta. v(t) crece "
        "y satura: promedia el tramo final donde ya no crece de forma "
        "sistemática, nunca un cuadro aislado ni el máximo. Una caída = una "
        "repetición; hay 7 configuraciones (A1–A5 varían la masa a diámetro "
        "fijo, B1–B2 varían el diámetro) con 3 repeticiones cada una. El "
        "resultado que se informa, C_d, sale del ajuste de v_t² contra m sobre "
        "el bloque A, que se hace en Excel con los promedios de las tres "
        "repeticiones.",
    ),
    Preset(
        key="fis101_lab3_colisiones",
        curso="FIS101",
        nombre="Lab 3 — Colisiones (momentum)",
        workflow="colision_2carros",
        descripcion="Dos carros: instante del choque, tramos de velocidad "
        "constante antes y después, v1i, v2i, v1f y v2f con su incertidumbre, "
        "y p_i, p_f, Δp, Δp/p_i, K_i, K_f y C_R.",
        signal="x_m",
        model=None,
        param_label="v1i, v2i, v1f, v2f (m/s)",
        ayuda="Sigue los DOS carros en el Paso 3 (Carro 1 y Carro 2, cada uno "
        "por su marca de color). El eje x va paralelo al riel y es positivo en "
        "el sentido en que parte el carro 1: si en el video avanza hacia la "
        "izquierda, la app invierte el signo sola. Un video por ensayo; el "
        "bloque (A–D) se lee del nombre ensayo01…ensayo12. Guarda cada "
        "ensayo en la tabla y descárgala para el análisis de grupo.",
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
        nombre="Lab 2 — Péndulo bifilar (momento de inercia)",
        workflow="period_zero",
        descripcion="Período de torsión por cruces por el equilibrio en el "
        "mismo sentido (10 oscilaciones) y control de magnitud de I_zz.",
        signal="x_m",
        model="sine",
        param_label="T (s)",
        ayuda="Trackea un punto de la barra en vista cenital: su oscilación es "
        "la señal de torsión. Los cruces se cuentan siempre en el MISMO "
        "sentido de giro; contarlos todos da medio período y un momento de "
        "inercia cuatro veces menor. Filma al menos 12 oscilaciones para poder "
        "medir sobre 10. El resultado que se informa, I_zz, sale del ajuste de "
        "T² contra H sobre los seis largos de hilo, que se hace en Excel. Ojo: "
        "en las fórmulas entra d, la SEMISEPARACIÓN de los hilos (2d = 5 cm), "
        "no el largo de la barra (15 cm).",
    ),
    Preset(
        key="fis301_lab3_lorentz",
        curso="FIS301",
        nombre="Lab 3 — Fuerza de Lorentz / bombeo MHD",
        workflow="orbita",
        descripcion="Rapidez del trazador cuadro a cuadro a lo largo de su "
        "órbita alrededor de la mina: v media en el tramo de giro estable, su "
        "desviación estándar s_v y el radio de la órbita.",
        signal="speed",
        model=None,
        param_label="v (mm/s)",
        ayuda="Vista cenital. Sigue el trazador de látex por COLOR en el Paso 3 "
        "(es más robusto que la plantilla frente a las burbujas y la mina). "
        "Marca la mina para obtener la distancia del trazador a ella. Con la "
        "cámara fija, la calibración y la mina sirven para los 24 videos: usa "
        "la etapa «Lote FIS301» para procesarlos todos e ingresar I de cada uno.",
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
