"""Análisis de grupo de los Laboratorios 3 (números, sin conclusiones).

* **FIS101**: promedio y desviación estándar muestral de ``Δp/p_i`` y ``C_R``
  **por bloque** (A, B, C, D), sin mezclar bloques, como pide la guía. Se
  acompaña siempre de ``Δp`` absoluto, porque en B y D ``p_i`` casi se anula
  y el cociente pierde sentido.
* **FIS301**: ajuste de ``v = k·I`` **por el origen** para cada distancia
  (D0, D1, D2), con ``k``, su error estándar y ``R²``.

El ``R²`` de una recta forzada por el origen se calcula como lo hace Excel al
fijar la intersección en cero (``ESTIMACION.LINEAL`` con ``constante = FALSO``
o la línea de tendencia con «intersección = 0»): ``1 − SS_res / Σ v²``. No es
comparable con el ``R²`` de una recta con intercepto.

Este módulo no redacta conclusiones ni compara con valores esperados: eso lo
hace el grupo en el informe.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from core.formato import desviacion


# ==========================================================================
# FIS101
# ==========================================================================
def resumen_bloques_fis101(df: pd.DataFrame) -> pd.DataFrame:
    """Promedio y desviación estándar por bloque de Δp/p_i, Δp y C_R.

    Espera las columnas de :data:`core.lote.COLUMNAS_FIS101`. Devuelve una fila
    por bloque presente, en el orden A, B, C, D.
    """
    filas = []
    if df is None or df.empty or "bloque" not in df:
        return pd.DataFrame(columns=[
            "bloque", "n", "Δp/p_i medio (%)", "s Δp/p_i (%)",
            "Δp medio (kg·m/s)", "s Δp (kg·m/s)", "C_R medio", "s C_R",
        ])
    for b in ["A", "B", "C", "D"]:
        g = df[df["bloque"].astype(str).str.upper() == b]
        if g.empty:
            continue
        rel = pd.to_numeric(g["Δp/p_i (%)"], errors="coerce").dropna()
        dp = pd.to_numeric(g["Δp (kg·m/s)"], errors="coerce").dropna()
        cr = pd.to_numeric(g["C_R"], errors="coerce").dropna()
        filas.append({
            "bloque": b,
            "n": int(len(g)),
            "Δp/p_i medio (%)": float(rel.mean()) if len(rel) else np.nan,
            "s Δp/p_i (%)": desviacion(rel),
            "Δp medio (kg·m/s)": float(dp.mean()) if len(dp) else np.nan,
            "s Δp (kg·m/s)": desviacion(dp),
            "C_R medio": float(cr.mean()) if len(cr) else np.nan,
            "s C_R": desviacion(cr),
        })
    return pd.DataFrame(filas)


# ==========================================================================
# FIS301
# ==========================================================================
@dataclass
class AjusteOrigen:
    """Recta por el origen ``y = k·x``."""

    k: float
    k_err: float      # error estándar de la pendiente
    r2: float         # R² con la intersección fijada en cero (como Excel)
    n: int

    def predict(self, x):
        return self.k * np.asarray(x, dtype=float)


def ajuste_por_origen(x, y) -> AjusteOrigen:
    """Mínimos cuadrados de ``y = k·x``: ``k = Σxy/Σx²``.

    Error estándar: ``√(SS_res/(n−1) / Σx²)`` (un solo parámetro libre).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    sxx = float(np.sum(x * x))
    if n < 1 or sxx == 0:
        return AjusteOrigen(np.nan, np.nan, np.nan, n)
    k = float(np.sum(x * y) / sxx)
    res = y - k * x
    ss_res = float(np.sum(res**2))
    k_err = float(np.sqrt(ss_res / (n - 1) / sxx)) if n >= 2 else np.nan
    ss_tot = float(np.sum(y * y))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return AjusteOrigen(k, k_err, float(r2), n)


def ajustes_fis301(df: pd.DataFrame) -> Dict[str, AjusteOrigen]:
    """Ajuste ``v = k·I`` por el origen para cada bloque de distancia.

    ``k`` queda en (mm/s)/mA, que es numéricamente igual a (m/s)/A.
    """
    out: Dict[str, AjusteOrigen] = {}
    if df is None or df.empty:
        return out
    for b in sorted(df["bloque"].dropna().astype(str).unique()):
        g = df[df["bloque"].astype(str) == b]
        I = pd.to_numeric(g["I (mA)"], errors="coerce")
        v = pd.to_numeric(g["v (mm/s)"], errors="coerce")
        out[b] = ajuste_por_origen(I, v)
    return out
