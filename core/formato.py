"""Formato de números para pantalla y exportación, según el manual de informes.

Convenciones de los laboratorios UAI:

  * **Coma decimal** en pantalla y en los archivos exportados.
  * CSV con **«;»** como separador, que es lo que Excel en español espera: un
    CSV separado por comas se abre con todo metido en la primera columna.
  * Desviación estándar muestral, con ``1/(n−1)``.
  * Incertidumbre = la desviación estándar **o** la resolución, la mayor,
    redondeada a **una cifra significativa**; el valor se redondea a la misma
    posición decimal. Sin propagación formal.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def coma(texto: str) -> str:
    """Cambia el punto decimal por coma en un texto numérico ya formateado."""
    return texto.replace(".", ",")


def num(x: Optional[float], decimales: int = 3) -> str:
    """Número con coma decimal y ``decimales`` cifras tras la coma; «—» si falta."""
    if x is None or not np.isfinite(x):
        return "—"
    return coma(f"{x:.{int(decimales)}f}").replace("-", "−")


def cifras(x: Optional[float], sig: int = 3) -> str:
    """Número con ``sig`` cifras significativas y coma decimal; «—» si falta."""
    if x is None or not np.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    dec = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    return num(x, dec)


def desviacion(valores: Iterable[float]) -> float:
    """Desviación estándar muestral (1/(n−1)); ``nan`` con menos de 2 valores."""
    v = np.asarray(list(valores), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    return float(np.std(v, ddof=1))


def incertidumbre(desv: float, resolucion: float = 0.0) -> float:
    """La mayor entre la desviación estándar y la resolución (sin redondear)."""
    candidatos = [c for c in (desv, resolucion) if c is not None and np.isfinite(c)]
    return float(max(candidatos)) if candidatos else float("nan")


def decimales_de(u: float) -> int:
    """Posición decimal de una incertidumbre redondeada a una cifra significativa.

    Devuelve el número de decimales (puede ser negativo: 30 → −1).
    """
    if u is None or not np.isfinite(u) or u <= 0:
        return 3
    exp = int(math.floor(math.log10(abs(u))))
    primera = round(u / 10**exp)
    if primera >= 10:          # 0,096 → 0,1: el redondeo sube una posición
        exp += 1
    return -exp


def redondear(valor: float, u: float) -> Tuple[float, float, int]:
    """Redondea ``u`` a 1 cifra significativa y ``valor`` a la misma posición.

    Devuelve ``(valor_redondeado, u_redondeada, decimales)``.
    """
    d = decimales_de(u)
    if not np.isfinite(valor):
        return float("nan"), float("nan"), d
    if u is None or not np.isfinite(u) or u <= 0:
        return float(valor), float("nan"), d
    return round(float(valor), d), round(float(u), d), d


def valor_inc(valor: float, u: float, unidad: str = "") -> str:
    """Texto «valor ± u unidad» con la regla de una cifra significativa."""
    if valor is None or not np.isfinite(valor):
        return "—"
    v, ur, d = redondear(valor, u)
    sufijo = f" {unidad}" if unidad else ""
    if not np.isfinite(ur):
        return cifras(valor, 3) + sufijo
    dd = max(d, 0)
    return f"{num(v, dd)} ± {num(ur, dd)}{sufijo}"


def csv_es(df: pd.DataFrame) -> bytes:
    """CSV para Excel en español: «;» como separador y coma decimal (UTF-8 con BOM)."""
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def leer_csv_es(datos) -> pd.DataFrame:
    """Lee un CSV exportado por la app (acepta «;» con coma o «,» con punto)."""
    import io

    if isinstance(datos, (bytes, bytearray)):
        texto = bytes(datos).decode("utf-8-sig")
    elif hasattr(datos, "read"):
        crudo = datos.read()
        texto = crudo.decode("utf-8-sig") if isinstance(crudo, bytes) else crudo
    else:
        texto = str(datos)
    primera = texto.splitlines()[0] if texto else ""
    if ";" in primera:
        return pd.read_csv(io.StringIO(texto), sep=";", decimal=",")
    return pd.read_csv(io.StringIO(texto))
