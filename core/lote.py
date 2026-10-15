"""Tablas de resultados por vídeo de los Laboratorios 3.

* **FIS301 (modo por lotes).** Son 24 vídeos por grupo, nombrados como
  ``D0_10V.mp4`` … ``D2_24V.mp4``: bloque de distancia (D0, D1, D2) y voltaje
  (10–24 V en pasos de 2 V). El bloque y el voltaje se leen del nombre; la
  corriente ``I`` (mA) la ingresa el grupo, porque se lee en el panel de la
  fuente durante la sesión. Cada vídeo aporta una fila: bloque, V, I, v,
  s_v y el radio de la órbita.
* **FIS101.** Doce ensayos (``ensayo07.mp4``); cada uno aporta sus cuatro
  velocidades con incertidumbre y, con las masas, p_i, p_f, Δp, Δp/p_i,
  K_i, K_f y C_R.

Las tablas son ``pandas.DataFrame`` con nombres de columna fijos, que son los
que se exportan al CSV y los que lee la pestaña de análisis de grupo.

Este módulo es independiente de Streamlit.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.colisiones import CalculosColision, ResultadoColision
from core.orbita import ResultadoOrbita

# ==========================================================================
# FIS301
# ==========================================================================
COLUMNAS_FIS301 = [
    "archivo", "bloque", "V (V)", "I (mA)",
    "v (mm/s)", "s_v (mm/s)", "u_v (mm/s)",
    "r (mm)", "s_r (mm)", "d_mina (mm)",
    "N", "vueltas", "t_ini (s)", "t_fin (s)", "avisos",
]

#: Bloques de distancia del plan de la guía.
BLOQUES_FIS301 = {"D0": "d0 (sin espaciador)", "D1": "d0 + 3 mm", "D2": "d0 + 6 mm"}

_RE_FIS301 = re.compile(r"(D\d)\s*[_\-\s]\s*(\d+(?:[.,]\d+)?)\s*V", re.IGNORECASE)


def nombre_fis301(nombre: str) -> Tuple[Optional[str], Optional[float]]:
    """Bloque y voltaje leídos de un nombre como ``D0_10V.mp4``.

    Devuelve ``(None, None)`` si el nombre no sigue la convención de la guía.
    """
    m = _RE_FIS301.search(str(nombre))
    if not m:
        return None, None
    return m.group(1).upper(), float(m.group(2).replace(",", "."))


def tabla_fis301_vacia() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNAS_FIS301})


def fila_fis301(
    archivo: str,
    res: ResultadoOrbita,
    corriente_mA: Optional[float] = None,
    bloque: Optional[str] = None,
    voltaje: Optional[float] = None,
) -> dict:
    """Fila de la tabla resumen a partir del análisis de un vídeo (en mm y mm/s)."""
    b, V = nombre_fis301(archivo)
    bloque = bloque or b
    voltaje = voltaje if voltaje is not None else V
    mm = 1000.0
    return {
        "archivo": archivo,
        "bloque": bloque,
        "V (V)": voltaje,
        "I (mA)": corriente_mA,
        "v (mm/s)": res.rapidez_media * mm,
        "s_v (mm/s)": res.rapidez_desv * mm,
        "u_v (mm/s)": res.incertidumbre * mm,
        "r (mm)": res.radio_medio * mm,
        "s_r (mm)": res.radio_desv * mm,
        "d_mina (mm)": res.dist_mina_media * mm if np.isfinite(res.dist_mina_media) else np.nan,
        "N": res.n,
        "vueltas": round(res.vueltas, 2),
        "t_ini (s)": res.t_ini,
        "t_fin (s)": res.t_fin,
        "avisos": " | ".join(res.avisos),
    }


def avisos_radio_bloque(df: pd.DataFrame, tolerancia: float = 0.25) -> pd.Series:
    """Marca los vídeos cuyo radio se aleja de la mediana de su bloque.

    La guía pide que el trazador gire a una distancia de la mina parecida en
    todos los puntos de un bloque: ``ℓ`` depende de esa distancia, y un punto
    con otro radio cae fuera de la recta ``v = kI``.
    """
    out = pd.Series([""] * len(df), index=df.index, dtype="object")
    if df.empty or "r (mm)" not in df:
        return out
    r = pd.to_numeric(df["r (mm)"], errors="coerce")
    for _, grupo in df.groupby(df["bloque"].fillna("?")):
        rg = r.loc[grupo.index]
        med = float(np.nanmedian(rg)) if rg.notna().any() else np.nan
        if not np.isfinite(med) or med <= 0:
            continue
        for i in grupo.index:
            if np.isfinite(r[i]) and abs(r[i] - med) / med > tolerancia:
                out[i] = (
                    f"radio {r[i]:.1f} mm vs {med:.1f} mm de mediana del bloque".replace(".", ",")
                )
    return out


# ==========================================================================
# FIS101
# ==========================================================================
COLUMNAS_FIS101 = [
    "ensayo", "bloque", "archivo", "m1 (g)", "m2 (g)",
    "v1i (m/s)", "u_v1i (m/s)", "v2i (m/s)", "u_v2i (m/s)",
    "v1f (m/s)", "u_v1f (m/s)", "v2f (m/s)", "u_v2f (m/s)",
    "p_i (kg·m/s)", "p_f (kg·m/s)", "Δp (kg·m/s)", "Δp/p_i (%)",
    "K_i (J)", "K_f (J)", "C_R",
    "t_choque (s)", "avisos",
]


def tabla_fis101_vacia() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNAS_FIS101})


def fila_fis101(
    ensayo: Optional[int],
    bloque: Optional[str],
    archivo: str,
    res: ResultadoColision,
    m1_g: Optional[float] = None,
    m2_g: Optional[float] = None,
    calc: Optional[CalculosColision] = None,
) -> dict:
    """Fila de la tabla de ensayos (velocidades en m/s, masas en g)."""
    nan = np.nan

    def par(vt):
        if vt is None:
            return nan, nan
        return vt.media, vt.incertidumbre

    v1i, u1i = par(res.v1i)
    v2i, u2i = par(res.v2i)
    v1f, u1f = par(res.v1f)
    v2f, u2f = par(res.v2f)
    fila = {
        "ensayo": ensayo, "bloque": bloque, "archivo": archivo,
        "m1 (g)": m1_g, "m2 (g)": m2_g,
        "v1i (m/s)": v1i, "u_v1i (m/s)": u1i, "v2i (m/s)": v2i, "u_v2i (m/s)": u2i,
        "v1f (m/s)": v1f, "u_v1f (m/s)": u1f, "v2f (m/s)": v2f, "u_v2f (m/s)": u2f,
        "p_i (kg·m/s)": nan, "p_f (kg·m/s)": nan, "Δp (kg·m/s)": nan,
        "Δp/p_i (%)": nan, "K_i (J)": nan, "K_f (J)": nan, "C_R": nan,
        "t_choque (s)": res.t_choque,
        "avisos": " | ".join(res.avisos),
    }
    if calc is not None:
        fila.update({
            "p_i (kg·m/s)": calc.p_i, "p_f (kg·m/s)": calc.p_f,
            "Δp (kg·m/s)": calc.dp, "Δp/p_i (%)": calc.dp_rel * 100.0,
            "K_i (J)": calc.K_i, "K_f (J)": calc.K_f, "C_R": calc.C_R,
        })
    elif res.v2i is not None:
        # C_R no necesita las masas.
        den = v2i - v1i
        fila["C_R"] = -(v2f - v1f) / den if np.isfinite(den) and den != 0 else nan
    return fila


# ==========================================================================
# Común
# ==========================================================================
def reemplazar_fila(df: pd.DataFrame, fila: dict, clave: str = "archivo") -> pd.DataFrame:
    """Agrega la fila o reemplaza la que tenga la misma ``clave`` (re-análisis)."""
    columnas = list(df.columns) if len(df.columns) else list(fila)
    nueva = pd.DataFrame([{c: fila.get(c) for c in columnas}])
    if not df.empty and clave in df and fila.get(clave) is not None:
        df = df[df[clave].astype(str) != str(fila[clave])]
    if df.empty:
        return nueva.reset_index(drop=True)
    return pd.concat([df, nueva], ignore_index=True)


def ordenar_fis301(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena por bloque y voltaje, como la tabla de registro de la guía."""
    if df.empty:
        return df
    v = pd.to_numeric(df["V (V)"], errors="coerce")
    return (df.assign(_v=v).sort_values(["bloque", "_v"], na_position="last")
              .drop(columns="_v").reset_index(drop=True))


def ordenar_fis101(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    e = pd.to_numeric(df["ensayo"], errors="coerce")
    return (df.assign(_e=e).sort_values(["_e", "archivo"], na_position="last")
              .drop(columns="_e").reset_index(drop=True))


def normalizar(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """Deja la tabla con exactamente ``columnas`` (para CSV importados o editados)."""
    out = df.copy()
    for c in columnas:
        if c not in out:
            out[c] = np.nan
    return out[columnas].reset_index(drop=True)
