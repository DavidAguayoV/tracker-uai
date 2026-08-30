"""Hace importable el paquete del proyecto (core/, models/) durante los tests.

Sin este archivo, ``pytest`` solo encuentra ``core`` cuando el directorio de
trabajo ya está en ``sys.path`` (lo que ocurre al invocar ``python -m pytest``
con un intérprete normal). El Python portable que instala el lanzador de
Windows usa un archivo ``._pth`` que desactiva ese comportamiento, así que la
ruta se agrega aquí de forma explícita.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
