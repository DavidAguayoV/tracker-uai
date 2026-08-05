#!/usr/bin/env bash
# ============================================================
#  Tracker UAI - Lanzador para macOS y Linux
#  Uso:  bash iniciar_tracker.sh   (o doble clic en macOS si es .command)
#  La primera vez crea el entorno e instala todo automaticamente.
# ============================================================
cd "$(dirname "$0")" || exit 1

# --- Busca Python 3 ---
PY=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
    echo "[ERROR] No se encontro Python 3."
    echo "  macOS:  instala desde https://www.python.org/downloads/ o 'brew install python'"
    echo "  Linux:  'sudo apt install python3 python3-venv'  (o el gestor de tu distro)"
    exit 1
fi

# --- Entorno virtual FUERA del proyecto (evita problemas de sincronizacion) ---
VENV="$HOME/.local/share/TrackerUAI/.venv"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "Primera ejecucion: preparando el entorno. Puede tardar varios minutos..."
    "$PY" -m venv "$VENV" || { echo "[ERROR] No se pudo crear el entorno."; exit 1; }
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
    python -m pip install --upgrade pip
    pip install -r requirements.txt || { echo "[ERROR] Fallo la instalacion."; exit 1; }
else
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
fi

echo ""
echo "Iniciando Tracker UAI... se abrira en tu navegador."
echo "Para cerrar el programa, presiona Ctrl+C en esta terminal."
echo ""
python -m streamlit run app.py
