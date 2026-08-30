# Tracker UAI

Aplicación de **análisis de video para laboratorios de física** universitarios,
inspirada en [Tracker (OpenSourcePhysics)](https://physlets.org/tracker/) pero
simplificada y dirigida a los cursos FIS101 / FIS201 / FIS301 de la Universidad
Adolfo Ibáñez.

Los estudiantes graban su experimento con el celular, suben el video y la app
los guía por etapas: **cargar → calibrar → trackear → cinemática → ajuste →
exportar**. Además incluye una herramienta de **gráficos de datos manuales** y
una de **medición geométrica sobre fotos**, ambas independientes del video.
Toda la lógica de cálculo vive en `core/` (sin Streamlit), así que es testeable
de forma independiente (`pytest`).

Hecho en **Python + Streamlit + OpenCV**; funciona en **Windows, macOS y Linux**.

## Características

- 🎥 Navegación por frames con saltos ±1/±5/±10 y selección de frame exacto.
- 📏 Calibración píxel→metro con **ejes de orientación libre** (plano inclinado).
- 🎯 **Tracking manual y automático** (plantilla, color HSV y CSRT si está
  disponible), con revisión y corrección frame a frame.
- 📈 Cinemática (posición, velocidad, aceleración, trayectoria) con suavizado
  Savitzky-Golay.
- 🧮 **Ajuste de modelos** con incertidumbres y R², y **presets por laboratorio**.
- 📊 **Gráficos** de datos manuales con ajuste lineal/cuadrático y lectura física
  de `g` (plantillas FIS101/FIS201).
- 📐 **Medición sobre foto**: distancias, ángulos (con eje horizontal
  redefinible) y áreas; exporta la imagen anotada.
- 💾 Exportación de datos (CSV) y figuras (PNG/SVG/ZIP).
- 🎨 Tema claro/oscuro y tipografía profesional.

---

## Descargar el proyecto

```bash
git clone https://github.com/DavidAguayoV/tracker-uai.git
cd tracker-uai
```

(O descarga el ZIP desde la página del repositorio con **Code → Download ZIP**.)

---

## Requisitos

- **Python 3.9 o superior** (desarrollado en 3.11; probado también en 3.14).
  En Windows **no hace falta tener Python**: el lanzador puede descargar una
  copia portable (ver más abajo).
- Los paquetes de `requirements.txt` (Streamlit, OpenCV, NumPy, SciPy, Plotly,
  pandas, Matplotlib).

## Instalación rápida en Windows (sin administrador)

Para uso docente o para un colega que no programa, **no hay que instalar nada
a mano**: doble clic en `Iniciar_Tracker_UAI.bat`. El lanzador, por orden:

1. Reutiliza el entorno de una ejecución anterior, si existe.
2. Busca un Python ≥ 3.9 ya instalado (`py -3`, `python`, o instalaciones por
   usuario en `%LOCALAPPDATA%\Programs\Python\`) y crea un entorno virtual.
3. Si no encuentra ninguno, **ofrece descargar Python portable** (~11 MB,
   distribución *embeddable* oficial de python.org) en
   `%LOCALAPPDATA%\TrackerUAI\python`.
4. Instala las dependencias y arranca la app.

Nada de esto requiere permisos de administrador ni modifica el `PATH`: todo
vive en `%LOCALAPPDATA%\TrackerUAI`, y desinstalar es borrar esa carpeta.

Opciones del lanzador:

```bat
Iniciar_Tracker_UAI.bat --portable
```

```bat
Iniciar_Tracker_UAI.bat --reparar
```

`--portable` fuerza la copia portable aunque haya Python en el equipo (útil si
la instalación del sistema está rota). `--reparar` borra el entorno y lo rehace
desde cero.

### El aviso "Windows protegió tu PC"

No es un problema del programa: Windows marca (*Mark-of-the-Web*) todo archivo
descargado de internet, y esa marca se propaga a cada archivo extraído del ZIP,
incluido el `.bat`. La solución definitiva es quitarla **del ZIP, antes de
descomprimir**: clic derecho sobre el ZIP → **Propiedades** → marcar
**Desbloquear** → Aceptar → recién ahí *Extraer todo*.

Si ya se descomprimió sin desbloquear, se puede limpiar la carpeta extraída:

```powershell
Get-ChildItem -Recurse 'C:\ruta\a\TrackerUAI' | Unblock-File
```

Y si aun así aparece la pantalla azul: **Más información → Ejecutar de todas
formas**.

## Instalación manual (desarrollo, macOS, Linux)

Desde una terminal, en la carpeta del proyecto:

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

> **Tracking automático de alta calidad (CSRT):** `requirements.txt` ya usa
> `opencv-contrib-python`, que incluye los trackers CSRT/KCF. **No instales
> además `opencv-python`**: entran en conflicto (elige uno solo). Si CSRT no
> aparece en la app, es que está instalado `opencv-python` a secas.

## Ejecución

```bash
python -m streamlit run app.py
```

Se abre en el navegador (normalmente http://localhost:8501).

> En Windows, si el comando `streamlit` "no se reconoce", usa siempre
> `python -m streamlit run app.py`. Es la forma más confiable.

---

## Compartir con estudiantes (carpeta + doble clic)

Pensado para Windows, sin que el estudiante sepa programar:

1. Ejecuta **`Crear_paquete_para_compartir.bat`** → genera
   `TrackerUAI_compartir.zip` (limpio, sin `.venv` ni videos).
2. Envía ese `.zip` al estudiante o colega.
3. Quien lo recibe sigue **`LEEME_ESTUDIANTES.txt`**: **desbloquea el ZIP**
   (clic derecho → Propiedades → Desbloquear), lo descomprime y hace doble clic
   en **`Iniciar_Tracker_UAI.bat`**. La primera vez instala todo solo; luego
   abre en segundos.

> **No hay requisitos previos en el equipo de destino.** Si no hay Python, el
> lanzador ofrece descargar su propia copia portable. En ningún momento se
> piden permisos de administrador.

---

## Flujo de uso (por etapas)

La barra lateral tiene el selector de **Etapa**, la carga del video y un
**indicador de progreso**. Cada etapa termina con un botón **Continuar →** que
lleva a la siguiente, de modo que un estudiante que abre el programa por primera
vez puede recorrer todo el análisis sin volver a la barra lateral. La navegación
libre entre etapas se mantiene: nada queda bloqueado.

1. **Video** — Sube el archivo (mp4/mov/avi/mkv/m4v/webm) y **recorta el tramo**
   que se va a analizar: todo lo posterior usa solo ese tramo. La etapa cierra
   mostrando qué quedó seleccionado (barra del tramo sobre el total y
   **miniaturas del frame inicial y final**).
2. **Calibración** — Marca dos puntos de **distancia conocida** (escala px→m) y
   el **origen**, guiado por una checklist numerada. La distancia se escribe en
   **cm, mm o m**, y la calibración **se aplica sola** al completarse. Se
   muestra a cuánto equivale un píxel y el ancho real del encuadre, con aviso si
   el valor es implausible. Opcional: **ejes con orientación libre** (marca la
   dirección +X) para montajes como un plano inclinado. El eje Y físico apunta
   hacia arriba.
3. **Tracking** — Arranca en **automático**:
   - **Plantilla / CSRT:** un **clic sobre el objeto** crea el recuadro (con
     deslizador de tamaño y recorte ampliado de lo seleccionado); también puedes
     arrastrar para dibujarlo a mano.
   - **Color:** clic para tomar el color, con vista previa de la máscara HSV.
   - **Manual:** clic sobre el objeto en cada frame (con auto-avance).
   - Al terminar se avisa de los **frames sin detección**. La vista con todas las
     detecciones superpuestas y la corrección frame a frame quedan en
     desplegables, para no hacer esperar a quien solo quiere seguir.
   - La **tabla** es editable: corrige `x_px/y_px` o elimina filas.
4. **Cinemática** — **Resumen numérico** del movimiento (duración, distancia
   recorrida, desplazamiento neto, rapidez media y máxima, aceleración media),
   **calidad de los datos** (ruido de marcado, frames faltantes) y los gráficos
   de posición, velocidad, aceleración y trayectoria. Suavizado opcional
   Savitzky-Golay (ventana y orden configurables).
5. **Ajuste** — Presets por laboratorio (ver abajo) + ajuste **genérico**
   (lineal, parabólico, sinusoidal, amortiguado, exponencial) con parámetros e
   **incertidumbres** (raíz de la diagonal de la covarianza) y R².
6. **Exportar** — Las figuras (posición, velocidad, aceleración, trayectoria,
   error de detección) **se ven en pantalla** y cada una tiene su botón de
   descarga. También puedes bajar el **CSV** de datos crudos y derivados, o un
   **ZIP** con todo junto.

**📊 Gráficos (datos manuales)** — Herramienta **independiente del video**
(no requiere cargar nada): el estudiante ingresa datos X e Y en una tabla,
crea varias series (con nombre, color y visibilidad), obtiene un gráfico de
dispersión y aplica **ajuste lineal o cuadrático** mostrando la **ecuación** y
el **R²**. Exporta la figura en **PNG o SVG** para el informe. Incluye
**plantillas de laboratorio** que leen `g` directamente del ajuste
(FIS101 `a` vs `sin θ`, FIS201 `T²` vs `L`).

**📐 Medición (foto)** — Herramienta **independiente del video** para medir
sobre una imagen fija: **distancia**, **ángulo con la horizontal** (con eje
horizontal redefinible para cambiar el sistema de coordenadas), **ángulo de
3 puntos** y **área** de un polígono. Los valores se dibujan sobre la imagen y
puedes **exportar la foto anotada** en PNG. Pensada para labs de foto (p. ej.
FIS301: distancia entre globos y ángulo del hilo) y para medir áreas/ángulos
del péndulo en FIS201.

### Presets de laboratorio disponibles

| Curso | Laboratorio | Qué entrega |
|-------|-------------|-------------|
| — | Genérico / manual | Ajuste libre de cualquier modelo |
| FIS101 | Lab 1 · Gravedad en plano inclinado | Aceleración `a` (parábola x(t) **o** recta v(t)) |
| FIS101 | Lab 2 · Arrastre y velocidad terminal | `v_t` (meseta de v(t)) + control de `C_d` |
| FIS101 | Lab 3 · Colisiones | `v_i`, `v_f` (recta por tramos) |
| FIS201 | Lab 1 · Péndulo cónico (Kepler) | `T`, semiejes, focos, áreas barridas |
| FIS201 | Lab 2 · Péndulo bifilar | `T` (cruces por cero) + control de `I_zz` |
| FIS301 | Lab 3 · Fuerza de Lorentz / MHD | Rapidez `v` del flujo |

#### Alcance de los presets de Lab 2

Los dos laboratorios 2 se resuelven con un **ajuste lineal sobre varios
videos**: `v_t²` contra `m` en FIS101 (cinco masas del bloque A) y `T²` contra
`H` en FIS201 (seis largos de hilo). Ese ajuste **se hace fuera de la app**, en
Excel, como piden las guías.

Lo que la app entrega por video es el dato que entra a ese ajuste (`v_t` o `T`)
y un **control de orden de magnitud** —`C_d` de esa caída, `I_zz` de ese
largo— para detectar en el momento un error de calibración, de unidades o de
lectura, antes de acumular veintiuna tomas mal medidas. Los paneles lo dicen
explícitamente para que nadie informe ese número como resultado.

---

## Consejos de grabación

- Cámara **perpendicular** al plano del movimiento y **fija** (trípode).
- La regla/objeto de calibración debe estar **en el mismo plano** que el
  movimiento (esta versión **no corrige perspectiva**).
- Buena iluminación; objeto bien visible y, si usas tracking por color, de
  **color distintivo** respecto del fondo.
- Para movimientos rápidos, graba en **cámara lenta** e indica el fps real de
  captura en *Opciones avanzadas*.

---

## Solución de problemas

- **El video no carga / "se congela":** suele ser el **códec HEVC/H.265** de
  algunos iPhone. Conviértelo a H.264:
  ```bash
  ffmpeg -i entrada.mov -c:v libx264 salida.mp4
  ```
- **Video muy grande:** el límite de subida está en 4 GB (`.streamlit/config.toml`).
- **Edité un archivo de `core/` y no se refleja:** Streamlit reutiliza los
  módulos ya cargados. **Reinicia el servidor** (Ctrl+C y relanzar); la tecla
  *R* solo re-ejecuta `app.py`.
- **No aparece CSRT:** tienes `opencv-python` en vez de `opencv-contrib-python`.
- **"Windows protegió tu PC" al abrir el `.bat`:** desbloquea el ZIP antes de
  extraerlo (ver más arriba).
- **Windows pide permisos de administrador:** no es Tracker UAI, es el
  instalador de Python. Cancela y deja que el lanzador use su copia portable
  (`Iniciar_Tracker_UAI.bat --portable`).
- **El entorno quedó a medias:** `Iniciar_Tracker_UAI.bat --reparar`.

---

## Tests

```bash
python -m pytest
```

Cubren `core/kinematics.py` y `core/fitting.py` (un movimiento sintético —MRU,
tiro parabólico, colisión, péndulo cónico— se ajusta con el error esperado), la
selección del objeto por clic y las figuras exportadas.

`tests/test_flujo_app.py` es una **prueba de integración de la interfaz**: usa
`streamlit.testing.AppTest` para ejecutar `app.py` headless y verificar que las
seis etapas se dibujen sin excepciones y que los botones **Continuar** lleven
del Paso 1 al Paso 6. Necesita `manual/ejemplos/carro.mp4`; si falta, se omite.

---

## Estructura del proyecto

```
tracker-uai/
├── app.py                       # interfaz Streamlit (solo UI)
├── export.py                    # CSV + figuras PNG/SVG + ZIP
├── requirements.txt
├── conftest.py                  # pone la raíz en sys.path para pytest
├── README.md                    # este archivo
├── MANUAL_DE_USO.md             # manual de uso para el usuario final
├── LEEME_ESTUDIANTES.txt        # guía corta para estudiantes
├── CHANGELOG.md                 # registro de cambios por versión
├── LICENSE                      # licencia MIT
├── Iniciar_Tracker_UAI.bat      # lanzador Windows (doble clic)
├── iniciar_tracker.sh           # lanzador macOS / Linux
├── Crear_paquete_para_compartir.bat  # arma un ZIP limpio para distribuir
├── .streamlit/config.toml       # tema y límite de subida
├── core/                        # lógica pura (sin Streamlit, testeable)
│   ├── video.py                 # lectura de video, caché de frames, overlay
│   ├── calibration.py           # escala px→m, origen, ejes rotados
│   ├── tracking.py              # tabla de puntos
│   ├── autotrack.py             # tracking automático (plantilla/color/CSRT)
│   ├── kinematics.py            # derivadas + Savitzky-Golay
│   └── fitting.py               # curve_fit, elipse, utilidades de presets
├── models/
│   └── experiment_presets.py    # presets por laboratorio
├── manual/                      # fuente LaTeX del manual (apunte UAI)
│   ├── manual-tracker-uai.tex
│   ├── uaiestilo.sty
│   └── manual-tracker-uai.pdf
└── tests/
    ├── test_kinematics.py
    ├── test_fitting.py
    ├── test_seleccion_y_export.py  # recuadro por clic, figuras exportadas
    └── test_flujo_app.py           # integración de la UI (streamlit AppTest)
```

---

## Limitaciones de esta versión y pendientes (v2)

- No corrige distorsión de lente ni perspectiva.
- **Multi-objeto simultáneo** no soportado (para colisiones, un carro por
  sesión). Marcado como TODO en el código.
- Sin cuentas de usuario ni persistencia entre sesiones: cada análisis se
  exporta y listo.
- Para videos muy largos (>2000 frames), la precarga se limita; conviene
  recortarlos.

---

## Créditos y licencia

Desarrollado por el **Prof. David Aguayo Vera** para los laboratorios de física
de la Universidad Adolfo Ibáñez. Distribuido bajo licencia **MIT** (ver
[LICENSE](LICENSE)): puedes usarlo, modificarlo y compartirlo libremente,
manteniendo el aviso de copyright.

Inspirado en [Tracker (OpenSourcePhysics)](https://physlets.org/tracker/).
