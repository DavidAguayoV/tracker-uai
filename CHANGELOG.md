# Registro de cambios

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [1.1] — 2026-08-29

Dos frentes: que instalar la app en Windows deje de doler, y que los presets
del Laboratorio 2 correspondan a lo que piden las guías vigentes 2026.

### Instalación en Windows sin fricción

- **El lanzador resuelve Python por sí solo y nunca pide administrador.**
  `Iniciar_Tracker_UAI.bat` busca, en orden: un entorno ya preparado, el
  lanzador `py -3`, `python`, e instalaciones por usuario en
  `%LOCALAPPDATA%\Programs\Python`. Antes fallaba en seco si `python` no estaba
  en el `PATH`, aunque hubiera un Python perfectamente utilizable en el equipo.
- **Python portable como último recurso.** Si no encuentra ninguno, ofrece
  descargar la distribución *embeddable* oficial de python.org (~11 MB) en
  `%LOCALAPPDATA%\TrackerUAI\python`, habilita `site-packages` en su archivo
  `._pth` e instala `pip`. No toca el `PATH`, no requiere administrador y se
  desinstala borrando esa carpeta. Los avisos de permisos que reportaban los
  colegas venían del instalador de python.org, no de la app.
- **Dos rescates:** `--portable` fuerza la copia portable aunque haya Python en
  el equipo; `--reparar` borra el entorno y lo rehace desde cero.
- **Mensajes de error en español** que dicen qué hacer en cada caso (sin red,
  red del campus bloqueando pypi.org, instalación a medias).
- **Documentado el paso que evita "Windows protegió tu PC":** desbloquear el
  ZIP *antes* de descomprimir. El aviso viene del *Mark-of-the-Web*, que se
  propaga del ZIP a cada archivo extraído, incluido el `.bat`.
- `conftest.py` pone la raíz del proyecto en `sys.path`: el Python embebido no
  agrega el directorio de trabajo y sin eso `pytest` no encuentra `core`.
- El empaquetador `Crear_paquete_para_compartir.bat` incluye `LICENSE` y
  `conftest.py`.

### Laboratorios 2 de FIS101 y FIS201

- **Lectura de meseta configurable** (FIS101). El tramo se elige por número de
  cuadros (por defecto 10, dentro del 8–15 que pide la guía) o por intervalo de
  tiempo, útil cuando el video sigue después de que el filtro toca el suelo. Se
  avisa si el número de cuadros queda fuera del rango y si el tramo todavía
  tiene pendiente —o sea, si no es meseta—, indicando cuánto crece la rapidez a
  lo largo de él. El gráfico sombrea el tramo promediado.
- **Control de orden de magnitud por video.** Un panel plegable calcula `C_d`
  de esa caída (FIS101, a partir de la masa y el diámetro, con área frontal y
  número de Reynolds) o `I_zz` de ese largo de hilo (FIS201, con el radio de
  giro comparado con `L/√12`). Ambos avisan explícitamente que **no** son el
  resultado del informe: sirven para cazar en el momento un error de
  calibración, de unidades o de lectura.
- **Avisos alineados con las guías** (FIS201): se advierte si se midieron menos
  de 10 oscilaciones, se muestra la incertidumbre de cronometraje a partir del
  fps efectivo, y se recuerda que en las fórmulas entra `d`, la semiseparación
  de los hilos, y no el largo de la barra (confundirlos da un momento de
  inercia nueve veces mayor).
- **Textos de los presets reescritos** contra las guías vigentes: plan de
  configuraciones, criterio de lectura, y qué ajuste entrega el resultado que
  se informa.
- **Nuevas utilidades en `core/fitting.py`**, todas con tests: `read_plateau`,
  `frontal_area`, `reynolds_number`, `drag_from_terminal_velocity`,
  `bifilar_inertia`, `bar_gyration_radius`, `period_timing_uncertainty`, más
  las constantes de las guías (`G_SANTIAGO`, `RHO_AIRE_SANTIAGO`, `MU_AIRE`).

### Notas

- El ajuste lineal del grupo (`v_t²` contra `m`, `T²` contra `H`) sigue
  haciéndose fuera de la app, en Excel, como indican las guías.
- La suite pasó de 19 a 39 tests.
- `plateau_value()` se mantiene como envoltorio de `read_plateau()`: el código
  existente no cambia de comportamiento.

## [1.0] — 2026-08-04

Primera versión estable: análisis de video en seis etapas (Video, Calibración,
Tracking, Cinemática, Ajuste, Exportar), pestañas de Gráficos y Medición,
presets por laboratorio, lanzadores para Windows y macOS/Linux, y manual del
estudiante en LaTeX.
