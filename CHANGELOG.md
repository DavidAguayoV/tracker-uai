# Registro de cambios

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [1.2] — 2026-08-30

Revisión completa de la interfaz, centrada en que un estudiante que abre el
programa por primera vez sepa en todo momento qué tiene que hacer.

### Flujo guiado

- **Botones "Continuar →" al final de cada etapa.** Antes no existía ninguno:
  el selector de etapa de la barra lateral no tenía `key`, así que no había
  forma de cambiar de etapa por programa y el estudiante tenía que descubrir
  solo que debía volver a la barra lateral.
- **Indicador de progreso** en la barra lateral (tramo elegido / calibración
  aplicada / objeto seguido, con el número de puntos). La navegación libre
  entre etapas se mantiene: nada queda bloqueado.
- Cada etapa abre diciendo **para qué sirve**, en una frase.

### Paso 1 · Video

- La etapa deja claro que lo que se hace aquí es **recortar el tramo** a
  analizar, y por qué conviene.
- **Confirmación visual de lo seleccionado**: barra que sitúa el tramo sobre la
  duración total y **miniaturas del frame inicial y del final**.
- El aviso sobre perspectiva pasa a un desplegable, para que la etapa abra con
  la tarea y no con una advertencia.

### Paso 2 · Calibración

- **Checklist numerada** con el estado de cada punto y una marca de dónde estás,
  más una instrucción grande y explícita para el paso actual.
- **Selector de unidad (cm/mm/m)** para la distancia conocida. Pedirla solo en
  metros invitaba a escribir `30` por una regla de 30 cm: un error de factor
  100 que nada detectaba.
- **La calibración se aplica sola** al completarse. Ya no hace falta acordarse
  de pulsar un botón, cuyo olvido dejaba las columnas en metros vacías sin
  explicación visible.
- **Verificación de plausibilidad**: se muestra a cuánto equivale un píxel y el
  ancho real del encuadre, y se avisa si ese ancho no tiene sentido para un
  laboratorio.

### Paso 3 · Tracking

- **Arranca en modo automático** (plantilla). Marcar a mano cientos de frames
  era la principal causa de abandono en la primera sesión.
- **Un clic sobre el objeto basta**: crea un recuadro verde centrado ahí, con
  deslizador de tamaño y un **recorte ampliado de lo seleccionado**. Antes la
  selección era solo por arrastre y un clic simple producía un recuadro de área
  nula que se descartaba **en silencio**: no aparecía nada ni ningún mensaje. El
  arrastre sigue disponible; un arrastre de pocos píxeles se interpreta como
  clic para no generar plantillas inservibles.
- **La vista superpuesta ya no se genera sola.** Calcula una mediana sobre todos
  los frames y obligaba a esperar antes de poder seguir; ahora está tras un
  botón. En su lugar se avisa, sin costo, de los **frames sin detección**.

### Paso 4 · Cinemática

- **Resumen numérico del movimiento**: puntos analizados, duración, distancia
  recorrida, desplazamiento neto, rapidez media y máxima, aceleración media e
  intervalo entre puntos.
- **Bloque de calidad de los datos**: ruido de marcado estimado, frames
  faltantes y aviso si hay muy pocos puntos para un ajuste confiable.
- Los controles de suavizado pasan a un desplegable.

### Paso 6 · Exportar

- **Las figuras se ven en pantalla** y cada una tiene su propio botón de
  descarga. Antes había que marcar casillas y pulsar "Generar" a ciegas, y la
  etapa parecía exportar solo el CSV.
- El CSV y el ZIP con todo siguen disponibles.

### Corregido

- **La figura exportada no coincidía con la de la pantalla.** Con el suavizado
  activo, los gráficos mostraban la posición suavizada (`x_s`) pero los PNG
  exportaban la cruda (`x_m`): el informe terminaba con una figura distinta de
  la que el estudiante había visto.

### Notas

- La suite pasó de 39 a 69 tests. Los nuevos incluyen una prueba de integración
  con `streamlit.testing.AppTest` que verifica que las seis etapas se dibujen
  sin excepciones y que la cadena de botones lleve del Paso 1 al Paso 6.
- El Paso 5 (Ajuste) no cambió, salvo el botón de avance.

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
