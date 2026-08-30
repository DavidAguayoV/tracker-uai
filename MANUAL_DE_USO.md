# Manual de uso — Tracker UAI

Aplicación para **analizar videos de experimentos de física** y **graficar
datos experimentales**, pensada para los laboratorios de la UAI (FIS101 /
FIS201 / FIS301).

Funciona en **Windows, macOS y Linux** (está hecha en Python, que es
multiplataforma).

---

## 1. Requisitos

- **Windows:** nada. El lanzador se encarga de todo, y **nunca pide permisos
  de administrador**.
- **macOS / Linux:** Python 3.9 o superior (probado hasta 3.14).
- Un navegador web (Chrome, Edge, Firefox, Safari).
- Conexión a internet solo la **primera vez** (para descargar las librerías).

---

## 2. Antes de empezar en Windows: desbloquear el ZIP

Windows marca como sospechoso todo archivo bajado de internet, y esa marca se
copia a cada archivo que extraes. Por eso aparece la pantalla azul *"Windows
protegió tu PC"*. Se quita **una sola vez, antes de descomprimir**:

1. Clic derecho sobre el ZIP descargado → **Propiedades**.
2. Abajo, marca la casilla **Desbloquear** (si aparece) → **Aceptar**.
3. Recién ahora: clic derecho → **Extraer todo**.

Si ya lo descomprimiste sin desbloquear, borra la carpeta extraída, desbloquea
el ZIP y extrae de nuevo. Y si aun así aparece la pantalla azul:
**Más información → Ejecutar de todas formas**.

---

## 3. Instalación de Python

### Windows: no hace falta

El lanzador busca Python en tu equipo y, si no lo encuentra, te ofrece
descargar su **propia copia portable** (unos 11 MB, del sitio oficial
python.org). Responde que sí y listo. Esa copia:

- **no** pide permisos de administrador,
- **no** modifica el `PATH` ni afecta a otros programas,
- vive en `%LOCALAPPDATA%\TrackerUAI` y se desinstala borrando esa carpeta.

> Si prefieres instalar Python de todos modos, en el instalador de python.org
> elige **"Install for me only" / "Solo para mí"**: así tampoco te pedirá
> administrador. Marcar *"Add python.exe to PATH"* ayuda, pero ya no es
> obligatorio.

### macOS
- Opción A: descarga el instalador desde <https://www.python.org/downloads/>.
- Opción B (si usas Homebrew): `brew install python`

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

Para comprobar que quedó instalado, abre una terminal y escribe:
```bash
python3 --version
```
Debe mostrar `Python 3.9` o superior.

---

## 4. Iniciar el programa

### Windows
- Doble clic en **`Iniciar_Tracker_UAI.bat`**.
- Si algo falla, hay dos opciones de rescate. Mantén **SHIFT** + clic derecho
  en la carpeta → *"Abrir la ventana de PowerShell aquí"* y escribe:
  - `.\Iniciar_Tracker_UAI.bat --portable` → usa la copia portable de Python,
    ignorando la instalación del sistema.
  - `.\Iniciar_Tracker_UAI.bat --reparar` → borra el entorno y lo rehace desde
    cero (no toca tus videos).

### macOS / Linux
- Abre una terminal en la carpeta del programa y ejecuta:
  ```bash
  bash iniciar_tracker.sh
  ```
- (En macOS, si quieres iniciarlo con doble clic: renombra el archivo a
  `iniciar_tracker.command`, y la primera vez haz clic derecho → *Abrir*.)

**La primera vez** tarda unos minutos porque instala las librerías; la ventana
de la terminal queda abierta mostrando el avance. Es normal. Las siguientes
veces abre en pocos segundos.

El programa se abre solo en tu navegador (normalmente en
`http://localhost:8501`).

> **Método manual** (cualquier sistema, si prefieres la terminal):
> ```bash
> pip install -r requirements.txt
> python -m streamlit run app.py
> ```

**Para cerrar el programa:** cierra la ventana de la terminal (Windows) o
presiona `Ctrl + C` en ella (macOS/Linux).

---

## 5. Guía de uso

La barra lateral izquierda tiene el **selector de etapas** y la **carga del
video**. Hay dos grandes usos:

- **Análisis de video** (etapas 1 a 6): medir posiciones en un video y obtener
  velocidad, aceleración, ajustes, etc.
- **Gráficos** (pestaña 📊): graficar datos escritos a mano, **sin video**.

### Análisis de video

No hace falta que memorices el orden: **cada paso termina con un botón
"Continuar →"** que te lleva al siguiente, y la barra lateral muestra tu
progreso. Igual puedes saltar libremente a cualquier paso cuando quieras.

**Paso 1 · Video** — elegir el tramo a analizar
1. Sube tu video (mp4, mov, avi, mkv, m4v o webm) en la barra lateral.
2. Con el deslizador y los botones, ubica el frame donde **empieza** el
   movimiento y pulsa **INICIO**; luego ve al frame donde **termina** y pulsa
   **FINAL**.
3. Abajo verás qué quedó seleccionado: una barra con el tramo sobre la duración
   total, y las **miniaturas del primer y último frame**. Todo el análisis usa
   solo ese tramo, y así el programa va mucho más rápido.

**Paso 2 · Calibración** — convertir píxeles a metros
1. Sigue la lista numerada: haz clic en **un extremo** de tu regla, luego en
   **el otro**, y por último donde quieras el **origen (0, 0)**. El programa te
   dice en cada momento qué toca marcar.
2. Escribe cuánto mide de verdad esa distancia, **eligiendo la unidad**
   (cm, mm o m). Si mediste una regla de 30 cm, elige `cm` y escribe `30`.
3. **No hay que pulsar ningún botón:** la calibración se aplica sola en cuanto
   están los tres puntos.
4. Revisa la verificación: el programa te dice a cuánto equivale un píxel y qué
   ancho real tiene el encuadre. Si ese ancho no se parece a lo que ves en el
   video, revisa la distancia y su unidad.
5. (Opcional) Activa **"Mi eje X no es horizontal"** si trabajas en un plano
   inclinado, y marca la dirección +X.

**Paso 3 · Tracking** — seguir el objeto
- Viene en **Automático**, que es lo que conviene casi siempre:
  - *Plantilla* (o *CSRT*): **haz clic sobre el objeto**. Aparece de inmediato
    un recuadro verde que puedes agrandar o achicar con el deslizador, y un
    recorte ampliado de lo que se va a seguir. Si prefieres, también puedes
    arrastrar para dibujar el recuadro a mano.
  - *Color*: haz clic sobre el objeto para tomar su color.
  - Pulsa **Ejecutar seguimiento** y verás la detección en vivo.
- Al terminar, el programa avisa si quedaron **frames sin detección**.
- **Manual:** si el automático se pierde, haz clic sobre el objeto en cada
  frame; el video avanza solo.
- **Corregir:** en los desplegables puedes ver todas las detecciones
  superpuestas y corregir puntos frame por frame. También puedes editar la
  **tabla** de datos directamente.

**Paso 4 · Cinemática**
- Arriba, un **resumen del movimiento**: puntos analizados, duración, distancia
  recorrida, desplazamiento neto, rapidez media y máxima, aceleración media.
- Un bloque de **calidad de los datos** te avisa si hay mucho ruido, frames
  faltantes o pocos puntos para un ajuste confiable.
- Abajo, los gráficos de posición, velocidad, aceleración y trayectoria.
- Activa **Suavizar** si los datos tienen ruido (útil sobre todo para la
  aceleración). Las figuras que exportes usarán esa misma señal.

**Paso 5 · Ajuste**
- Elige un **preset** según tu laboratorio (gravedad, arrastre y velocidad
  terminal, colisiones, péndulo cónico, bifilar, MHD) o el modo **genérico**.
- El programa entrega los parámetros con su **incertidumbre** y el **R²**.

**Paso 6 · Exportar**
- Las figuras **se ven en pantalla** y cada una tiene su botón de descarga.
- También puedes bajar el **CSV** con todos los datos, o un **ZIP** con todo
  junto.

### 📊 Gráficos (datos manuales) — sin video

Ideal para Física I. Ejemplo: graficar la aceleración en función de `sin(θ)`
en el péndulo simple.

1. Escribe el **título** y los **nombres de los ejes**.
2. En la **tabla**, escribe tus valores de **X** e **Y** (agrega filas con el
   `+` de abajo).
3. El **gráfico de dispersión** aparece automáticamente.
4. Elige el **ajuste** (Lineal o Cuadrático): verás la **curva**, la
   **ecuación** y el **R²**.
5. Puedes crear **varias series** (con nombre y color) para comparar.
6. Descarga el gráfico en **PNG o SVG** para pegarlo en tu informe.

---

## 6. Consejos para grabar los videos

- Cámara **perpendicular** al plano del movimiento y **fija** (trípode o apoyada).
- La regla de calibración debe estar **en el mismo plano** que el movimiento.
- Buena iluminación; objeto bien visible y, si usarás color, **de un color
  distinto al fondo**.
- Para movimientos rápidos, graba en **cámara lenta** y activa el fps de captura
  en *Opciones avanzadas*.

---

## 7. Solución de problemas

| Problema | Solución |
|---|---|
| "Windows protegió tu PC" | Desbloquea el ZIP **antes** de extraerlo (sección 2). O bien: *Más información → Ejecutar de todas formas*. |
| Windows pide permisos de administrador | No es Tracker UAI, es el instalador de Python. Cancela y deja que el lanzador use su copia portable. |
| "No se encontró Python" | Vuelve a ejecutar el `.bat` y acepta la copia portable, o ejecútalo con `--portable`. |
| "Falló la instalación de las librerías" | Casi siempre es la red. Reintenta; si quedó a medias, ejecuta `.\Iniciar_Tracker_UAI.bat --reparar`. |
| El video `.mov` no carga o se cuelga | Suele ser el códec HEVC. Conviértelo a H.264: `ffmpeg -i entrada.mov -c:v libx264 salida.mp4` |
| El video es muy largo / lento | En la Etapa 1, elige un **intervalo** corto de frames. |
| No aparece el método CSRT | Instalaste `opencv-python` en vez de `opencv-contrib-python`. |
| Editar frame por frame va lento | Acota el intervalo; usa tracking automático y corrige solo los puntos malos. |

---

## 8. Compartir el programa

- **Windows:** ejecuta `Crear_paquete_para_compartir.bat` para generar un `.zip`
  limpio y envíalo. El estudiante sigue `LEEME_ESTUDIANTES.txt`.
- **macOS/Linux:** comparte la carpeta (sin `.venv`) y el archivo
  `iniciar_tracker.sh`.

En **Windows no hay requisitos previos**: si el equipo no tiene Python, el
lanzador ofrece descargar su propia copia portable, sin permisos de
administrador. Recuérdale a quien lo reciba que **desbloquee el ZIP antes de
descomprimir** (sección 2): es lo que evita la pantalla azul de Windows.

En **macOS/Linux** sí hace falta **Python 3.9+** instalado.

---

*Ante cualquier duda, contacta a tu profesor o ayudante de laboratorio.*
