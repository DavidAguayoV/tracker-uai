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

La barra lateral izquierda tiene la **carga del video**, el selector de
**Laboratorio** (ajusta el seguimiento y el análisis a tu guía) y el
**selector de etapas**. Hay dos grandes usos:

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
6. Si grabaste varios videos **sin mover el celular**, marca **"Cámara fija:
   mantener esta calibración"**: al subir el video siguiente no hay que volver
   a calibrar. En los Laboratorios 3 viene marcado.

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
- **Varios objetos:** en *"Seguir más de un objeto en este video"* agregas
  otro objeto (hasta 4). Eliges cuál sigues con el selector **Objeto activo** y
  lo sigues igual que siempre. Con más de un objeto (o en un Laboratorio 3) se
  activa el **seguimiento robusto**: predice dónde estará el objeto, descarta
  lo que cae sobre otro objeto ya seguido y deja **sin punto** los cuadros en
  que no lo ve, en vez de inventar una posición.
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
- El **preset** es el laboratorio elegido en la barra lateral (gravedad,
  arrastre y velocidad terminal, colisiones, péndulo cónico, bifilar, MHD) o
  el modo **genérico**; puedes cambiarlo aquí también.
- El programa entrega los parámetros con su **incertidumbre** y el **R²**.

**Paso 6 · Exportar**
- Las figuras **se ven en pantalla** y cada una tiene su botón de descarga.
- También puedes bajar el **CSV** con todos los datos, o un **ZIP** con todo
  junto. Con varios objetos, el CSV trae todos (una fila por frame).
- El CSV usa **«;» como separador y coma decimal**: se abre directo en Excel
  configurado en español.

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

## 6. Laboratorios 3

Las magnitudes llevan los mismos nombres que en las guías (v1i, v2i, v1f, v2f,
p_i, p_f, Δp, K_i, K_f, C_R; V, I, v, s_v). En pantalla y en los CSV los
números van con **coma decimal**, y cada resultado con su **incertidumbre
redondeada a una cifra significativa**: la desviación estándar de los puntos
o la resolución, la mayor, sin propagación formal.

### 6.1 FIS101 · Lab 3 — Conservación del momentum lineal

Nombra los videos como dice la guía: `ensayo01.mp4` … `ensayo12.mp4`. Del
nombre la app saca el número de ensayo y el bloque (A: 1–3, B: 4–6, C: 7–9,
D: 10–12).

1. En la barra lateral elige **Laboratorio → FIS101 · Lab 3 — Colisiones
   (momentum)**. Aparecen dos objetos, **Carro 1** y **Carro 2**, y queda
   activada la cámara fija.
2. Sube `ensayo01.mp4`. **Paso 1:** elige el tramo desde que los dos carros
   ruedan libres (sin manos) hasta poco después del choque. Si un carro llega
   a un tope, deja ese rebote fuera del tramo si puedes.
3. **Paso 2:** calibra con la regla que está sobre el riel (una sola vez: los
   otros once videos usan la misma calibración). Si el riel no queda
   horizontal en la imagen, activa el eje X rotado y marca +X a lo largo del
   riel.
4. **Paso 3:** con **Carro 1** activo, haz clic sobre su marca de color y
   pulsa **Ejecutar seguimiento**. Cambia a **Carro 2**, haz clic sobre su
   marca y ejecuta de nuevo. Aunque las dos marcas sean del mismo color, el
   segundo seguimiento descarta lo que cae sobre el carro 1.
5. **Paso 5:** arriba están **v1i, v2i, v1f y v2f**, con signo e
   incertidumbre. Van a la tabla **«Velocidades con signo»** de la guía.
   - El eje x es paralelo al riel y **positivo en el sentido en que parte el
     carro 1**. Si en el video el carro 1 avanza hacia la izquierda, la app
     invierte el signo sola y lo avisa. Los rebotes salen negativos.
   - El gráfico muestra la velocidad de los dos carros, el instante del choque
     y los **tramos promediados** (sombreados). La app los propone sola,
     descartando la interacción magnética y dos puntos más a cada lado del
     salto. Con **"Ajustar a mano los límites de los tramos"** eliges otros.
   - Avisa, sin bloquear, si hay un **segundo choque** (contra un tope) dentro
     de un tramo o si un tramo tiene **menos de 8 puntos**. Si el carro 2 parte
     quieto (bloques A y C), v2i sale ≈ 0: es lo esperado.
6. Escribe **m1 y m2** en gramos, de la tabla **«Masas medidas»** (en C y D,
   m2 es el carro 2 **más** la masa adicional). Aparecen **p_i, p_f, Δp,
   Δp/p_i, K_i, K_f y C_R**: van a la tabla **«Cálculos derivados por
   ensayo»**. Anota también **Δp**: en los bloques B y D p_i casi se anula y
   Δp/p_i se dispara aunque Δp sea pequeño (la app lo avisa).
7. Revisa el número de ensayo y el bloque y pulsa **"Guardar este ensayo en la
   tabla"**. En el **Paso 6** descarga el CSV del ensayo: trae las posiciones
   y velocidades de los dos carros (son los **datos crudos** que se entregan).
8. Repite los pasos 2 a 7 con los doce videos. Al final, descarga la **tabla de
   ensayos (CSV)**. Si cierras la app, súbela de nuevo para seguir otro día.
9. Etapa **📈 Análisis de grupo → FIS101**: promedio y desviación estándar
   (1/(n−1)) de **Δp/p_i** y **C_R** por bloque, sin mezclar bloques, con Δp
   al lado. Es lo que pide la sección «Valor de cada bloque» de la guía. La
   pestaña no saca conclusiones: eso va en el informe.

### 6.2 FIS301 · Lab 3 — Fuerza de Lorentz y bombeo MHD

Nombra los videos como dice la guía: `D0_10V.mp4` … `D2_24V.mp4` (bloque de
distancia y voltaje). La corriente **I** se lee en la fuente durante la sesión
y se anota en la tabla **«Registro de los tres bloques»**.

**Modo por lotes (recomendado para los 24 videos).**

1. En la barra lateral elige **Laboratorio → FIS301 · Lab 3 — Fuerza de
   Lorentz / bombeo MHD**.
2. Sube **uno** de los videos en la barra lateral (por ejemplo `D0_10V.mp4`) y
   calíbralo en el **Paso 2** con la regla en el plano del trazador. Deja
   marcada la cámara fija.
3. Ve a la etapa **🌀 Lote FIS301 (24 videos)** y sube los videos (todos, o un
   bloque a la vez si pesan mucho).
4. En **"Color del trazador y posición de la mina"**: haz clic sobre el
   trazador (toma su color); luego elige **"la mina"** y haz clic sobre la mina
   alrededor de la cual gira. La máscara debe mostrar el trazador y casi nada
   más: ni burbujas ni la mina.
5. Pulsa **Procesar**. Para cada video, la app sigue el trazador, descarta el
   arranque y calcula, en el tramo de giro estable:
   - **v**: la rapidez media cuadro a cuadro (mm/s) → columna *v* de la tabla
     de registro;
   - **s_v**: su desviación estándar cuadro a cuadro, y **u_v**, la
     incertidumbre de v (s_v o la resolución, la mayor);
   - **r**: el radio de la órbita (círculo ajustado) y **d_mina**, la distancia
     a la mina, con su dispersión. La guía pide que sea parecido entre puntos:
     la app avisa si un video se aleja de la mediana de su bloque.
6. Escribe **I (mA)** en la columna editable de cada video, desde tu registro.
7. Revisa los avisos (menos de una vuelta, radio que cambia, trazador perdido)
   y, si quieres, la órbita de cada video en **"Ver la órbita de un video"**.
8. Descarga **"Resumen + datos crudos (ZIP)"**: la tabla resumen y la
   trayectoria de cada video, que se entregan con el informe. Si cierras la
   app, sube la tabla resumen para seguir.
9. Etapa **📈 Análisis de grupo → FIS301**: ajuste de **v = k·I por el
   origen** para cada distancia, con **k**, su **error estándar** y **R²**
   (calculado como Excel con la intersección en cero), el gráfico **v(I)** y
   el gráfico **v(V)** sin ajuste.

**Video a video.** Para el control de la guía al inicio de D0 (*"analice los
dos primeros videos antes de seguir"*) o para revisar un video dudoso: sube el
video, sigue el trazador por **Color** en el Paso 3 y abre el **Paso 5**. Ahí
marcas la mina, ves la órbita con el círculo ajustado, la rapidez cuadro a
cuadro con el tramo estable sombreado y el radio a lo largo del video, y con
**"Agregar a la tabla del lote"** (escribiendo I) lo sumas a la misma tabla.

---

## 7. Consejos para grabar los videos

- Cámara **perpendicular** al plano del movimiento y **fija** (trípode o apoyada).
- La regla de calibración debe estar **en el mismo plano** que el movimiento.
- Buena iluminación; objeto bien visible y, si usarás color, **de un color
  distinto al fondo**.
- Para movimientos rápidos, graba en **cámara lenta** y activa el fps de captura
  en *Opciones avanzadas*.

---

## 8. Solución de problemas

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
| El CSV se abre con todo en una columna | Tu Excel no está en español: usa *Datos → Desde texto/CSV* y elige «;» como separador. |
| En un Lab 3 el segundo carro "salta" al primero | Vuelve a seguir el carro 2 después del carro 1 (la exclusión usa los puntos ya seguidos del otro carro). |
| El trazador de FIS301 se pierde entre burbujas | Usa el método **Color** y sube la saturación mínima hasta que la máscara muestre solo el trazador. |

---

## 9. Compartir el programa

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
