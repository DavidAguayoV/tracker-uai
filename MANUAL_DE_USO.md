# Manual de uso — Tracker UAI

Aplicación para **analizar videos de experimentos de física** y **graficar
datos experimentales**, pensada para los laboratorios de la UAI (FIS101 /
FIS201 / FIS301).

Funciona en **Windows, macOS y Linux** (está hecha en Python, que es
multiplataforma).

---

## 1. Requisitos

- **Python 3.11 o superior** (probado hasta 3.14). Es lo único que debes
  instalar tú; el resto lo instala la app sola la primera vez.
- Un navegador web (Chrome, Edge, Firefox, Safari).
- Conexión a internet solo la **primera vez** (para descargar las librerías).

---

## 2. Instalación de Python (una sola vez)

### Windows
1. Entra a <https://www.python.org/downloads/> y descarga Python 3.11+.
2. En el instalador, **marca la casilla "Add Python to PATH"** antes de
   continuar. (Alternativa: instalar "Python 3.12" desde la Microsoft Store.)

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
python --version      # en Windows
python3 --version     # en macOS/Linux
```
Debe mostrar `Python 3.11` o superior.

---

## 3. Iniciar el programa

### Windows
- Doble clic en **`Iniciar_Tracker_UAI.bat`**.

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

## 4. Guía de uso

La barra lateral izquierda tiene el **selector de etapas** y la **carga del
video**. Hay dos grandes usos:

- **Análisis de video** (etapas 1 a 6): medir posiciones en un video y obtener
  velocidad, aceleración, ajustes, etc.
- **Gráficos** (pestaña 📊): graficar datos escritos a mano, **sin video**.

### Análisis de video

**Etapa 1 · Video**
1. Sube tu video (mp4, mov, avi, mkv, m4v o webm) en la barra lateral.
2. Usa el reproductor para ubicar el tramo de interés.
3. Con el slider y los botones, ubica el frame exacto.
4. Define el **intervalo de procesamiento** (frame inicial y final): así el
   análisis trabaja solo sobre ese tramo y es mucho más rápido.

**Etapa 2 · Calibración** (convierte píxeles a metros)
1. Marca **dos puntos** de una distancia que conozcas (ej. una regla) y escribe
   esa distancia en metros.
2. Marca el **origen** de coordenadas.
3. (Opcional) Activa **"Ejes con orientación libre"** si tu eje X no es
   horizontal (ej. un plano inclinado) y marca la dirección +X.
4. Pulsa **Aplicar calibración**.

**Etapa 3 · Tracking** (seguir el objeto)
- **Manual:** haz clic sobre el objeto en cada frame; el video avanza solo.
- **Automático:** elige *Plantilla* (arrastras un recuadro sobre el objeto),
  *Color* (haces clic para tomar su color) o *CSRT*. Verás la detección en vivo
  y, al terminar, todas las detecciones superpuestas.
- **Corregir:** navega frame por frame y haz clic para mover un punto mal
  detectado, o bórralo. También puedes editar la **tabla** de datos directamente.

**Etapa 4 · Cinemática**
- Muestra los gráficos de posición, velocidad, aceleración y trayectoria.
- Activa **Suavizar** si los datos tienen ruido (útil sobre todo para la
  aceleración).

**Etapa 5 · Ajuste**
- Elige un **preset** según tu laboratorio (gravedad, velocidad terminal,
  colisiones, péndulo cónico, bifilar, MHD) o el modo **genérico**.
- El programa entrega los parámetros con su **incertidumbre** y el **R²**.

**Etapa 6 · Exportar**
- Marca con **casillas** qué gráficos quieres y descarga un **ZIP** con las
  imágenes (PNG) y el **CSV** de todos los datos.

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

## 5. Consejos para grabar los videos

- Cámara **perpendicular** al plano del movimiento y **fija** (trípode o apoyada).
- La regla de calibración debe estar **en el mismo plano** que el movimiento.
- Buena iluminación; objeto bien visible y, si usarás color, **de un color
  distinto al fondo**.
- Para movimientos rápidos, graba en **cámara lenta** y activa el fps de captura
  en *Opciones avanzadas*.

---

## 6. Solución de problemas

| Problema | Solución |
|---|---|
| "No se encontró Python" | Reinstala Python marcando **Add to PATH** (Windows). |
| El video `.mov` no carga o se cuelga | Suele ser el códec HEVC. Conviértelo a H.264: `ffmpeg -i entrada.mov -c:v libx264 salida.mp4` |
| El video es muy largo / lento | En la Etapa 1, elige un **intervalo** corto de frames. |
| No aparece el método CSRT | Instalaste `opencv-python` en vez de `opencv-contrib-python`. |
| Editar frame por frame va lento | Acota el intervalo; usa tracking automático y corrige solo los puntos malos. |

---

## 7. Compartir el programa

- **Windows:** ejecuta `Crear_paquete_para_compartir.bat` para generar un `.zip`
  limpio y envíalo. El estudiante sigue `LEEME_ESTUDIANTES.txt`.
- **macOS/Linux:** comparte la carpeta (sin `.venv`) y el archivo
  `iniciar_tracker.sh`.

En todos los casos, el único requisito en el equipo del estudiante es tener
**Python 3.11+** instalado.

---

*Ante cualquier duda, contacta a tu profesor o ayudante de laboratorio.*
