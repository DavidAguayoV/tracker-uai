# Tracker UAI v1.3 — Laboratorios 3 de FIS101 y FIS301

Fecha: 15 de octubre de 2026. Guías de referencia:
`FIS101/Lab3/FIS101_Lab3_v3.tex` y `FIS301/Lab3/Fis301_Lab3_v3.tex`, con la
«Revisión 2» de `CAMBIOS_LAB3.md`. Las guías no se tocaron.

Estado: 138 tests en verde (eran 69). Cuatro commits locales y la etiqueta
`v1.3`. **No se hizo push.**

---

## (a) Auditoría: lo que encontré

**Contra las guías de los Laboratorios 3**

1. **FIS101, preset «Colisiones».** Seguía un solo carro por sesión, como
   decía su ayuda: «multi-objeto es v2». Ajustaba una recta a x(t) a cada
   lado de un instante elegido con un deslizador. No descartaba los puntos de
   la interacción magnética, no aplicaba la convención de signos de la guía,
   no pedía las masas y no calculaba p, K ni C_R. Para tener las cuatro
   velocidades de un ensayo había que analizar el video dos veces y
   emparejar los instantes del choque a mano.
2. **FIS301, preset «Lorentz / MHD».** Ajustaba una recta a x(t) o y(t) y
   tomaba la pendiente como rapidez del flujo. Con la revisión 2 el trazador
   **gira alrededor de una mina**, así que esa pendiente no significa nada:
   en una vuelta completa sale ≈ 0. No había radio, ni tramo estable, ni s_v.
3. **Faltaban el modo por lotes y el análisis de grupo.** Cada video se
   analizaba por separado. Además, al cargar un video nuevo se borraba la
   calibración: había que calibrar 12 veces en FIS101 y 24 en FIS301, con la
   misma cámara fija.
4. **No hay archivo de proyecto.** La app no guarda ni abre proyectos: los
   «datos crudos» que puede entregar un estudiante son los CSV. Ahora
   también la tabla de ensayos o del lote, y un ZIP con la trayectoria de
   cada video de FIS301.

**Contra las convenciones (coma decimal, «;», 1/(n−1), 1 cifra significativa)**

5. **El CSV usaba «,» como separador y «.» como decimal.** Excel en español
   espera «;»: lo abría con todas las columnas metidas en la primera. El
   README decía «se abre directo en Excel».
6. **Las pantallas de los Labs 1 y 2 muestran punto decimal** (`f"{x:.3f}"`
   en todo `app.py`). Ver las limitaciones en (d).
7. La desviación estándar ya era muestral (`ddof=1` en `read_plateau`). No
   había redondeo de la incertidumbre a una cifra significativa.

**Errores encontrados al probar**

8. **Tiempos corridos en la precarga** (`core/video.py`). El timestamp se leía
   antes de `read()`, así que cada frame recibía el tiempo del anterior. Los
   frames 1 y 2 quedaban con el **mismo t** y una velocidad infinita, que el
   resumen del Paso 4 descartaba sin avisar. Afectaba a **todos** los labs,
   porque la app precarga siempre. Corregido; las posiciones no cambian.
9. **Seguimiento por color frágil para los Lab 3.** Toma el contorno más
   grande de **todo** el cuadro. Con dos carros marcados con la misma cinta
   puede saltar de uno a otro, y en FIS301 a una burbuja o un reflejo. Además:
   - la apertura morfológica de 5×5 px borra objetos de pocos píxeles;
   - el color sale de un solo píxel;
   - el tono no se trata como circular (el rojo está en ambos extremos de H).

   En `carro.mp4`, el color detecta la marca roja en 8 de 93 cuadros.
10. **La plantilla salta al objeto más parecido de todo el cuadro** cuando
    pierde la coincidencia, y solo entrega píxeles enteros. Con la plantilla
    en gris, el trazador naranja y la mina negra son dos discos oscuros casi
    iguales para la correlación (0,84 frente a 0,97). En la prueba sintética
    la plantilla saltaba a la mina cada vez que el trazador se ocultaba.
11. Algunos docstrings estaban desactualizados (`TODO (v2)` en
    `core/tracking.py`, las «etapas implementadas» de `app.py`).

---

## (b) Implementado

### Común a los dos laboratorios

- **Selector «Laboratorio»** en la barra lateral, sincronizado con el preset
  del Paso 5. Elegir un Lab 3 configura el seguimiento y activa la cámara fija.
- **Cámara fija.** La calibración (y la mina) se conservan al subir el video
  siguiente, salvo que cambie la resolución. Es una casilla del Paso 2.
- **Seguimiento robusto** (`core/seguimiento.py`). Se usa con varios objetos o
  en un Lab 3; los Labs 1 y 2 siguen con los métodos de siempre.
  - Predice la posición a velocidad constante y elige el candidato más
    cercano a la predicción.
  - Una compuerta de salto rechaza lo que queda demasiado lejos: ese cuadro
    queda **sin punto**, no se inventa una posición.
  - Excluye las posiciones de los otros objetos ya seguidos.
  - Si pierde el objeto varios cuadros seguidos, lo recupera buscando en todo
    el cuadro.
  - **Color:** muestra de 5×5 px, tono circular y apertura de 3×3 px.
  - **Plantilla:** en color, con un umbral relativo a la coincidencia típica y
    precisión subpíxel.
- **Formato** (`core/formato.py`). Coma decimal en pantalla (en lo nuevo) y en
  los CSV, que llevan «;»; desviación estándar con 1/(n−1); incertidumbre =
  máx(s, resolución) redondeada a una cifra significativa.
- **Etapa «📈 Análisis de grupo (Lab 3)»** (opción B, elegida al inicio de la
  sesión). Muestra números y gráficos, sin conclusiones ni comparación con
  valores esperados.

### FIS101 · Lab 3 (colisiones)

- **Dos objetos, Carro 1 y Carro 2**, que se siguen uno después del otro. El
  segundo seguimiento descarta lo que cae sobre el primero, así que las marcas
  pueden ser del mismo color.
- **Detección del choque** (`core/colisiones.py`). Usa el salto de la
  velocidad relativa v2 − v1, que salta incluso con un carro quieto o cuando
  los carros intercambian velocidades. Ajusta dos escalones a la señal, mide
  la transición y descarta 2 puntos más a cada lado.
- **Tramos antes y después**, comunes a los dos carros, automáticos o
  ajustables a mano con dos deslizadores.
- **v1i, v2i, v1f y v2f**: el promedio de cada tramo, con su desviación
  estándar, su número de puntos, la resolución y la incertidumbre.
- **Convención de la guía:** si el carro 1 parte hacia −x, la app invierte el
  signo de todo y lo avisa.
- **Avisos que no bloquean:**
  - un segundo choque (contra un tope) dentro de un tramo; además, el tramo
    automático se corta antes del rebote;
  - un tramo con menos de 8 puntos;
  - falta el carro 2;
  - p_i casi nulo (bloques B y D).
- **Con m1 y m2:** p_i, p_f, Δp, Δp/p_i, K_i, K_f y C_R, con las ecuaciones de
  la guía. C_R se muestra aunque falten las masas. Δp se muestra siempre junto
  a Δp/p_i.
- **Gráfico como el de la guía**, con los tramos sombreados y los promedios en
  línea discontinua, descargable en PNG.
- **Tabla de ensayos.** El número de ensayo y el bloque salen de
  `ensayoNN.mp4`. Se descarga y se vuelve a cargar en CSV.
- **Análisis de grupo:** promedio y s de Δp/p_i, Δp y C_R por bloque A–D, sin
  mezclar bloques, y gráficos por ensayo.

### FIS301 · Lab 3 (bombeo MHD)

- **Paso 5** (`core/orbita.py`):
  - rapidez |v| cuadro a cuadro;
  - tramo de giro estable: descarta el arranque (mediana móvil hasta el 90 %
    del régimen), o se elige a mano;
  - **v** (promedio) y **s_v** (desviación estándar cuadro a cuadro, la que
    la guía usa como incertidumbre de v);
  - **radio**: distancia media al centro de un círculo ajustado (Kåsa más
    Gauss-Newton), con su dispersión, visible en un gráfico del radio en el
    tiempo;
  - si se marca la mina, **la distancia a la mina** y un aviso si el centro de
    la órbita no coincide con ella;
  - vueltas, sentido de giro y un **control ω·R** a partir del ángulo barrido.
- **Etapa «🌀 Lote FIS301»:**
  - una calibración, un color del trazador y una mina para todos los videos;
  - **bloque y voltaje leídos del nombre** (`D0_10V.mp4`; también acepta
    `d2_24v.MOV` o `D1-18V.mp4`);
  - procesa automáticamente: en el primer cuadro toma la mancha más cercana a
    la mina;
  - **I (mA) editable** por video; si un video se reprocesa, su I se conserva;
  - tabla resumen con bloque, V, I, v, s_v, u_v, r, s_r, d_mina, N y
    vueltas;
  - aviso si el radio de un video se aleja más de un 25 % de la mediana de su
    bloque;
  - descarga en CSV, o un ZIP con la tabla y los datos crudos de cada video;
  - la tabla se vuelve a cargar desde CSV;
  - vista de la órbita de cada video.
- **Análisis de grupo:**
  - **v = k·I por el origen** para cada distancia, con k, su error estándar
    √(SS_res/(n−1)/ΣI²) y R²;
  - gráfico **v(I)** con las rectas y barras de u_v;
  - gráfico **v(V)** sin ajuste.

---

## (c) Verificación con videos sintéticos

Los videos se generan en cada corrida de pytest (`tests/sinteticos.py`); no se
versionan, porque `*.mp4` está en `.gitignore`. El procesamiento es el mismo
que en la app: seguimiento robusto y análisis, sin otra intervención que el
clic inicial sobre cada objeto. Tolerancia: **2 %** de cada valor, o del 2 %
de v1i cuando el valor exacto es 0. Para regenerar la tabla:
`python tests/verificar_lab3.py`.

**FIS101: dos carros con la misma marca naranja.** 1,5 mm/px, 30 fps,
interacción magnética de 0,1 s que conserva p en cada instante. Caso A:
m1 = m2 = 250 g, elástico. Caso D: m1 = 250 g, m2 = 500 g, e = 0,7.

| Caso | Método | v1i | v2i | v1f | v2f | C_R |
|---|---|---|---|---|---|---|
| A, esperado | — | +0,3000 | 0,0000 | 0,0000 | +0,3000 | 1,0000 |
| A | color | +0,3005 | +0,0002 | +0,0001 | +0,3006 | 1,0005 |
| A | plantilla | +0,3001 | 0,0000 | 0,0000 | +0,3000 | 0,9999 |
| D, esperado | — | +0,3000 | −0,2000 | −0,2667 | +0,0833 | 0,7000 |
| D | color | +0,3000 | −0,2009 | −0,2657 | +0,0835 | 0,6970 |
| D | plantilla | +0,3001 | −0,1999 | −0,2666 | +0,0833 | 0,6999 |
| D, eje invertido | color | +0,3007 | −0,2013 | −0,2669 | +0,0838 | 0,6987 |
| D, eje invertido | plantilla | +0,3001 | −0,2000 | −0,2666 | +0,0833 | 0,6997 |

Velocidades en m/s. **Peor error: 0,63 %** (v2i, color, eje invertido). En
los seis casos:

- el choque se detectó en 1,18–1,22 s (exacto: 1,20 s);
- ningún tramo tocó la interacción magnética;
- los carros nunca se confundieron (error máximo de posición < 1 px);
- con el eje invertido, el signo se corrigió solo.

En el caso D, p_i = −0,025 kg·m/s y Δp/p_i salió entre −3 % y +4 %, aunque
la simulación conserva p exactamente: es el problema de los bloques B y D
que ya anota la guía, y la app lo avisa.

**FIS301: trazador de 2 mm alrededor de una mina.** 0,2 mm/px, 30 fps, ruido
de imagen, temblor de 0,08 px, 40 burbujas blancas y 4 cuadros con el
trazador oculto.

| Órbita | Método | v esperada | v obtenida | r esperado | r obtenido | d_mina |
|---|---|---|---|---|---|---|
| R = 5 mm, v = 5 mm/s, horario | color | 5,000 | 5,037 (+0,74 %) | 5,000 | 5,001 | 5,000 |
| ídem | plantilla | 5,000 | 5,022 (+0,44 %) | 5,000 | 5,001 | 5,005 |
| R = 4 mm, v = 3 mm/s, antihorario, arranque de 1,5 s | color | 3,000 | 3,059 (+1,98 %) | 4,000 | 4,005 | 4,005 |
| ídem | plantilla | 3,000 | 3,017 (+0,57 %) | 4,000 | 3,999 | 3,999 |

Rapidez en mm/s; radio y distancia en mm. **Peor error: 1,98 %** (rapidez,
color, órbita lenta). Además:

- los 4 cuadros ocultos quedaron sin punto, en vez de saltar a la mina o a
  una burbuja;
- el sentido de giro salió correcto en los cuatro casos;
- el arranque se descartó: el tramo estable empezó en 1,27–1,37 s;
- s_v salió entre 0,39 y 0,62 mm/s, y la resolución es de 3 mm/s (ver (e)).

El error de la rapidez es siempre **positivo**: el ruido infla |v| (ver (d)).
El control ω·R dio 5,000 y 3,004 mm/s.

**Modo por lotes.** Se procesan dos videos (`D0_20V`, `D1_12V`) con una
calibración, un color y la mina, sin clics: bloque, V, v y r salen dentro del
2 %. Se probó en `core` y también subiendo los archivos en la interfaz.

**`manual/ejemplos/carro.mp4` (un objeto, flujo de los Labs 1 y 2).** Lo
corrí antes y después de los cambios con la misma selección. Plantilla: 93 de
93 puntos. Color: 8 de 93.

- **Posiciones y cuadros: idénticos**, con los dos métodos.
- **Lo único que cambia son los tiempos**, por la corrección del error 8 de
  (a):
  - la duración pasó de 3,033 a 3,067 s (92/30 s, el valor correcto);
  - la rapidez media pasó de 0,6007 a 0,5933 m/s, porque los frames 1 y 2 ya
    no tienen velocidad infinita y entran al promedio.
- El CSV ahora lleva «;» y coma decimal.
- En un objeto no aparecen el selector de objetos ni el seguimiento robusto.
  Lo verifica un test.

**Tests.** La suite pasó de 69 a 138:

| Archivo | Tests | Qué cubre |
|---|---|---|
| `test_lab3_core.py` | 40 | Lógica de los Lab 3, sin video |
| `test_lab3_videos.py` | 11 | De extremo a extremo, con videos sintéticos |
| `test_flujo_lab3.py` | 18 | AppTest de ambos flujos |

Los 18 de AppTest incluyen:

- seguir los dos carros desde la interfaz;
- tramos a mano;
- guardar un ensayo y verlo en el análisis de grupo;
- subir videos al lote y procesarlos;
- importar el CSV del lote;
- la cámara fija con y sin la casilla.

Los 69 tests anteriores siguen pasando sin cambios.

---

## (d) Limitaciones conocidas

1. **La rapidez cuadro a cuadro tiene un sesgo positivo por el ruido** (FIS301).
   - **Causa:** el promedio de |v| de vectores ruidosos supera a la rapidez
     verdadera, en ≈ σ²/(2v). La guía pide exactamente ese promedio.
   - **Tamaño:** en los videos sintéticos fue de +0,4 % a +2 %. Es mayor
     cuanto más lento es el giro y más ruidoso el seguimiento.
   - **Control:** la app muestra ω·R, que no tiene ese sesgo, y avisa si v lo
     supera en más de un 5 %.
   - **Recomendación:** seguir por **color**, que en la prueba sintética fue
     el método más robusto para el trazador.
2. **La resolución domina la incertidumbre** con la definición actual (ver
   (e) 2): 0,02 m/s en FIS101 a 1,5 mm/px y 3 mm/s en FIS301 a 0,2 mm/px.
3. **Objetos uno a la vez.** Los objetos no se siguen en una sola pasada: el
   segundo usa los puntos ya seguidos del primero para excluirlos. Si el
   carro 1 se siguió mal, el carro 2 hereda el error.
4. **Tramos comunes a los dos carros.** Si un solo carro choca con un tope,
   el tramo «después» se acorta para los dos.
5. **Segundos choques.** Se detecta un salto mayor que el 25 % de la escala de
   velocidades del ensayo (o que 6 veces el ruido). Un rebote muy lento puede
   no detectarse.
6. **Modo por lotes:**
   - el trazador debe verse en el primer cuadro de cada video, y debe ser la
     mancha de su color más cercana a la mina;
   - los videos subidos quedan en la memoria del navegador y del servidor:
     con 24 videos pesados conviene subirlos por bloque;
   - las tablas viven en la sesión: si se recarga la página se pierden, salvo
     las descargadas, que se pueden volver a subir.
7. **La plantilla robusta deja más huecos que la original.** En `carro.mp4`
   da 82 de 93 cuadros, frente a 93 de 93: rechaza las coincidencias débiles
   en vez de aceptarlas. Por eso los Labs 1 y 2 siguen con la plantilla
   original.
8. **Coma decimal incompleta.**
   - Ya usan coma: todo lo nuevo (pantallas de los Lab 3, gráficos Plotly y
     PNG nuevos, todos los CSV).
   - Siguen con punto: los números de las pantallas de los Labs 1 y 2, las
     figuras PNG antiguas del Paso 6 y las tablas editables
     (`st.data_editor`, como la columna I del lote).
9. **Sin corrección de perspectiva ni de lente**, como antes.
10. **Videos sintéticos idealizados.** No tienen desenfoque de movimiento, ni
    perspectiva, ni sombras. Validan la cadena de cálculo, no la robustez
    frente a videos reales del laboratorio, que no tuve.
11. **El manual en LaTeX** (`manual/manual-tracker-uai.tex` y su PDF) no se
    actualizó. Sí se actualizaron `MANUAL_DE_USO.md`, `LEEME_ESTUDIANTES.txt`,
    `README.md` y `CHANGELOG.md`.

---

## (e) Puntos que David debe decidir

1. **Las guías nombran «Tracker», no «Tracker UAI».**
   - FIS101, receta del Anexo A: «herramienta *calibration stick*», que es de
     Tracker de OSP.
   - FIS101, Anexo B: datos crudos = «archivos de *Tracker* de los doce
     ensayos (o su exportación en CSV)».
   - FIS301: «Control con Tracker», «En Tracker se anota el radio» y lo mismo
     sobre los datos crudos.

   Tracker UAI no produce archivos `.trk`. Entrega:
   - FIS101: el CSV del Paso 6 por ensayo y la tabla de ensayos;
   - FIS301: la tabla resumen y el ZIP con la trayectoria de cada video.

   Hay que decidir si las guías nombran Tracker UAI y piden esos archivos.
   `CAMBIOS_LAB3.md` (FIS301, punto 3) ya dejaba abierta la pregunta de `.trk`
   o CSV.
2. **Qué es «la resolución».** La guía de FIS301 dice «la resolución de la
   calibración de Tracker» y la de FIS101 no la define para velocidades. Usé
   **escala ÷ (2Δt)**: lo que cambia una velocidad por diferencias centradas
   al mover un punto un píxel.
   - **FIS101** (1,5 mm/px, 30 fps): 0,02 m/s, mayor que s en los sintéticos
     (0,001–0,01 m/s).
   - **FIS301** (0,2 mm/px, 30 fps): 3 mm/s, cuando v vale 3–5 mm/s. Con esta
     regla, u(v) es del orden de v en casi todos los videos.

   Opciones:
   - mantenerla;
   - usar una resolución subpíxel (por ejemplo, 0,1 px);
   - usar solo s_v, como sugiere la primera frase de la sección
     «Incertidumbre de la medición de velocidad».

   Es una constante en `core/colisiones.py` (`resolucion_velocidad`).
3. **R² de la recta por el origen.** Se calcula como Excel con la intersección
   en cero (1 − SS_res/Σv²), que es lo que el grupo vería en su planilla. No
   es comparable con el R² de una recta con intercepto. ¿Se acepta esa
   definición?
4. **El radio no tiene columna en la tabla de registro de FIS301.** La guía
   dice «En Tracker se anota el radio de la órbita» y la tabla de errores lo
   usa, pero la Tabla «Registro de los tres bloques» solo tiene V, I y v. La
   app exporta r, s_r y d_mina en el resumen. ¿Se agrega una columna a la
   guía?
5. **Δp no tiene columna en la tabla de cálculos de FIS101.** La obs. de la
   guía pide «Reporte siempre ambos: Δp en kg·m/s y Δp/p_i en %», pero la
   Tabla «Cálculos derivados por ensayo» solo tiene Δp/p_i. La app muestra y
   exporta los dos.
6. **Bloque D** (`CAMBIOS_LAB3.md`, FIS101 (e) 1). Si se elige normalizar Δp
   por |m1·v1i| + |m2·v2i|, la app ya calcula esa escala (`escala_p`, la usa
   para el aviso de p_i pequeño): mostrarla es un cambio de una línea.
7. **Incertidumbre por bloque en FIS101.** La guía dice: «la desviación
   estándar de las repeticiones o la dispersión de los puntos dentro de un
   tramo, la que sea mayor». Para Δp/p_i y C_R, esa segunda cantidad no está
   en las mismas unidades. La pestaña de grupo muestra solo s de las
   repeticiones. ¿Basta?
8. **Análisis de grupo dentro de la app** (opción B). Contradice la decisión
   de agosto que registra el CHANGELOG v1.1 y la receta «Ajuste del grupo»
   de FIS301, que el grupo haría en su planilla. Hay que decidir si la guía
   la menciona, o si se pide igual el ajuste propio y se usa la pestaña como
   control.
9. **El CSV cambió de formato** («;» y coma) para todos los labs. Quien tenga
   Excel en inglés o lea los CSV con un script verá el cambio. Avisar en la
   próxima entrega.
10. **Duración de la grabación en FIS301.** La guía pide 5 a 10 s y «al menos
    una vuelta». Con v ≈ 4 mm/s y r ≈ 5 mm, una vuelta dura ≈ 8 s: con 5 s la
    app avisará «menos de una vuelta» casi siempre. Conviene pedir 10 s o más.
11. **Sentido de giro** (objetivo 4, polaridad). La app informa «horario» o
    «antihorario» visto desde arriba. Sirve si la cámara mira hacia abajo sin
    espejar la imagen; algunos celulares espejan la cámara frontal. ¿Se usa
    para el objetivo 4 o se deja como dato?
12. **Parámetros por defecto que conviene confirmar:**
    - 2 puntos descartados a cada lado del salto;
    - aviso bajo 8 puntos por tramo (la tarea decía «~8»);
    - tramo estable de FIS301 desde que la rapidez llega al 90 % del régimen;
    - aviso de radio a más de un 25 % de la mediana del bloque.
13. **Manual en LaTeX** (`manual/`): ¿se actualiza con una sección de los
    Laboratorios 3, como la del Lab 1?
14. **Cambio local tuyo sin incluir.** Antes de empezar había un cambio en
    `manual/uaiestilo.sty` y un `Actualizar_GitHub.bat` sin versionar. No los
    toqué ni los incluí en los commits.
