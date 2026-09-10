# Salida de ejemplo

Artefactos producidos por una corrida completa sobre el material de muestra:

```bash
python main.py --input data/raw --output output --method all
```

## Contenido

```
run_metadata.json                  argumentos, dispositivo, versiones, commit, fallos
<video>/method_1/
├── <video>_osd.mp4                video anotado: cajas por nivel de riesgo, pretil, HUD
├── berm_height.{png,svg}          curva temporal de altura con banda de incertidumbre
├── vehicle_spatial.{png,svg}      dispersión cenital + matriz de distancias mínimas
├── berm_profile.csv               serie por frame: altura, cobertura, condición, distancia
├── proximity_events.csv           posición y nivel de riesgo por equipo y frame
├── distance_matrix.csv            distancia mínima registrada por par de equipos
└── metadata.json                  FPS, ms/frame por etapa, alertas, resoluciones, jitter
```

Los cuatro videos están procesados con el **Método 1** (camino óptimo). El **Método 2**
—la línea base por `argmax`— se incluye sólo sobre `video_02`, como evidencia del
contraste que analiza `reporte_benchmark.md`. Comparar `video_02/method_1` con
`video_02/method_2` muestra la diferencia de estabilidad del perfil.

Cada gráfico va acompañado del CSV que lo origina, para que las figuras se puedan
reproducir desde los datos en lugar de tener que creerlas.

---

## Dos advertencias sobre cómo leer esto

### Tres de los cuatro videos son resultados *dentro de muestra*

El detector se entrenó con frames de `video_01`, `video_02` y `video_03`. Sus salidas
aquí están producidas por un modelo que vio esos frames exactos durante el
entrenamiento, y por tanto **se ven mejor de lo que el sistema realmente es**.

`video_04` se reservó íntegro como conjunto de validación y **nunca se usó para
entrenar**. Es la única evidencia fuera de muestra de este directorio, y por tanto el
único predictor honesto de cómo se comportará el sistema sobre material nuevo.

Las métricas del reporte se calculan sobre `video_04`.

### Los videos se recodificaron para el entregable

El pipeline escribe `mp4v`, elegido porque es el único códec siempre disponible en las
ruedas de OpenCV y por tanto el que garantiza que la imagen funcione sin retoques en
una máquina ajena (ver la nota en `bermguard/io/video_writer.py`).

Los archivos de este directorio se recodificaron a **H.264 (CRF 24)** para que quepan
en el repositorio: 109 MB pasan a 19 MB sin cambio visible. El contenido —cada frame,
cada caja, cada curva— es el que produjo el pipeline. Al ejecutar el contenedor se
obtienen los mismos artefactos en `mp4v`, más pesados.

---

## Qué mirar primero

1. **`video_04/method_1/video_04_osd.mp4`** — la única salida fuera de muestra. Cajas
   verdes, ámbar y rojas según proximidad, con identidad persistente por equipo, y el
   pretil delineado sobre el quiebre de terreno.

   Conviene mirarlo sabiendo qué se degrada aquí y no en los otros tres. En el frame
   130, por ejemplo, el CAEX y el bulldozer se detectan como **entidades separadas** y
   ambos en nivel de precaución —que es el comportamiento que el modelo base no podía
   producir—, pero la caja del camión se extiende bastante más allá de la máquina, y la
   curva del pretil se engancha a la nube de polvo en el tercio derecho del frame.
   Ambos defectos son los que el `reporte_benchmark.md` cuantifica: precisión 0.297 en
   el detector, y una segmentación que sigue estructuras de gradiente fuerte sin poder
   distinguir terreno de polvo.
2. **`video_01/method_1/berm_height.png`** — la curva de altura con su banda de
   incertidumbre y los huecos declarados en los dos cortes de escena que tiene ese clip.
3. **`video_02/method_1` contra `video_02/method_2`** — el contraste entre las dos
   formulaciones, en el campo `berm_crest_jitter_px` de cada `metadata.json`.
