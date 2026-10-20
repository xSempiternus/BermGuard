# ADR 0004: Entrenar con frames del propio material en vez de datasets públicos

- Estado: aceptada
- Fecha: 2026-09-08
- Depende de: ADR 0002

## Contexto

El ADR 0002 decidió entrenar un detector. Faltaba decidir con qué datos.

El enunciado sugiere datasets públicos y nombra Roboflow Universe. Evalué tres, fijándome
sobre todo en qué tan parecidos son a este material, porque lo que hizo fallar a YOLO-World
fue justamente la distancia entre sus datos de entrenamiento y estos videos.

| Dataset | Clases | Imágenes | Licencia | Problema |
|---|---|---|---|---|
| `brancos-workspace/mining-truck-detection-w53tl` | 1 (`haul_truck`) | 264 | CC BY 4.0 | No tiene bulldozer. Solo aporta la clase que el modelo base ya detecta, y parte de las imágenes son aumentos de un mismo frame |
| `mohamads-workspace-rl4ui/bulldozer-lpoe2` | 7, incluye `Bull_dozer` | 1.324 | CC BY 4.0 | Tiene las clases, pero son fotos de catálogo: la máquina llena el encuadre, bien iluminada y sin oclusión |
| `rf100-vl/apoce-...-construction-equipment` | 7, incluye `bulldozer` | 928 | MIT | Fotos cenitales de dron sobre obras urbanas, con objetos muy pequeños |

Ninguno se parece a estos videos: vista oblicua desde altura, distancia media, desierto con
polvo y más de la mitad de los frames de noche.

## Decisión

**Entrenar con frames anotados del propio material.** Los extraje con
`scripts/prepare_labeling_set.py`, estratificados por condición de luz, y anoté 171.

Creo que ese volumen alcanza por dos razones:

- El entrenamiento parte de pesos preentrenados, que ya saben de bordes, texturas y
  vehículos. Lo que falta enseñar es cómo se ven estas dos máquinas desde esta cámara.
- El material es homogéneo: misma faena, mismas máquinas y encuadres parecidos. En otro
  contexto sería un riesgo de sobreajuste, pero lo más probable es que el conjunto de
  evaluación venga del mismo lugar.

**Las clases son `caex` y `bulldozer`**, que son roles en el botadero. Los datasets
públicos etiquetan `truck` o `vehicle`, que no es la distinción que necesito.

## Alternativas descartadas

- **`bulldozer-lpoe2` como fuente principal.** Enseñaría a esperar máquinas grandes,
  centradas y bien iluminadas, lo contrario de este material. Queda como experimento:
  preentrenar con él y afinar con los frames propios.
- **APOCE.** Una vista cenital y una oblicua dan siluetas distintas de la misma máquina.
- **Juntar los tres datasets sin datos propios.** Más volumen no corrige el sesgo: 2.500
  imágenes fuera de dominio siguen fuera de dominio.

## Consecuencias

- Anotar me tomó cerca de dos horas.
- El modelo queda muy especializado y no va a generalizar a otras faenas ni a otros
  ángulos de cámara.
- La validación **no puede ser aleatoria**. Frames del mismo video son casi iguales, y una
  partición aleatoria mediría memoria y no generalización. Separé por video y dejé
  `video_04` completo para validación.

## Revisión (2026-10-18): preentrenar con datos públicos, como variante medida

Volví a buscar datasets públicos, ahora para el experimento que quedó abierto: preentrenar
con datos públicos y afinar con los frames propios. Revisé las imágenes de cada uno en
Roboflow Universe y conté, con el filtro por clase, cuántas tienen bulldozer:

| Dataset | Imágenes con bulldozer | Vista | Veredicto |
|---|---|---|---|
| [`mines-project/mines`](https://universe.roboflow.com/mines-project/mines-jo13g) (CC BY 4.0) | **68 de 261** (70 dozers, contados al importar; ver la revisión del 2026-10-20), más `dumper` | Fotos en una mina a cielo abierto, a media distancia, varias fechas entre 2022 y 2024; **polígonos** | **Usar.** El más parecido en escena y máquinas |
| [`bulldozer-lpoe2`](https://universe.roboflow.com/mohamads-workspace-rl4ui/bulldozer-lpoe2) (CC BY 4.0) | **1.324 de 1.324** | Mezcla de catálogo y fotos de faena a media distancia | **Usar.** Volumen y variedad de la forma del bulldozer |
| [`ccl-kl4nk/dji_65`](https://universe.roboflow.com/ccl-kl4nk/dji_65) (CC0) | 19 de 136 (20 dozers) | Dron oblicuo sobre una mina | **Descartado.** Los 19 frames son casi seguidos de un mismo vuelo (272–290 s y 212–219 s): en la práctica, una sola máquina |
| [`ws1/m`](https://universe.roboflow.com/ws1-i1agt/m-oi9pf) (CC BY 4.0) | 102 (+153 `dozer_night`) | Cámara de tablero de un camión minero, a nivel del suelo | **Descartado.** Otra perspectiva, dozers de pocos píxeles, y las nocturnas son versiones sintéticas de las diurnas (`*_day2night.jpg`) |
| [`rockaware/conference`](https://universe.roboflow.com/rockaware/conference-x8lzn) | — | Maquetas de juguete en una oficina | **Descartado** |
| [`minitruck/mining-truck-v1.4`](https://universe.roboflow.com/minitruck/mining-truck-v1.4) | 0 | Cámara fija, CAEX en primer plano | Sólo CAEX, que ya se detecta bien |
| [`igensol/mines-vehicles`](https://universe.roboflow.com/igensol/mines-vehicles) | 0 | CCTV nocturno en una mina | Sin bulldozer |
| [mineCareerDB](https://data.mendeley.com/datasets/c5s76mj4bm/5) (Mendeley) | 0 | Dron cenital a ~400 m | Sin bulldozer |

Corrijo lo que dije de `bulldozer-lpoe2` en el contexto: no son sólo fotos de catálogo,
también hay muchas de faena con varias máquinas a media distancia.

Sigue sin existir un dataset con este encuadre, y la conclusión sobre la fuente principal
no cambia. Pero el preentrenamiento no necesita el mismo dominio: sirve si le enseña al
modelo la forma de un bulldozer, que es la clase con 33 instancias propias. La pregunta es
empírica, así que queda como una tercera variante de la validación cruzada del ADR 0009:

- `scripts/prepare_pretrain.py` junta los exports YOLO de Roboflow y traduce las clases a
  `caex` y `bulldozer` (`bermguard/training/pretrain.py`). Las clases sin traducción, como
  `excavator` o `drill`, se descartan: forzarlas a una de las dos enseñaría una confusión.
- `--require bulldozer` conserva sólo las imágenes con al menos un bulldozer: las demás no
  aportan a la clase que falta. En las imágenes que quedan, los camiones (`dumper`,
  `Dumb_truck`) se etiquetan como `caex`; sin etiqueta, el modelo aprendería que un camión
  es fondo.
- `scripts/cross_validate.py --variants pretrain` preentrena una sola vez y afina cada fold
  con la misma partición e hiperparámetros que `base`. Lo único que cambia es el punto de
  partida, así que la diferencia con `base` mide el preentrenamiento. Como ninguna imagen
  pública viene de estos videos, no hay fuga entre folds.

Los pasos de esta revisión quedaron desactualizados: Mines no tiene zip de descarga, la
carpeta de lpoe2 es `data/externos/bulldozer/` y el comando de preparación lleva
`--exclude` y `--repeat`. Los pasos vigentes están en la revisión siguiente y en
[`data/README.md`](../../data/README.md).

Un riesgo a mirar en el resultado: `bulldozer-lpoe2` tiene 20 veces más imágenes que
`mines`, así que el preentrenamiento queda dominado por él. La revisión siguiente lo
corrige en parte con `--repeat mines=5`. Si la variante no mejora, el siguiente intento es
preentrenar sólo con `mines`.

Los datasets descargados quedan fuera del repo (`data/externos/`, `data/pretrain/`).

## Revisión (2026-10-20): cómo se importó Mines y cómo se curó el set

### Mines no tiene versión descargable

[`mines-project/mines-jo13g`](https://universe.roboflow.com/mines-project/mines-jo13g)
es público, pero el proyecto no tiene ninguna versión generada, así que no hay *Download
Dataset* ni zip. Las imágenes y las anotaciones igual son públicas: la página de cada
imagen las pide a un endpoint interno de Universe. Las junté desde el navegador (Chrome,
con la sesión abierta):

1. La lista de imágenes con bulldozer salió de `/datasets/roboql/publicDataset` con la
   consulta `class:dozer split:<split>`, una vez por partición.
2. La anotación de cada una, de
   `/query/source?dataset=Dih8GBz9EZBNdTGYVfm1&source=<id>`.
3. Todo quedó en `data/externos/mines/anotaciones_universe.json` (fuente, licencia,
   consulta y, por imagen, id, nombre, partición, URL y anotación).

`scripts/import_universe.py` convierte ese JSON en un export YOLO con el mismo formato que
uno de Roboflow, para que `prepare_pretrain.py` lo trate como a cualquier otra fuente.
Descarga cada imagen y **aplica la rotación EXIF a los píxeles** antes de guardarla:
Roboflow anota sobre la imagen ya rotada, y una foto de celular guardada con la rotación
sólo en el EXIF quedaría con las etiquetas giradas para cualquier lector que no la aplique.
El script falla si la orientación guardada no calza con la de la anotación.

```bash
python scripts/import_universe.py data/externos/mines/anotaciones_universe.json \
    --out data/externos/mines
```

Resultado, contado sobre el JSON y la carpeta generada: **68 imágenes** (50 train, 16
valid, 2 test) y **70 dozers**, todos como polígono, más 31 `dumper` y 9 instancias de
clases que se descartan (`drill`, `pc`, `camper`). La estimación anterior, ~75 dozers, era
de la vista de Universe.

`bulldozer-lpoe2` sí tiene versión: la 1, exportada en formato YOLO
(`Bulldozer.v1i.yolo26.zip`), 1.324 imágenes reescaladas a 512×512. Se descomprime en
`data/externos/bulldozer/`.

### Curación

Revisé a ojo las 68 imágenes de Mines y 121 de lpoe2 (muestra estratificada con semilla
42), más las 49 escenas de lpoe2 con varias máquinas, con las etiquetas dibujadas encima.
También busqué duplicados en todo lpoe2 por dHash (16×16, distancia ≤ 24 de 256 bits),
verificando una muestra a ojo. Lo que encontré:

- **Duplicados en lpoe2.** La lista versionada saca 215 imágenes, de 198 grupos; en 85 de
  ellos el grupo tenía copias en train y en val (contando `test` como val, como hace
  `prepare_pretrain.py`). Se conserva una por grupo, la de train si el grupo cruza
  particiones. Sin esto, la validación del preentrenamiento mide en parte memoria, y
  `best.pt` se elige con ella.
- **Etiquetas erróneas.** Un cargador de orugas (CAT 973D) etiquetado como `Bull_dozer`, y
  un camión de faena (CAT 793) etiquetado como `Loader`, que con `--require bulldozer`
  quedaría además como fondo sin su `caex`.
- **Juguetes y maquetas** sobre fondo blanco (dos imágenes).
- **Catálogo.** En la muestra, alrededor de 8–9 % son fotos de catálogo (máquina sola,
  fondo limpio). Se dejan: enseñan la forma de la máquina y no confunden la clase.
- **Mines:** un duplicado de la misma toma y cinco polígonos degenerados (dos o tres
  puntos casi superpuestos, un clic de más al anotar), que Ultralytics convierte en cajas
  de menos de un píxel.

Lo que se aplica:

- `data/pretrain_excluir.txt`, **versionado**: un nombre de archivo por línea con su
  motivo. `--exclude` lo lee. Son 220 entradas (219 de lpoe2 y 1 de Mines).
- Las instancias degeneradas (ancho o alto normalizado < 0,5 %) se quitan siempre
  (`descartar_degeneradas` en `bermguard/training/pretrain.py`).
- `--repeat mines=5`: cada imagen de **train** de Mines entra cinco veces (copias con
  sufijo `_r1`…`_r4`); val nunca se repite.

**Por qué ×5.** Mines es la única fuente fotografiada en una mina real, y sin repetir son
49 de 872 imágenes de train (5,6 %): el preentrenamiento casi no la vería. Con ×5 son 245
de 1.068 (23 %). Más repeticiones acercarían Mines a la mitad del set, pero con 49 imágenes
distintas eso es memorizarlas; ×5 es un punto intermedio, no un valor ajustado. Se repite
sólo train porque repetir val no agrega información y sesgaría la elección de `best.pt`.

### Set final

```bash
python scripts/prepare_pretrain.py \
    --source data/externos/mines --source data/externos/bulldozer \
    --map dozer=bulldozer bull_dozer=bulldozer dumper=caex dumb_truck=caex \
    --require bulldozer --exclude data/pretrain_excluir.txt --repeat mines=5
```

Según `data/pretrain/resumen.json`:

| Fuente | Train | Val | Excluidas | Degeneradas | `bulldozer` | `caex` |
|---|---|---|---|---|---|---|
| `mines` | 49 (245 con ×5) | 18 | 1 | 5 | 68 | 27 |
| `bulldozer` (lpoe2) | 823 | 282 | 219 | 0 | 1.122 | 46 |
| **Total** | **1.068** | **300** | 220 | 5 | | |

Las instancias por clase están contadas sin la repetición. `scripts/gpu_session.py --check`
rechaza un `data/pretrain/` generado sin `--exclude` o sin `--repeat`.
