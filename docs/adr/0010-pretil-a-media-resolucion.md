# ADR 0010: Segmentar el pretil a media resolución

- Estado: aceptada
- Fecha: 2026-10-18
- Implementación: `bermguard/vision/berm/scaled.py`, `bermguard/pipeline/factory.py`,
  `berm.analysis_scale` en `configs/`

## Contexto

Con GPU, el pretil era la etapa más cara del pipeline: 36 ms por frame a 720p y 74 ms a
1080p en el contenedor, contra 16 y 29 ms del detector (reporte, sección 5). La GPU no lo
acelera porque corre en NumPy, y la mitad de ese costo es el camino óptimo, que recorre las
columnas una por una desde Python.

El pretil se busca sobre un gradiente suavizado con sigma 6. A esa escala, la resolución
completa no aporta nada que el método use.

## Decisión

**El pretil se segmenta a la mitad de la resolución del frame** (`analysis_scale: 0.5`) y el
perfil se devuelve a escala completa. Los parámetros en píxeles (desenfoque, ventana de
suavizado y margen de las cajas) se escalan en la misma proporción, de modo que se busca la
misma estructura física. El salto máximo entre columnas no se escala, porque es una
pendiente: filas y columnas se reducen por igual.

Se implementa como un envoltorio (`ScaledBermSegmenter`) que vale para los dos métodos del
benchmark, sin tocar ninguno.

## Mediciones

**Escenas sintéticas 720p** (cresta conocida, 0, 6 y 10 huellas, 8 escenas cada una):

| | Columnas a ≤5 px | ms por frame |
|---|---|---|
| Resolución completa | 97.3 % | 56 |
| Media resolución, parámetros escalados | **97.9 %** | **16.5** |

**Material real**, mismo detector y misma sesión, entre 3.6 y 4.6 veces menos tiempo en la etapa
(`docs/experimentos/pretil_media_resolucion.json`):

| Video | Etapa del pretil, completa → media | Jitter por columna M1, completa → media | M2 a media |
|---|---|---|---|
| `video_01` (1080p) | 110 → 29.0 ms | 5.52 → 5.71 px | 6.75 px |
| `video_02` | 49.6 → 13.7 ms | 0.49 → 0.60 px | 0.63 px |
| `video_03` | 55.4 → 15.1 ms | 0.89 → 0.95 px | 1.51 px |
| `video_04` | 62.3 → 13.4 ms | 2.10 → 2.14 px | 2.91 px |

La cobertura se mantiene (de 0.45–0.65 a 0.49–0.69) y el camino óptimo sigue siendo más
estable que el argmax en los cuatro videos.

La primera medición de esta tabla salió peor (jitter hasta 28 % mayor) por un bug del
envoltorio: al volver a escala completa rellenaba los huecos con cero antes de interpolar,
y las columnas vecinas a un hueco quedaban a mitad de camino hacia la fila 0. En el OSD se
veían como espigas verticales. Hay un test que lo cubre.

Los tiempos absolutos de esta sesión no son comparables con los del reporte: la GPU estaba
compartida con otras aplicaciones de escritorio. La razón entre las dos variantes sí lo es,
porque corrieron una detrás de la otra en las mismas condiciones.

## Alternativas descartadas

- **Pasar la programación dinámica a la GPU.** Es la opción con más techo, pero exige
  reescribir el método y mantener dos implementaciones. La media resolución da 3.6–4.6× con un
  envoltorio de 60 líneas.
- **Ajustar el desenfoque para el sintético.** Con sigma 4 a media resolución, el sintético
  llega al 100 %, pero eso sería ajustar al generador. Se usan los parámetros escalados en
  proporción, que conservan lo que ya se validó sobre el material.
- **Un cuarto de resolución.** En 720p deja la banda de búsqueda en unas 40 filas; el
  margen de las cajas y la ventana de suavizado quedan en pocos píxeles. No lo medí.

## Consecuencias

- El pretil deja de ser el cuello de botella con GPU: pasa a costar menos que la detección.
- El jitter por columna sube entre 2 y 22 %; en 720p sigue bajo un píxel.
- `analysis_scale: 1.0` reproduce exactamente el comportamiento anterior, y hay un test
  que lo verifica.
