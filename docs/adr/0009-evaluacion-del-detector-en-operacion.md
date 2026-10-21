# ADR 0009: Evaluar el detector en su punto de operación, con etiquetas auditadas y LOVO

- Estado: aceptada
- Fecha: 2026-10-17
- Implementación: `bermguard/benchmark/detection.py`, `scripts/evaluate_detector.py`,
  `scripts/apply_label_audit.py`, `scripts/cross_validate.py`,
  `bermguard/training/copy_paste.py`

## Contexto

Las métricas del detector del reporte anterior tenían tres problemas, y ninguno era del
modelo.

**1. No eran las del punto de operación.** Ultralytics reporta la precisión y el recall en
el umbral de confianza que maximiza el F1, no en el 0.25 con el que corre el pipeline
(`configs/method_1.yaml`). El reporte las presentó como si fueran del umbral de operación.
Medidas en 0.25 y con el NMS de operación (0.45), sobre las etiquetas originales de
`video_04`:

| CAEX | Reporte anterior | En el punto de operación |
|---|---|---|
| Precisión | 0.297 | **0.614** |
| Recall | 1.000 | **0.897** |

El detector no inventaba el 70 % de sus cajas, como decía el reporte: inventa el 39 %, y a
cambio pierde uno de cada diez camiones. Para seguridad el segundo error pesa más.

**2. Las etiquetas tenían bulldozers marcados como CAEX.** 191 de las 309 etiquetas de CAEX
seguían en formato caja: eran pre-anotaciones del modelo COCO, que entrega toda la
maquinaria como `truck`, y nunca pasaron por la corrección a polígono. Revisé las 191 en
hojas de contacto. **29 eran bulldozers**, con hoja, orugas y ripper visibles: 24 en
entrenamiento y 5 en validación. La guía de anotación pedía corregir la clase de los
bulldozers pre-anotados, y estas quedaron sin revisar.

La clase `bulldozer` no tenía 33 ejemplos de entrenamiento sino 57, y el modelo había
aprendido con 24 de ellos etiquetados como lo contrario. Con las etiquetas corregidas, el
mismo `detector_v1` mide:

| `detector_v1` en `video_04` | Etiquetas originales | Etiquetas auditadas |
|---|---|---|
| AP@0.5 `caex` | 0.734 | 0.668 |
| AP@0.5 `bulldozer` | 0.313 | **0.586** |
| Bulldozers en validación | 11 | 16 |

El modelo acertaba más bulldozers de los que se le reconocían: los predecía sobre máquinas
que la validación llamaba CAEX.

**3. Un solo video de validación y sin intervalos.** Con 11 bulldozers, un punto de AP no
dice nada sin saber cuánto se movería con otra muestra, y una sola partición es una
muestra de tamaño uno de "cómo rinde en un video nuevo".

## Decisión

1. **El detector se evalúa en el umbral y el NMS de operación**, con un evaluador propio
   (`bermguard/benchmark/detection.py`): emparejamiento estilo COCO, AP@0.5 con 101 puntos,
   barrido de umbral y desglose por condición de luz.
2. **Toda métrica lleva un intervalo de confianza del 95 %** por bootstrap de bloques de 5
   frames anotados consecutivos (unos 2.5 s de video). Remuestrear frames sueltos trataría
   como independientes imágenes casi iguales y daría intervalos demasiado angostos.
3. **Las etiquetas se corrigen con una auditoría registrada**
   (`data/auditoria_etiquetas.json`). Cada cambio guarda la línea original y el script solo
   la reescribe si sigue igual. Cuatro casos quedan para revisión manual: una caja fusionada
   y tres máquinas ambiguas, dos de ellas híbridas inventadas por el generador de video.
4. **Las mejoras del detector se comparan con validación cruzada dejando un video fuera**
   (LOVO): cuatro entrenamientos, cada uno validado en el video que no vio.
5. **Primera mejora a evaluar así: copy-paste de bulldozers** con los polígonos del propio
   dominio, escalados por perspectiva y con el brillo ajustado al destino.

## Validación del evaluador

Contra la validación de Ultralytics sobre las etiquetas originales, con NMS 0.7 como usa
ella: bulldozer 0.302 contra 0.306, y CAEX 0.770 contra 0.812, dentro del intervalo
(0.62–0.94). La diferencia restante viene de los lotes rectangulares de Ultralytics. El
evaluador propio mide la ruta de inferencia que usa el pipeline.

## Alternativas descartadas

- **Seguir con las métricas de Ultralytics.** No dan el punto de operación ni intervalos.
- **Re-anotar todo desde cero.** Más caro y sin trazabilidad. La auditoría corrige lo que
  está mal y deja registrado qué cambió.
- **Partición aleatoria de frames.** Ya descartada en el ADR 0004: mide memoria.

## Consecuencias

- Las cifras de detección del reporte anterior quedan reemplazadas por las de
  `docs/experimentos/`. Las anteriores se conservan en
  `docs/experimentos/detector_v1_val_etiquetas_originales.json`.
- Las conclusiones del reporte sobre "sobre-detección" se moderan: el problema principal en
  operación no es la precisión del CAEX, sino que el bulldozer casi nunca supera el umbral.
- LOVO cuesta cuatro entrenamientos por variante. Con la GPU compartida con el escritorio
  (4 GB de VRAM), el entrenamiento se desborda a memoria compartida y cada época pasa de
  18 s a varios minutos, así que hay que correrlo con la máquina libre. La variante
  `pretrain` (ADR 0004, revisiones) agrega un preentrenamiento y cuatro folds más. El
  comando y la duración vigentes están en la revisión del 2026-10-20 y en
  [`docs/entrenamiento/sesion_gpu.md`](../entrenamiento/sesion_gpu.md).
- **El efecto del copy-paste y el umbral por clase aún no están medidos con LOVO.** Hasta
  entonces el pipeline sigue con `detector_v1` y un umbral único.

## Revisión (2026-10-20): protocolo de LOVO, regla de decisión y umbral fuera de fold

Antes de correr LOVO revisé el protocolo, y tenía una fuga. Esta revisión fija cómo se
entrena, cómo se decide y qué riesgos quedan declarados, **antes** de ver un resultado.
Implementación: `scripts/cross_validate.py`, `bermguard/benchmark/lovo.py`,
`scripts/summarize_lovo.py`, `bermguard/training/etiquetas.py` y `scripts/gpu_session.py`.

### Épocas fijas y `last.pt`

En cada fold la partición `val` es el video excluido, que es también el de evaluación. Con
el protocolo anterior (80 épocas, paciencia 20, `best.pt`), la época del checkpoint y el
corte temprano se elegían mirando ese video. Eso infla la métrica de todas las variantes,
y no por igual: favorece a la que tenga la curva de validación más ruidosa, porque tiene
más picos entre los que elegir.

Ahora cada fold entrena **60 épocas fijas, sin paciencia y sin validar en cada época**
(`--patience 0 --no-val`, nuevo en `train_detector.py`), y se evalúa `last.pt`. Elegí 60
porque `detector_v1` seguía mejorando en la época 40 de 42. Es un valor fijado de
antemano, no ajustado por fold.

El preentrenamiento sí usa `best.pt` y paciencia: su validación son imágenes públicas, no
los videos del proyecto.

### Etiquetas homogeneizadas

Las etiquetas de 36 de los 51 frames de `video_01` mezclan cajas (pre-anotaciones sin
corregir) y polígonos. Ultralytics 8.4 trata un archivo así como corrupto y descarta la
imagen entera. Con el protocolo anterior, `base` habría entrenado sin esos frames y
`copypaste` los habría visto a través de sus copias aumentadas, lo que le habría atribuido
a la augmentation datos que `base` nunca tuvo. Ahora, al armar cada fold, un archivo mixto
pasa sus cajas a polígonos de 4 vértices (`homogeneizar`). Las etiquetas versionadas no se
tocan, porque la auditoría identifica cada corrección por su línea original. Con el
chequeo de etiquetas de Ultralytics: 36 frames descartados antes, 0 después en todos los
folds y variantes. `detector_v1` entrenó sin esos 36 frames (`docs/resultados.md` §1.2).

### Regla de decisión

`bermguard/benchmark/lovo.py` agrega los JSON por fold y decide con una regla escrita
antes de los datos:

- **Por métrica**, diferencia pareada por fold, `variante − base`. Un fold mejora si
  Δ > 0,01. La métrica `mejora` si mejora en al menos 3 de 4 folds **y** la media de Δ es
  ≥ 0,02; `empeora` en el caso simétrico; si no, `no distinguible del ruido`.
- **Candidata**: una variante con los cuatro folds de `base`, con veredicto `mejora` en el
  AP@0.5 del bulldozer, y cuyo AP@0.5 del CAEX no cae más de 0,02 en media ni más de 0,05
  en ningún fold.
- **Ganadora**: la candidata con mayor mejora media del bulldozer. Si dos quedan a menos
  de 0,01, gana la más simple (`copypaste`, sólo datos, antes que `pretrain`, un
  entrenamiento extra). Si no hay candidatas, gana `base` por parsimonia.

La decisión usa AP@0.5, que no depende del umbral. Con 4 folds ningún resultado es
significativo al 5 % (4 de 4 a favor da p = 0,125 en la prueba de signos bilateral, que se
reporta igual): es un criterio de consistencia entre videos y de tamaño mínimo, no una
prueba. El evaluador ahora también escribe AP@0.5:0.95, que se reporta pero no decide.

### Umbral fuera de fold

El umbral del bulldozer se elige sobre la variante ganadora. Para cada fold *k*, se suman
los conteos (`tp`, `fp`, objetos) de los barridos de los **otros tres** folds, se elige el
umbral y se mide en *k*. El barrido de *k* nunca entra en la elección del umbral de *k*.
El criterio por defecto es `precision_min` con 0,5: el umbral de mayor recall cuya
precisión agregada es al menos 0,5. Para el bulldozer manda el recall (una máquina no
detectada es una máquina sin vigilar), y el piso acota las falsas alarmas por frame, que
el tracker filtra después con `min_hits`.

El umbral de producción se elige con los cuatro folds juntos. Su rendimiento esperado es
el fuera de fold, no el que da aplicado a esos mismos folds. El barrido ahora baja hasta
0,01 (antes empezaba en 0,05), porque `detector_v1` dejó todas las confianzas del
bulldozer bajo 0,1; si el umbral elegido cae en el borde inferior de la grilla, el
resumen lo marca.

### Riesgos declarados

- **Una semilla por fold.** La varianza entre folds mezcla la diferencia entre videos con
  la de la inicialización y el orden de los lotes. Una diferencia pequeña entre variantes
  puede ser de semilla. Repetir con más semillas multiplica el costo, que ya es de unas
  10 horas.
- **Fuga indirecta, de segundo orden, en el umbral.** El umbral de *k* sale de los modelos
  de los otros folds, y esos modelos entrenaron con el video *k*. La distribución de
  confianzas que se usa para elegir el umbral está influida por *k*, aunque ninguna
  predicción sobre *k* entre en la elección. Además, algunas decisiones del protocolo (60
  épocas, el barrido desde 0,01) se tomaron mirando cómo se comportó `detector_v1` en
  `video_04`.
- **Sin bootstrap pareado por fold.** Cada JSON trae intervalos por bootstrap de bloques
  para cada variante, pero la diferencia entre variantes no lleva intervalo propio: el
  veredicto descansa en la consistencia entre folds, no en un intervalo de Δ dentro de
  cada video.
- Cuatro videos generados con IA no son una muestra de minas (ADR 0004).
- **Pasos de optimización desiguales.** Faltaba en esta lista: `copypaste` hace el doble de
  pasos que `base` con las mismas épocas. Se agregó con su control en la revisión del
  2026-10-21.

### Consecuencias

- Un fold terminado deja `resumen_entrenamiento.json`. Volver a lanzar el mismo comando
  salta el preentrenamiento y los folds terminados, así que la corrida se puede cortar con
  Ctrl+C y retomar en otra ventana de GPU.
- `scripts/gpu_session.py` revisa antes de empezar el dataset, el set de preentrenamiento,
  los pesos COCO locales, la VRAM libre, los procesos pesados en la GPU y 5 GB de disco
  libre, y después corre LOVO y el resumen.
- Duración estimada en la GPU de desarrollo, extrapolada de una prueba de humo en CPU de
  toda la cadena: unas 10 h ± 25 % (sección "Durante" de
  [`sesion_gpu.md`](../entrenamiento/sesion_gpu.md)). No es un número medido. Si
  `copypaste` resulta candidata, el control `base2x` suma unas 3,4 h (revisión del
  2026-10-21).
- `scripts/summarize_lovo.py` escribe `docs/experimentos/lovo/resumen.json` y
  `resumen.md`. **Todavía no hay resultados**: esta revisión fija el método, no el
  ganador.

## Revisión (2026-10-21): control de cómputo para `copypaste`

Escrita antes de entrenar ningún fold, igual que la anterior: fija qué se mide y cómo se
decide **antes** de ver un resultado. Implementación: `bermguard/benchmark/lovo.py`
(`CONTROL`), `scripts/cross_validate.py` (`--variants base2x`) y `scripts/gpu_session.py`.

### El problema

`copypaste` escribe un frame aumentado por cada frame de entrenamiento, así que su fold
tiene el doble de imágenes que el de `base`. Armando los folds con
`cross_validate.preparar_fold`:

| Fold (video fuera) | Imágenes de train, `base` | Imágenes de train, `copypaste` |
|---|---|---|
| `video_01` | 120 | 240 |
| `video_02`, `video_03`, `video_04` | 131 | 262 |

Con 60 épocas fijas y batch 4, `copypaste` hace unas 66 iteraciones por época y `base`
unas 33: el doble de pasos de optimización. `detector_v1` seguía mejorando en la época 40
de 42, así que más pasos solos pueden mejorar a `base`. Una victoria de
`copypaste` podría ser de cómputo y no de augmentation. Este riesgo no estaba entre los
declarados en la revisión del 2026-10-20.

### Decisión

- **Variante de control `base2x`**: `base` con el doble de épocas (120), es decir, los
  mismos pasos que `copypaste`. Misma partición y semilla, y también se evalúa `last.pt`.
- **`copypaste` sólo puede ser candidata si además da `mejora` en el AP@0.5 del bulldozer
  contra `base2x`**, con la misma regla pareada por fold (Δ > 0,01 en al menos 3 de 4 folds y
  media ≥ 0,02). El resto de la regla no cambia: se sigue comparando contra `base`.
- **Sin los cuatro folds de `base2x`**, la decisión queda `pendiente_control`: `copypaste`
  no es candidata y el motivo de la decisión dice que es provisoria.
- **`base2x` nunca es candidata.** Es un control, no una variante que se pueda adoptar.
  Si `base2x` da `mejora` contra `base`, eso dice que 60 épocas eran pocas: la decisión
  marca `epocas_dobles` y la ganadora, sea cual sea, se reentrena para v2 con 120 épocas.
- El resumen muestra la diferencia pareada de `copypaste` contra `base2x` (media, mínimo,
  máximo y folds), no sólo el veredicto, porque es la evidencia que decide.
- La igualación de pasos es aproximada: Ultralytics apaga el mosaico en las últimas 10
  épocas fijas (`close_mosaic`), que en `base2x` son 10 de 120 y en `copypaste` 10 de 60.
- `pretrain` no se compara contra `base2x`: entrena con las mismas imágenes y épocas que
  `base`, así que su diferencia no es de pasos.

### Por qué un control condicional

Entrenar `base2x` siempre cuesta cuatro entrenamientos de 120 épocas, unas 3,4 h con la
estimación de `base` (~1,7 h para 4 folds × 60 épocas). Sólo hace falta si `copypaste`
ganaría: si no es candidata contra `base`, el control no cambia la decisión.
`scripts/gpu_session.py` corre el resumen, y si queda `pendiente_control` entrena
`base2x` y vuelve a resumir, en la misma sesión.

### Consecuencias

- La sesión de GPU dura unas 10 h si `copypaste` no es candidata, y unas 3,4 h más si lo
  es. Las dos son estimaciones, no mediciones
  ([`sesion_gpu.md`](../entrenamiento/sesion_gpu.md)).
- Una semilla por fold sigue siendo un riesgo: el control iguala los pasos, no el ruido
  de inicialización.
- **Todavía no hay resultados.** Esta revisión cambia la regla antes de los datos.
