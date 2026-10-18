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
- LOVO cuesta cuatro entrenamientos por variante, unos 15 a 20 minutos cada uno con la GPU
  libre. Con la GPU compartida con el escritorio (4 GB de VRAM), el entrenamiento se desborda
  a memoria compartida y cada época pasa de 18 s a varios minutos, así que hay que correrlo
  con la máquina libre: `python scripts/cross_validate.py --variants base copypaste --batch 2`.
- **El efecto del copy-paste y el umbral por clase aún no están medidos con LOVO.** Hasta
  entonces el pipeline sigue con `detector_v1` y un umbral único.
