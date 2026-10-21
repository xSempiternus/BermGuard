# Resultados

Resultados del sistema, separados por objetivo:

1. **Detectar CAEX y bulldozers y medir la proximidad entre ellos.**
2. **Medir la altura del pretil.**

Para cada uno presento los datos, las métricas y lo que muestran. Al final están el
análisis, la conclusión de cada objetivo y lo que queda por hacer. La comparación entre
métodos (zero-shot contra fine-tuning, camino óptimo contra argmax, CPU contra GPU) está en
[`reporte_benchmark.md`](../reporte_benchmark.md).

> **Revisión de octubre de 2026.** Medir mejor cambió varias cifras de la versión anterior
> de este documento: las del detector no eran las del punto de operación, 29 bulldozers
> estaban etiquetados como CAEX, y la validación del pretil aceptaba ruido. Las cifras
> anteriores se conservan en `docs/experimentos/` y cada corrección tiene su ADR (0008, 0009
> y 0010). Además, el material resultó ser generado con IA (`docs/analisis_material.md`,
> sección 6), lo que cambia cómo leer la altura métrica.

```bash
python scripts/apply_label_audit.py --check
python scripts/evaluate_detector.py --weights weights/detector_v1.pt \
    --images data/dataset/images/val --labels data/dataset/labels/val \
    --out docs/experimentos/detector_v1_val.json
python scripts/benchmark_berm_synthetic.py --out docs/experimentos/pretil_sintetico.json
python main.py --input data/raw --output output --method all
```

---

## 1. Detección y proximidad de CAEX y bulldozer

### 1.1 Datos

171 frames anotados a mano, estratificados por condición de luz y separados por video. La
tabla ya incluye la auditoría de etiquetas (ADR 0009): 29 bulldozers que estaban marcados
como `caex` pasaron a su clase, y el 2026-10-20 se agregó un bulldozer que faltaba en
`video_03_f00126` (lista `agregados` de la auditoría).

| Partición | Imágenes | `caex` | `bulldozer` |
|---|---|---|---|
| train (`video_01`, `02`, `03`) | 131 | 246 | 58 |
| val (`video_04` completo) | 40 | 34 | 16 |
| **Total** | **171** | **280** | **74** |

Antes de la auditoría la proporción era de 7 CAEX por bulldozer; ahora es de 4. Separé por
video y no al azar porque dos frames cercanos son casi la misma imagen, y con una partición
aleatoria la validación premiaría memorizar.

### 1.2 Entrenamiento

`detector_v1` se entrenó desde `yolo11s.pt` (COCO), con un máximo de 80 épocas, `batch=8`,
`imgsz=640`, precisión mixta y semilla 0, **con las etiquetas anteriores a la auditoría**.
Se detuvo por paciencia en la época 42 y la mejor fue la 22.

**`detector_v1` entrenó sin 36 de los 51 frames de `video_01`.** Sus etiquetas mezclan
cajas y polígonos en el mismo archivo, y Ultralytics 8.4 descarta esos frames enteros
(`bermguard/training/etiquetas.py`). Vio 95 de los 131 frames de train. La validación no
se afecta: ningún frame de `video_04` mezcla formatos, y las cifras de esta sección siguen
siendo las medidas. LOVO ya homogeneiza las etiquetas de cada fold. Reentrenarlo con las etiquetas
corregidas, y medir el copy-paste de bulldozers, es el siguiente paso (sección 4).

**Eso explica buena parte de la debilidad del bulldozer.** Con las etiquetas auditadas,
train tiene 58 bulldozers: 27 en `video_01`, 14 en `video_02` y 17 en `video_03`. 25 de
los 27 de `video_01` están en los frames descartados, así que los frames que vio
`detector_v1` contienen 33 de los 58 (57 %). Y los vio con las etiquetas anteriores a la
auditoría: de esos 33, 24 estaban marcados como `caex` y uno no estaba etiquetado (el
agregado del 2026-10-20). Aprendió la clase con 8 cajas de bulldozer. Es coherente con su
AP@0.5 de 0.306 en la validación de Ultralytics, sobre las etiquetas originales
(`docs/entrenamiento/metricas_por_clase.json`), y con su matriz de confusión normalizada
(`docs/entrenamiento/confusion_matrix_normalized.png`): no predice `bulldozer` nunca; de
los bulldozers de validación, el 45 % sale como `caex` y el 55 % como fondo. Ninguna cifra
de esta sección cambia por esto.

Los bulldozers por video son 27 (`video_01`), 14 (`video_02`), 17 (`video_03`) y 16
(`video_04`). En LOVO cada fold se valida sobre 14 a 27 bulldozers, así que el AP del
bulldozer de un fold se mueve con pocas máquinas.

### 1.3 Métricas del detector

Sobre `video_04`, que el modelo no vio al entrenar, con las etiquetas auditadas y **en el
punto de operación del pipeline**: confianza 0.25 y NMS 0.45. Entre corchetes, el intervalo
de confianza del 95 % por bootstrap de bloques de 5 frames anotados consecutivos
(`docs/experimentos/detector_v1_val.json`).

| Clase | AP@0.5 | Precisión | Recall | Objetos |
|---|---|---|---|---|
| `caex` | 0.668 [0.46–0.89] | 0.526 [0.40–0.70] | 0.882 [0.77–0.97] | 34 |
| `bulldozer` | 0.586 [0.27–0.96] | — | **0.000** | 16 |

El bulldozer no tiene precisión porque al umbral 0.25 el modelo no emite ninguna detección
de esa clase: todas sus predicciones quedan bajo 0.1.

#### Qué cambió respecto de la versión anterior

| | Versión anterior | Ahora | Por qué |
|---|---|---|---|
| Precisión `caex` | 0.297 | 0.526 | Ultralytics reporta P y R en el umbral de máximo F1, no en el de operación |
| Recall `caex` | 1.000 | 0.882 | Ídem |
| AP `bulldozer` | 0.306 | 0.586 | 5 bulldozers de validación estaban etiquetados como CAEX |
| AP `caex` | 0.812 | 0.668 | Lo mismo, y el NMS de operación (0.45) en vez del de validación (0.7) |

Con las etiquetas originales, el mismo evaluador da AP 0.734 para `caex` y 0.313 para
`bulldozer`, que es lo que reportaba Ultralytics (0.812 y 0.306) salvo por el NMS y por
cómo arma los lotes (ADR 0009).

#### Qué mide el AP y por qué lo uso

Para cada clase, las detecciones se ordenan por confianza. Una detección cuenta como
correcta si su caja se superpone al menos un 50 % con una caja anotada (IoU ≥ 0.5). Bajando
el umbral paso a paso se calculan la precisión y el recall, y eso dibuja la curva
precisión-recall. El AP es el área bajo esa curva (1 es perfecto). Lo reporto por clase
porque con clases desbalanceadas el promedio lo domina la grande.

El AP no dice cómo se comporta el modelo en un umbral concreto, así que lo acompaño con la
precisión y el recall **en el umbral con el que corre el sistema**, que además separan los
dos tipos de error. En seguridad no pesan igual: un error de precisión es una falsa alarma;
uno de recall es un equipo sin vigilar.

### 1.4 Umbral y condición de luz

**El umbral global borra al bulldozer.** Barrido de umbral en `video_04`:

| Umbral | Precisión `bulldozer` | Recall `bulldozer` | Precisión `caex` | Recall `caex` |
|---|---|---|---|---|
| 0.05 | 0.73 | 0.50 | 0.26 | 0.94 |
| 0.10 | 1.00 | 0.06 | 0.33 | 0.91 |
| 0.25 | — | 0.00 | 0.53 | 0.88 |

Un umbral propio para el bulldozer (0.05) recuperaría la mitad de los bulldozers con 73 %
de precisión. El pipeline ya lo soporta (`detector.class_confidence`), pero **no lo activé**:
elegir el umbral mirando `video_04` y reportar sobre `video_04` sería ajustar sobre la
validación. El valor se tiene que elegir con la validación cruzada de los otros tres videos.

**Por condición de luz** (pendiente desde el ADR 0005):

| Condición | Imágenes | AP `caex` | Objetos `caex` |
|---|---|---|---|
| día | 13 | 0.29 | 9 |
| crepúsculo | 4 | 1.00 | 2 |
| noche | 23 | 0.82 | 23 |

El CAEX rinde peor de día que de noche, contra lo que esperaba. Con 9 objetos diurnos el
intervalo es muy ancho, así que no lo trato como un hallazgo sino como algo a revisar con
más datos.

### 1.5 Separación de máquinas

La proximidad necesita que cada máquina tenga su propia caja. En el frame 120 de
`video_02`, con un CAEX y un bulldozer juntos:

| Modelo | Detecciones | Cajas |
|---|---|---|
| `yolo11s` COCO | 1 | `truck` 0.74, x=[138,759], **622 px, las dos máquinas** |
| `detector_v1` | 2 | 0.48, x=[480,765], 285 px, **el bulldozer**<br>0.43, x=[65,614], 549 px, **el CAEX** |

Ver `docs/showcase/img/separacion_coco_vs_propio.png`.

### 1.6 Proximidad

Cada equipo queda en nivel seguro, precaución (a menos de 20 m de otro) o crítico (a menos
de 10 m). Para que el nivel no parpadee, sube después de 3 frames seguidos y baja después de
10. Una alerta es una **subida de nivel**, no un frame en riesgo.

| Video | Alertas | Distancia mínima | Mediana |
|---|---|---|---|
| `video_01` | 39 | 0.6 m | 17.4 m |
| `video_02` | 12 | 2.9 m | 23.7 m |
| `video_03` | 16 | 4.3 m | 19.4 m |
| `video_04` | 13 | 0.9 m | 19.3 m |

El tracker confirma un track solo después de 3 detecciones seguidas y suprime las cajas
duplicadas por contención (intersección sobre el área de la menor), lo que bajó las
alertas de 43 a 39 en `video_01`. Los tests fijan la histéresis (`tests/test_proximidad.py`).

---

## 2. Altura del pretil

### 2.1 Cómo se mide

El Método 1 (ADR 0006) encuentra la cresta y la base del pretil como un camino óptimo, a
media resolución (ADR 0010). La altura sale de:

```
H = (y_base − y_cresta) · h / (y_base − y_horizonte)
```

donde `h` es la altura de la cámara, estimada con el ancho nominal de un CAEX detectado. La
focal se cancela, y hay tests que lo verifican con casos calculados a mano
(`tests/test_metrologia.py`).

### 2.2 Exactitud, por primera vez

Sobre escenas sintéticas con la cresta conocida (ADR 0008), fracción de columnas a 5 px o
menos de la cresta verdadera, 20 escenas por punto:

| Contraste de la cresta | Huellas | Camino óptimo | Argmax + filtro |
|---|---|---|---|
| 18 | 0 | 100 % | 100 % |
| 18 | 6 | **90 %** | 75 % |
| 18 | 12 | **60 %** | 49 % |
| 12 | 4 | **71 %** | 59 % |
| 8 | 6 | 17 % | 21 % |

El camino óptimo gana cuando la cresta es visible. Cuando es más débil que las huellas no
gana ninguno. Es exactitud de la formulación en condiciones controladas, no del sistema
sobre el material.

### 2.3 Estabilidad en el material

Jitter por columna: el cambio mediano de la cresta, columna a columna, entre dos frames
seguidos de la misma toma, contando solo las columnas medidas en ambos (ADR 0008).

| Video | Cobertura M1 | Jitter M1 | Jitter M2 |
|---|---|---|---|
| `video_01` (1080p, 3 tomas) | 0.69 | **5.71 px** | 6.75 px |
| `video_02` | 0.52 | **0.60 px** | 0.63 px |
| `video_03` | 0.49 | **0.95 px** | 1.51 px |
| `video_04` | 0.56 | **2.14 px** | 2.91 px |

La cobertura bajó de ~0.9 a 0.5–0.7 respecto de la versión anterior porque ahora la
validación deja de contar ruido como pretil. En el frame 130 de `video_04`, donde la curva
antes seguía las huellas de neumático con 85 % de cobertura, ahora el método se abstiene.

### 2.4 Variación con la luz

Un pretil no cambia de altura al oscurecer, así que la diferencia entre condiciones es error
de medición y sirve como piso del error (Método 1, los cuatro videos):

| Condición | Altura mediana | Frames |
|---|---|---|
| día | 0.35 m | 429 |
| crepúsculo | 0.37 m | 106 |
| noche | 0.44 m | 479 |

Son ±11 % en torno a la mediana global de 0.41 m.

---

## 3. Análisis y conclusiones

### 3.1 Detección y proximidad

**El problema principal no era la precisión del CAEX, sino el bulldozer.** La versión
anterior decía que el detector "inventaba el 70 % de sus cajas"; en el umbral de operación
inventa el 47 % y pierde el 12 % de los camiones. El problema serio es otro: el bulldozer
nunca supera el umbral, así que en operación el sistema no ve bulldozers como tales.

**La causa era sobre todo de datos, y una parte estaba en las etiquetas.** El modelo
aprendió con 24 bulldozers etiquetados como CAEX, la clase contraria. Con la etiqueta
corregida en validación su AP sube de 0.31 a 0.59: el modelo distinguía más de lo que se le
reconocía, pero con confianzas bajas, que es lo esperable cuando el entrenamiento le dice
dos cosas opuestas.

**Conclusión.** El detector cumple lo que la proximidad necesita: encuentra el 88 % de los
CAEX y separa las máquinas vecinas. No sirve todavía para decir qué tipo de equipo es. Los
dos pasos siguientes son de bajo costo y ya están implementados: reentrenar con las
etiquetas corregidas y elegir un umbral por clase con validación cruzada.

### 3.2 Altura del pretil

**Ahora mido exactitud, aunque sea en sintético.** El camino óptimo supera al argmax por
hasta 15 puntos cuando la cresta es visible, y los dos fallan cuando no lo es.

**En el material el camino óptimo es más estable que el argmax en los cuatro videos**, con
una métrica que no confunde cobertura con movimiento. La conclusión del ADR 0006 se
sostiene; la cifra anterior ("44 % menos jitter") no, porque estaba calculada con la métrica
defectuosa.

**La altura absoluta no tiene un valor verdadero contra el cual comparar.** El material es
generado con IA, así que no hay una geometría métrica real que recuperar. El sesgo de ~4×
frente a la normativa puede venir del método o de la escena. Lo que sí se puede usar es la
altura **relativa**: el error de escala es el mismo para todas las mediciones de una toma y
se cancela.

**Conclusión.** El método del pretil es más defendible que antes: se abstiene cuando no ve
una cresta, mide exactitud en condiciones controladas y cuesta 3.6 a 4.6 veces menos. Para
validarlo sobre material real hace falta video real y el perfil anotado en unos 30 frames.

---

## 4. Qué queda por hacer

### Detección y proximidad

1. **Reentrenar con las etiquetas auditadas y medir con validación cruzada el copy-paste y el
   preentrenamiento con datos públicos.** Todo está implementado
   (`scripts/gpu_session.py`, ADR 0004 y 0009, guía en
   [`docs/entrenamiento/sesion_gpu.md`](entrenamiento/sesion_gpu.md)); falta correrlo con
   la GPU libre, unas 10 h estimadas, más unas 3,4 h si `copypaste` gana y hay que
   entrenar su control de cómputo (`base2x`, ADR 0009, revisión del 2026-10-21).
2. **Elegir el umbral del bulldozer** con las predicciones fuera de muestra de esa
   validación cruzada, y activarlo en `configs/`.
3. **Cerrar los 4 casos pendientes de la auditoría** (`data/auditoria_etiquetas.json`).
4. **Evaluar las alertas una por una** en un video: cuántas son verdaderas. Es la métrica
   que le importa a un operador.
5. **Justificar los umbrales de 20 y 10 m** contra la normativa y la distancia de frenado.

### Altura del pretil

1. **Validar con video real**: el sintético mide la formulación, y el material es generado.
2. **Anotar el perfil del pretil en ~30 frames** reales, para medir exactitud en el dominio.
3. **Reportar la altura relativa al radio de rueda**, que cancela la escala.
4. **Un segmentador neural** (SAM2 guiado por el camino óptimo) contra el geométrico, sobre
   el mismo benchmark sintético y el mismo ground truth.

### Generales

- **Exportar el detector a ONNX/TensorRT** y medir la ganancia en FP16: con el pretil a
  media resolución, el detector pasó a ser la etapa más cara.
- **Un segundo anotador** para medir el acuerdo entre anotadores.
