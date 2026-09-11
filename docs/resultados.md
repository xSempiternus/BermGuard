# Resultados

Resultados del sistema final, separados por objetivo:

1. **Detectar CAEX y bulldozers y medir la proximidad entre ellos.**
2. **Medir la altura del pretil.**

Para cada uno presento los datos, las métricas y lo que muestran. Al final están el
análisis, la conclusión de cada objetivo y lo que queda por hacer. La comparación entre
métodos (zero-shot contra fine-tuning, camino óptimo contra argmax, CPU contra GPU) está en
[`reporte_benchmark.md`](../reporte_benchmark.md).

```bash
python scripts/split_dataset.py data/export --out data/dataset --val video_04
python scripts/train_detector.py --data data/dataset/data.yaml --epochs 80
python scripts/eval_detector.py
python main.py --input data/raw --output output --method all
```

---

## 1. Detección y proximidad de CAEX y bulldozer

### 1.1 Datos

171 frames anotados a mano, estratificados por condición de luz y separados por video.

| Partición | Imágenes | `caex` | `bulldozer` |
|---|---|---|---|
| train (`video_01`, `02`, `03`) | 131 | 270 | 33 |
| val (`video_04` completo) | 40 | 39 | 11 |
| **Total** | **171** | **309** | **44** |

Hay 7 CAEX por cada bulldozer. Viene del material: en `video_01` hay cuatro o cinco CAEX a
la vez y un solo bulldozer, y en los otros clips el bulldozer se ve más chico, tapado por el
polvo o fuera de cuadro.

Separé por video y no al azar porque dos frames cercanos son casi la misma imagen, y con una
partición aleatoria la validación premiaría memorizar. Dejé `video_04` para validación y no
`video_03` porque este solo tiene 4 bulldozers: con 4 objetos, fallar uno mueve el resultado
un 25 %.

### 1.2 Entrenamiento

Desde `yolo11s.pt` (COCO), con un máximo de 80 épocas, `batch=8`, `imgsz=640`, precisión
mixta y semilla 0. Se detuvo por paciencia en la época 42 y la mejor fue la 22. Las curvas
están en `docs/entrenamiento/` y los pesos en `weights/detector_v1.pt`.

### 1.3 Métricas del detector

Sobre `video_04`, que el modelo no vio al entrenar:

| Clase | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---|---|---|---|
| `caex` | 0.812 | 0.585 | 0.297 | **1.000** |
| `bulldozer` | **0.306** | 0.188 | 0.464 | 0.455 |
| global | 0.559 | 0.387 | 0.381 | 0.727 |

#### Qué mide el mAP y por qué lo uso

Para cada clase, las detecciones se ordenan por confianza. Una detección cuenta como
correcta si su caja se superpone al menos un 50 % con una caja anotada (IoU ≥ 0.5). Bajando
el umbral de confianza paso a paso se calculan la precisión y el recall, y eso dibuja la
curva precisión-recall. El AP es el área bajo esa curva (1 es perfecto) y el mAP es el
promedio entre clases. El mAP@0.5 exige un 50 % de superposición; el mAP@0.5:0.95 promedia
exigencias de 50 % a 95 %, así que también mide qué tan ajustada queda la caja.

Es la métrica estándar en detección (COCO, Pascal VOC, Ultralytics) y resume en un número si
el modelo encuentra los objetos, si inventa y si ubica bien la caja, sin depender de un
umbral elegido a mano. Uso mAP@0.5 porque para alertar me importa encontrar la máquina, no
ajustar la caja al píxel, y lo reporto por clase porque con un desbalance de 7:1 el promedio
lo domina la clase grande.

El mAP no dice cómo se comporta el modelo en un umbral concreto, así que lo acompaño con la
precisión y el recall, que además separan los dos tipos de error. En seguridad no pesan
igual:

- **Precisión:** de lo que el modelo marcó como máquina, cuánto lo era. Un error aquí es una
  falsa alarma.
- **Recall:** de las máquinas reales, cuántas encontró. Un error aquí es un equipo sin
  vigilar.

La curva precisión-recall está en `docs/entrenamiento/BoxPR_curve.png` y la matriz de
confusión en `docs/entrenamiento/confusion_matrix_normalized.png`.

### 1.4 Separación de máquinas

La proximidad necesita que cada máquina tenga su propia caja. En el frame 120 de
`video_02`, con un CAEX y un bulldozer juntos:

| Modelo | Detecciones | Cajas |
|---|---|---|
| `yolo11s` COCO | 1 | `truck` 0.74, x=[138,759], **622 px, las dos máquinas** |
| `detector_v1` | 2 | `caex` 0.48, x=[480,765], 285 px, **el bulldozer**<br>`caex` 0.43, x=[65,614], 549 px, **el CAEX** |

### 1.5 Proximidad

Cada equipo queda en nivel seguro, precaución (a menos de 20 m de otro) o crítico (a menos
de 10 m). Para que el nivel no parpadee, sube después de 3 frames seguidos y baja después de
10. Una alerta es una **subida de nivel**, no un frame en riesgo: un equipo diez segundos en
zona crítica es un evento, no doscientas alertas.

Corrida de `output_ejemplo/`:

| Video | Alertas | Distancia mínima | Mediana |
|---|---|---|---|
| `video_01` | 39 | 0.6 m | 17.4 m |
| `video_02` | 12 | 2.9 m | 23.7 m |
| `video_03` | 16 | 4.3 m | 19.4 m |
| `video_04` | 13 | 0.9 m | 19.3 m |

El tracker tiene dos mecanismos contra las detecciones falsas:

| Mecanismo | Efecto medido |
|---|---|
| Confirmar un track solo después de 3 detecciones seguidas | Elimina falsos positivos aislados sobre polvo |
| Eliminar cajas duplicadas sobre la misma máquina | Alertas de 43 a 39 en `video_01`; distancia mínima de 0.6 a 2.9 m en `video_02` |

Para detectar duplicados uso la intersección sobre el área de la caja más chica y no el IoU:
dos cajas desplazadas sobre el mismo camión tienen un IoU de ~0.35, igual que dos equipos
reales, pero una queda casi contenida en la otra.

---

## 2. Altura del pretil

### 2.1 Cómo se mide

El Método 1 (ADR 0006) encuentra la cresta y la base del pretil en la imagen. La altura sale
de:

```
H = (y_base − y_cresta) · h / (y_base − y_horizonte)
```

donde `h` es la altura de la cámara, que se estima con el ancho nominal de un CAEX
detectado. La focal se cancela, así que el resultado no depende del campo de visión
asumido. Cada medición lleva una banda de incertidumbre: la dispersión entre los equipos
usados como referencia, con un mínimo de ±20 % cuando hay uno solo.

### 2.2 Métricas

No tengo alturas reales medidas en terreno, así que no puedo calcular un error directo (MAE
o R²). Uso tres medidas que no necesitan esos datos.

**Estabilidad de la curva (jitter).** Cuánto se mueve la cresta de un frame al siguiente
dentro de una misma toma, en píxeles. Es lo que el enunciado penaliza como parpadeo. En
promedio sobre los 4 videos, 14.57 px con el Método 1 y 26.17 px con el Método 2. Por
condición de luz:

| Condición | Jitter Método 1 | Jitter Método 2 | Frames |
|---|---|---|---|
| día | **15.30 px** | 30.26 px | 424 |
| crepúsculo | **17.04 px** | 42.35 px | 94 |
| noche | **19.21 px** | 33.46 px | 473 |

**Variación de la altura con la luz.** Un pretil no cambia de altura al oscurecer, así que
la diferencia entre día y noche es error de medición y sirve como piso del error:

| Condición | Altura mediana (Método 1) |
|---|---|
| día | 0.33 m |
| crepúsculo | 0.36 m |
| noche | 0.42 m |

Son ±12 % en torno a la mediana global de 0.38 m. Antes del ADR 0007 eran ±26 %.

**Comparación con la normativa.** La normativa pide un pretil de al menos el radio de rueda
del equipo mayor, del orden de 1.5–2 m. La mediana medida, 0.38 m, es unas cuatro veces
menor.

La cobertura (qué parte de la curva se da por válida, 89.3 %) no la uso como métrica porque
depende del umbral de validación y no de la escena (reporte, sección 4.3).

---

## 3. Análisis y conclusiones

### 3.1 Detección y proximidad

**El mAP global esconde el problema.** 0.559 es moderado (un detector bien entrenado con
muchos datos anda entre 0.80 y 0.90), pero promedia un `caex` usable (0.812) con un
`bulldozer` que no lo es (0.306).

**La matriz de confusión y el mAP no se contradicen.** La matriz muestra la fila de
`bulldozer` vacía, como si nunca se predijera, pero el mAP es 0.306. La diferencia es el
umbral: la matriz usa un umbral de confianza fijo y el mAP recorre todos. El modelo sí
predice `bulldozer`, pero casi siempre bajo 0.25, el umbral de operación de
`configs/method_1.yaml`, así que en la práctica el OSD marca todo como `caex`.

**La precisión baja se convierte en falsas alertas.** `caex` tiene recall 1.000 y precisión
0.297: encuentra todos los camiones, pero ~70 % de sus cajas no corresponde a ninguno. En el
frame 120 de `video_02`, el NMS por defecto (0.7) deja cuatro cajas solapadas; el `iou=0.45`
de la configuración deja las dos correctas, pero en otros frames quedan falsos positivos.
Los gráficos lo muestran: la matriz de distancias de los ocho equipos con más permanencia no
tiene ningún par bajo 14 m, pero el mapa muestra puntos en nivel crítico. Esas alertas vienen
de tracks de vida corta, es decir, de cajas duplicadas. El tracker las reduce, pero
`video_01` sigue con un mínimo de 0.6 m.

**Las dos fallas tienen la misma causa: el desbalance.** Con 33 bulldozers contra 270 CAEX,
el modelo aprende mal la clase chica y, ante la duda, apuesta por la grande, lo que sube el
recall y baja la precisión.

**Aprendió a separar, aunque no a clasificar.** Con 33 ejemplos no alcanzó a aprender la
clase `bulldozer`, pero entrenar con cajas separadas le enseñó a no juntar máquinas vecinas,
que era lo que bloqueaba la proximidad.

**Conclusión.** El detector cumple lo que la proximidad necesita: encuentra todos los CAEX y
separa las máquinas vecinas. No sirve para saber qué tipo de equipo es, y su baja precisión
genera alertas falsas que el tracker reduce pero no elimina. Hoy las alertas sirven como
aviso, no como conteo exacto de eventos. Los dos problemas son de datos, no de diseño.

### 3.2 Altura del pretil

**Mido estabilidad, no exactitud.** Sin el perfil real anotado sé cuánto tiembla la curva,
pero no cuánto se aleja del pretil verdadero.

**El Método 1 es más estable en todas las condiciones**, con 44 % menos jitter que el
Método 2 a 1.44 veces el costo. De noche el Método 2 parecía mejor, pero era porque se
quedaba pegado al horizonte: al corregir la búsqueda (ADR 0007) su jitter nocturno subió
90 %, y eso confirmó la hipótesis.

**El error por luz es de al menos ±12 %.** Es un piso, porque un error común a todas las
condiciones, como un ancho nominal equivocado, no aparece en esa variación.

**El valor absoluto está unas cuatro veces bajo lo esperado.** No ajusté parámetros para
acercarlo. Tengo tres hipótesis sin resolver: que la base se detecta demasiado arriba en un
talud suave (la más probable), que la curva sigue el quiebre de la plataforma y no el
pretil, o que el ancho nominal del CAEX no es el de esta máquina.

**El error de escala se cancela dentro de una toma.** Es el mismo para todas las
mediciones, así que un cambio relativo, como el pretil bajando respecto de sí mismo, sí es
confiable.

**Conclusión.** La altura sirve para vigilar si el pretil se degrada respecto de su propia
línea base, pero no para verificar cumplimiento: el valor absoluto está unas cuatro veces
bajo lo esperado y varía ±12 % solo por la luz. La curva es estable con el Método 1, pero
sin ground truth no sé cuán exacta es.

---

## 4. Qué queda por hacer

Ordenado por impacto dentro de cada objetivo.

### Detección y proximidad

1. **Anotar más bulldozers**, extrayendo frames solo de los tramos donde aparece. Unas
   sesenta anotaciones más, cerca de una hora, duplicarían la clase.
2. **Copy-paste augmentation** con los 44 bulldozers que ya anoté con polígono: pegar sus
   recortes en otros frames multiplica la clase sin anotar más, y son del dominio exacto.
3. **Bajar las falsas alertas desde el tracker**, exigiendo más detecciones antes de
   confirmar un track, en vez de subir el umbral de confianza, que le quitaría recall al
   bulldozer. Hay que medir el efecto en las alertas.
4. **Medir el detector por condición de luz**, que quedó pendiente (ADR 0005).
5. **Justificar los umbrales de 20 y 10 m** contra la normativa y la distancia de frenado de
   un CAEX cargado. Hoy son preliminares.

### Altura del pretil

1. **Anotar el perfil real del pretil** en unos 30 frames estratificados por luz. Permite
   medir el error de la cresta, elegir entre métodos por exactitud y probar las tres
   hipótesis del sesgo.
2. **Calibrar la cámara**, con un tablero de ajedrez o los datos de montaje, para que la
   escala no dependa de la maquinaria.
3. **Reportar la altura relativa al radio de rueda** del equipo mayor. Es como la normativa
   define el criterio, y cancela el error de escala.
4. **Limitar la búsqueda con la posición típica de los equipos** en vez de la del que está
   más abajo, para que una caja demasiado grande no la arrastre al primer plano (ADR 0007).
5. **Un segmentador neural del pretil**, por ejemplo SAM2 guiado por la curva del Método 1,
   para comparar un enfoque aprendido contra el geométrico.

### Generales

- **Acelerar el pretil**, llevándolo a GPU o calculándolo a menor resolución. Es la etapa
  más cara con GPU (36 ms por frame a 720p).
- **Exportar a ONNX Runtime** y medir la ganancia frente a PyTorch en FP32 y FP16.
- **Validar los umbrales de luz y de cortes** con videos nuevos, porque hoy están ajustados
  con cuatro.
- **Sumar un segundo anotador**, para medir el acuerdo y la incertidumbre de las etiquetas.
- **Vigilar la deriva en producción**: alertar si la confianza media del detector empieza a
  caer.
