# Reporte de benchmark

Comparación de los métodos sobre el material de muestra: 4 videos, 1.022 frames, dos
resoluciones y luminancia media entre 19 y 163 (de noche cerrada a pleno día). Todas las
cifras se pueden reproducir con los comandos indicados.

---

## 0. Los dos métodos corren sobre los cuatro videos

Un solo comando corre ambos métodos sobre todos los videos:

```bash
python main.py --input data/raw --output output --method all
```

Genera 8 corridas (4 videos × 2 métodos), cada una en su carpeta:

```
output/
├── run_metadata.json
├── video_01/
│   ├── method_1/    <- camino óptimo
│   └── method_2/    <- línea base por argmax
├── video_02/  (method_1, method_2)
├── video_03/  (method_1, method_2)
└── video_04/  (method_1, method_2)
```

Ejemplo del `metadata.json` de `video_02`:

| Campo | `method_1` | `method_2` |
|---|---|---|
| `method_name` | Prior geométrico | Línea base por argmax |
| `berm_crest_jitter_px` | **9.94** | 17.46 |
| `average_fps` | 12.7 | **17.2** |
| `proximity_alerts` | 12 | 12 |

Las alertas coinciden porque la proximidad no depende de cómo se segmenta el terreno. En
`output_ejemplo/` están los cuatro videos con el Método 1, y `video_02` también con el
Método 2.

---

## 1. Qué comparé

| Eje | A | B |
|---|---|---|
| Detección de maquinaria | COCO preentrenado y zero-shot | Fine-tuning con datos del dominio |
| Segmentación del pretil | Camino óptimo (programación dinámica) | Argmax de gradiente por columna |
| Despliegue | CPU | CUDA |

Lo que no implementé es un segmentador **neural** del pretil, que era el Método 2 del plan
inicial. Prioricé que el Docker funcionara y estuviera verificado, que el pipeline no se
cayera con material nuevo, y el módulo de proximidad, que dependía del detector. Queda en la
sección 8.

Por eso el eje de segmentación compara dos formulaciones que comparten todo
(preprocesado, exclusión de maquinaria, banda de búsqueda y estimador de altura) y solo
cambian en cómo eligen la cresta. Es una comparación más acotada que la planeada, pero la
diferencia medida se puede atribuir a esa única decisión.

---

## 2. Métricas

| Métrica | Qué mide | Por qué |
|---|---|---|
| mAP@0.5 por clase | Calidad de detección | Para alertar importa encontrar el equipo, no ajustar la caja al píxel. Por clase, porque con un desbalance de 7:1 el promedio lo domina la clase grande |
| Precisión y recall | Cada tipo de error | Un equipo no detectado es un riesgo; un falso positivo es una falsa alarma. El mAP no los separa |
| Jitter de la cresta (px) | Estabilidad del perfil en el tiempo | El enunciado penaliza el parpadeo, y es la única medida de calidad del pretil que tengo sin ground truth |
| ms/frame y p95 | Costo | El p95, porque un video se percibe por sus peores frames |
| ms por etapa | Dónde está el cuello de botella | Un FPS total no dice qué optimizar |
| Dispersión de la altura entre condiciones de luz | Error de la altura | Ver sección 6 |

El jitter es la desviación estándar del cambio de la cresta entre frames consecutivos,
dentro de cada toma.

La **cobertura** del pretil estaba planeada como métrica y la descarté después de medirla
(sección 4.3).

**No hay ground truth del pretil.** El tiempo de anotación lo usé entero en el detector, que
era lo que bloqueaba el pipeline. Es la limitación más importante del reporte: mido
estabilidad y costo, no exactitud.

---

## 3. Detección: zero-shot contra fine-tuning

```bash
python scripts/probe_detector.py data/raw --stride 12
python scripts/probe_openvocab.py --conf 0.05
python scripts/eval_detector.py
```

### 3.1 COCO

COCO no tiene maquinaria minera. YOLO11s detecta `truck` (confianza media 0.55–0.69) y poco
más. `bus` y `train`, que había incluido por si acaso, no aparecen en ningún frame.

El conteo parecía bueno, pero en el frame 120 de `video_02`, con un CAEX y un bulldozer
juntos, el modelo pone una sola caja sobre los dos:

```
imgsz=640, conf>=0.20  ->  1 detección
    truck  0.74  x=[138,759]  ancho=622px   <- las dos máquinas
```

Subir la resolución a 1280 o 1920 no las separa.

### 3.2 Zero-shot (YOLO-World)

`yolov8s-worldv2`, siete conjuntos de prompts, `imgsz=1280`:

| Prompts | Detecciones | Resultado |
|---|---|---|
| `["truck","bulldozer"]` | 1 | `truck` 0.56, 683 px, fusionada |
| `["mining haul truck","bulldozer","excavator"]` | 1 | 0.34, 648 px, fusionada |
| `["large yellow mining dump truck","yellow tracked bulldozer with blade"]` | 1 | 0.20, 693 px, fusionada |
| `["bulldozer"]`, umbral 0.05 | **0** | |
| `["tracked bulldozer"]`, umbral 0.05 | **0** | |
| `["crawler dozer with blade"]`, umbral 0.05 | **0** | |

Mientras más específico el prompt, más baja la confianza.

Con el umbral en 0.03 aparecen las cajas correctas, pero pierden contra la fusionada:

```
truck  0.562  x=[130,814]  ancho=683   <- fusionada, gana
truck  0.141  x=[135,555]  ancho=420   <- solo el CAEX
truck  0.089  x=[515,763]  ancho=248   <- el bulldozer
```

Son dos problemas distintos. Que gane la caja fusionada se podría intentar arreglar con
umbrales. Pero que el bulldozer solo, sin competencia y con umbral casi cero, dé cero
detecciones con tres prompts distintos no se arregla con hiperparámetros: el texto no calza
con lo que el modelo ve. Es brecha de dominio: dos máquinas ocres, superpuestas, sobre suelo
ocre, con polvo y a media distancia.

### 3.3 Fine-tuning

131 frames de entrenamiento anotados a mano y 40 de validación (`video_04` completo),
separados por video y no al azar. Detalle en `docs/resultados.md`.

| Clase | Instancias (train) | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---|---|---|---|---|
| `caex` | 270 | **0.812** | 0.585 | 0.297 | **1.000** |
| `bulldozer` | 33 | **0.306** | 0.188 | 0.464 | 0.455 |
| global | | 0.559 | 0.387 | 0.381 | 0.727 |

### 3.4 Resultado

| | COCO | Zero-shot | Fine-tuned |
|---|---|---|---|
| Separa CAEX y bulldozer | No | No | **Sí** |
| Clasifica bien el bulldozer | No | No | Débil (0.306) |
| Datos anotados | 0 | 0 | 171 frames (~2 h) |

El fine-tuning resolvió lo que bloqueaba el pipeline, aunque no la clasificación. La
proximidad necesitaba que las máquinas quedaran separadas, y eso sí lo aprendió:

```
COCO:         1 caja de 622 px
detector_v1:  285 px (bulldozer) + 549 px (CAEX)
```

Con 33 ejemplos no alcanzó a aprender la clase `bulldozer`, pero entrenar con cajas
separadas le enseñó a no juntar máquinas vecinas.

Un detalle de configuración pesó tanto como el entrenamiento: con el NMS por defecto de
Ultralytics (iou=0.7) salen cuatro cajas solapadas; con `iou=0.45`
(`configs/method_1.yaml`) quedan las dos correctas.

---

## 4. Segmentación del pretil

```bash
python main.py --input data/raw --output output --method all
```

### 4.0 En qué se diferencian

La cresta del pretil es continua: no salta de una columna a la siguiente. Los dos métodos
usan ese hecho en momentos distintos:

- **Método 1, camino óptimo.** Busca el recorrido de mayor respuesta que cruza todas las
  columnas, con un salto vertical máximo entre columnas vecinas. La continuidad es parte de
  la búsqueda.
- **Método 2, argmax por columna.** Cada columna elige su máximo por separado, y después un
  filtro suaviza el resultado.

Lo que esperaba, y lo que se midió, es que el filtro no puede recuperar la continuidad que
el argmax ya perdió. Suavizar decisiones malas no las vuelve buenas.

Todo lo demás es igual en ambos: CLAHE sobre la luminancia, gradiente vertical a escala
gruesa, exclusión de las cajas de maquinaria, la misma banda de búsqueda (el mismo código,
`bermguard/vision/berm/band.py`) y el mismo estimador de altura.

### 4.1 Resultados

| | Método 1 (camino óptimo) | Método 2 (argmax + filtro) |
|---|---|---|
| **Jitter de la cresta** | **14.57 px** | 26.17 px |
| Cobertura media | 89.3 % | 86.5 % |
| Costo de la etapa | 55.0 ms/frame | **38.1 ms/frame** |
| Pipeline completo | 10.7 fps | **13.9 fps** |
| ms/frame (p95) | 117.7 | 97.8 |
| Altura mediana | 0.38 m | 0.37 m |

**El Método 1 tiene 44 % menos jitter y cuesta 1.44 veces más.** La búsqueda global es más
cara, pero da un perfil que no puede saltar.

Los tiempos de ambos métodos salen de la misma corrida, en el host y sin otros procesos
usando la GPU, así que son comparables entre sí. No los comparo con corridas de otras
sesiones, porque en un portátil la variación entre sesiones es real.

La altura mediana es casi igual porque ambos usan el mismo estimador; lo que cambia es
cuánto oscila.

### 4.2 Por condición de luz

| Condición | Jitter M1 (px) | Jitter M2 (px) | Frames |
|---|---|---|---|
| día | **15.30** | 30.26 | 424 |
| crepúsculo | **17.04** | 42.35 | 94 |
| noche | **19.21** | 33.46 | 473 |

El Método 1 es más estable en las tres: 2.0× de día, 2.5× en crepúsculo y 1.7× de noche.

#### El Método 2 parecía mejor de noche

En la primera medición el Método 2 salía más estable de noche (17.58 px contra 22.10).
Anoté entonces que eso no lo hacía mejor: el jitter mide estabilidad y no exactitud, y un
argmax que se queda pegado frame tras frame en el mismo borde equivocado da un jitter bajo.

Revisando los videos nocturnos encontré ese borde: la línea entre el terreno oscuro y el
valle iluminado del fondo, que de noche es el borde horizontal más fuerte y estaba dentro de
la banda de búsqueda. El ADR 0007 limitó la banda por arriba con la maquinaria y dejó ese
borde afuera, y con eso se pudo probar la hipótesis:

| Noche | Antes del ADR 0007 | Después |
|---|---|---|
| Método 1 | 22.10 px | 19.21 px |
| Método 2 | **17.58 px** | **33.46 px** |

Sin el horizonte disponible, el jitter nocturno del Método 2 casi se duplicó: su estabilidad
venía de estar pegado al objetivo equivocado. No es una medición de exactitud, pero es un
cambio controlado que confirma la explicación.

El arreglo dejó dos defectos a la vista, descritos en el ADR 0007:

- Donde la banda de búsqueda no tiene un borde claro, la curva baja hasta su límite
  inferior.
- Si el detector entrega una caja demasiado grande, el límite inferior de la banda baja al
  primer plano. En el frame 130 de `video_04` la caja del CAEX llega a la fila 664, y la
  curva termina sobre las huellas de neumático en vez de sobre el banco detrás del
  bulldozer.

Es un recordatorio de que menos jitter significa un perfil más estable, no uno más
correcto.

### 4.3 La cobertura no sirve como métrica

La cobertura es el porcentaje de columnas donde el perfil se considera válido. Probé dos
criterios de validación:

| Criterio | Cobertura |
|---|---|
| Percentil 55 de la respuesta del camino | 43 % en todos los videos y condiciones |
| 3 × la mediana de la banda de búsqueda | 92–95 % en todos los videos y condiciones |

Ninguna describe el pretil. Un percentil siempre acepta la misma fracción, y el segundo
criterio acepta casi todo porque el gradiente suavizado casi nunca es cero. Que la cifra no
cambie entre día y noche, con dos órdenes de magnitud de diferencia en el contraste, muestra
que mide el umbral y no la escena. La reporto con esta advertencia.

Después del ADR 0007 bajó unos cuatro puntos (de 93.6 a 89.3 % en el Método 1, y 87 % de
noche), lo esperable al achicar la banda.

---

## 5. Despliegue: CPU contra CUDA

Medido dentro del contenedor, con el comando de referencia y la imagen construida desde este
código. La fila de 720p junta los tres videos de esa resolución (entre paréntesis, el rango
entre videos); la de 1080p es `video_01`, el único en esa resolución.

| Resolución | CPU (sin `--gpus`) | GPU (`--gpus all`) | Ganancia |
|---|---|---|---|
| 1280×720 | 8.1 fps (7.5–9.1) | 15.0 fps (14.8–15.2) | **1.8×** |
| 1920×1080 | 5.2 fps | 7.3 fps | **1.4×** |

La ganancia es de 1.8×, no de un orden de magnitud. El desglose de `video_02`, en ms/frame,
muestra por qué:

| Etapa | CPU | GPU |
|---|---|---|
| detección | 60.2 | **15.6** |
| pretil (NumPy/OpenCV) | 35.7 | 36.4 |
| luminancia y horizonte | 3.6 | 3.9 |
| dibujo del OSD | 2.9 | 3.0 |
| corte de toma | 0.9 | 1.0 |
| altura, proximidad y tracking | < 0.5 | < 0.5 |

La GPU divide la detección por casi cuatro y no cambia nada más. En CPU la etapa más cara es
la detección; con GPU pasa a ser el pretil, que cuesta 2.3 veces lo que la detección y no se
acelera. A 1080p el pretil sube a 74 ms y la ganancia baja a 1.4×. Es la ley de Amdahl, y
deja dos conclusiones:

1. El modo CPU es usable (8 fps a 720p y 5 fps a 1080p), lo que respalda el ADR 0001.
2. Optimizar el detector serviría poco. Lo que rendiría es llevar el pretil a GPU o bajarle
   la resolución.

Notas sobre la medición:

- En CPU la detección varió entre 60 y 123 ms por frame en videos de la misma resolución.
  Su costo no depende del contenido, así que la variación es del equipo: al repetir
  `video_04` solo, bajó de 123 a 79 ms. Es la temperatura del portátil en una corrida larga.
  La tabla usa esa repetición, y por eso doy el rango.
- CPU y GPU no dan resultados idénticos, porque la GPU infiere en fp16 y la CPU en fp32. En
  `video_01` difieren en una alerta (33 contra 34); en los otros tres videos coinciden.
- Estas cifras no se comparan con las de las secciones 0 y 4, que son del host Windows en
  otra sesión (ver 4.1).

---

## 6. Rigor de la medición métrica

La altura se calcula sin calibrar la cámara (`bermguard/analytics/height.py`). La fórmula
tiene una ventaja: la focal se cancela.

```
H = (y_base − y_cresta) · h / (y_base − y_horizonte)
```

Así la altura no depende del campo de visión asumido, que es el dato más incierto. Solo
necesita el horizonte y la altura de la cámara, y esta sale del ancho nominal de un CAEX
detectado.

### 6.1 Una cota del error sin ground truth

Altura mediana por condición de luz (Método 1):

| Condición | Altura mediana |
|---|---|
| día | 0.33 m |
| crepúsculo | 0.36 m |
| noche | 0.42 m |

Un pretil no cambia de altura cuando oscurece, así que toda esa variación es error de
medición. Eso da una cota inferior del error sin datos anotados: **±12 % en torno a la
mediana de 0.38 m**, solo por el cambio de luz.

Antes del ADR 0007 era ±26 %, con la noche en 0.63 m. Casi toda esa diferencia venía de
medir la cresta en el horizonte: la distancia hasta la base se alargaba y la altura salía
inflada.

Es una cota inferior porque un error común a las tres condiciones, como un ancho nominal
equivocado, no aparece en esta dispersión.

### 6.2 El sesgo sistemático

La normativa pide un pretil de al menos el radio de rueda del equipo mayor, del orden de
1.5–2 m. La mediana medida es 0.38 m, más o menos cuatro veces menos. Antes del ADR 0007 el
factor era 3: el arreglo quitó un error que inflaba las alturas nocturnas y dejó el sesgo
más a la vista.

No ajusté ningún parámetro para acercarla al valor esperado. Tengo tres hipótesis, sin
resolver:

1. **La base se detecta muy arriba.** El estimador la pone donde el gradiente vuelve al
   nivel de fondo, y en un talud suave ese punto puede quedar bastante sobre el pie real. Es
   la más probable.
2. **La estructura que sigue no es el pretil**, sino el quiebre general de la plataforma,
   que tiene menos relieve.
3. **El ancho nominal del CAEX no es el correcto** para esta máquina, lo que escalaría todas
   las alturas por igual.

Para distinguirlas hace falta ground truth. **La altura absoluta no sirve para verificar
cumplimiento.**

### 6.3 Dónde sí es confiable

El error principal es de escala, y es el mismo para todas las mediciones de una toma. Por
eso detectar que el pretil baja respecto de su propia línea base sí es confiable, porque ese
factor se cancela; decir que mide 0.38 m no lo es. A un supervisor lo que le sirve es saber
si la altura está bajando.

---

## 7. Proximidad

Con el detector entrenado ya hay dos entidades entre las que medir. Cuento las alertas como
**subidas de nivel** y no como frames en riesgo: un equipo diez segundos en zona crítica es
un evento, no doscientas alertas. Los niveles usan histéresis (3 frames para subir, 10 para
bajar) para no parpadear.

| Video | Alertas | Distancia mínima | Mediana |
|---|---|---|---|
| `video_01` | 39 | 0.6 m | 17.4 m |
| `video_02` | 12 | 2.9 m | 23.7 m |
| `video_03` | 16 | 4.3 m | 19.4 m |
| `video_04` | 13 | 0.9 m | 19.3 m |

Parte de estas alertas es falsa, y los gráficos lo muestran: la matriz de distancias de los
ocho equipos con más permanencia no tiene ningún par bajo 14 m, pero el mapa muestra puntos
en nivel crítico. Esas alertas vienen de tracks de vida corta, es decir, de cajas duplicadas
sobre una misma máquina. Es efecto de la precisión 0.297 del detector.

El tracker lo mitiga con dos mecanismos:

| Mecanismo | Efecto |
|---|---|
| Confirmar un track tras 3 detecciones seguidas | Elimina falsos positivos aislados sobre polvo |
| Suprimir duplicados por contención | Alertas de 43 a 39 en `video_01`; distancia mínima de 0.6 a 2.9 m en `video_02` |

Para los duplicados uso la intersección sobre el área menor y no IoU: dos cajas desplazadas
sobre el mismo camión tienen un IoU de ~0.35, igual que dos equipos reales, pero una
contención alta.

Es una mitigación parcial: `video_01` sigue con un mínimo de 0.6 m. La causa es la precisión
del detector, o sea, faltan datos.

---

## 8. Limitaciones y trabajo futuro

Ordenado por impacto:

1. **No hay ground truth del pretil.** Sin él no mido exactitud. El ADR 0007 explicó la
   inversión nocturna con un experimento controlado, pero sigo sin saber cuánto se aleja la
   cresta del pretil real. Anotar el perfil en unos 30 frames estratificados permitiría
   medir ese error y decidir el eje por exactitud.
2. **La clase `bulldozer` es débil** (mAP 0.306, recall 0.455) por falta de datos: 33
   instancias contra 270. Los datasets externos que evalué no sirven por brecha de dominio
   (ADR 0004). Lo más eficiente sería copy-paste augmentation con las 44 máscaras
   poligonales de bulldozer que ya anoté, que son del dominio exacto.
3. **El detector sobre-detecta** (precisión 0.297, recall 1.000). Subir el umbral le quitaría
   recall a la clase que menos tiene; es mejor exigir más persistencia en el tracker y medir
   el efecto en las alertas falsas.
4. **El sesgo de la altura** (factor ~4) tiene tres hipótesis abiertas (sección 6.2).
5. **Falta el segmentador neural del pretil.** Con SAM2 usando el camino del Método 1 como
   prompt, la comparación pasaría a ser prior geométrico contra modelo aprendido, que es lo
   que propone el enunciado.
6. **El pretil es el cuello de botella** con GPU: 36 ms/frame a 720p y 74 ms a 1080p, contra
   16 y 29 ms de la detección. Llevarlo a GPU, o calcularlo a menor resolución e
   interpolar, es la optimización que vale la pena.
7. **Exportar a ONNX Runtime** y medir la ganancia frente a PyTorch en FP32 y FP16. No
   alcancé a medirlo.
8. **Los umbrales se calibraron con cuatro videos** (luz, cortes, campo de visión). Están en
   `configs/`, pero no sé cómo generalizan al conjunto de evaluación.
9. **Anoté yo solo**, sin acuerdo entre anotadores, así que las métricas de detección tienen
   una incertidumbre que no medí.
