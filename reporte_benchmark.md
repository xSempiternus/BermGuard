# Reporte de benchmark — BermGuard AI

Comparación de los métodos implementados, sobre el material de muestra: 4 videos,
1.022 frames, en dos resoluciones y con condiciones lumínicas que recorren de luma
media 19 a 163 — dos órdenes de magnitud de contraste entre las escenas nocturnas y
las diurnas.

Todo lo que sigue es medido y reproducible con los comandos que se indican. Cuando una
cifra no se puede interpretar, el reporte lo dice en lugar de presentarla.

---

## 0. Los dos métodos se ejecutaron sobre los cuatro videos

Antes de las comparaciones, la evidencia de que ambos corrieron de verdad.

Un solo comando ejecuta los dos métodos sobre todos los videos de la carpeta de
entrada — es el argumento «todo en 1» que pide el enunciado:

```bash
python main.py --input data/raw --output output --method all
```

Produce **8 corridas**: 4 videos × 2 métodos, cada una con su subdirectorio propio.

```
output/
├── run_metadata.json
├── video_01/
│   ├── method_1/    <- camino optimo
│   └── method_2/    <- linea base por argmax
├── video_02/  (method_1, method_2)
├── video_03/  (method_1, method_2)
└── video_04/  (method_1, method_2)
```

Cada `metadata.json` registra qué método lo produjo y con qué resultado. Sobre
`video_02`, por ejemplo:

| Campo | `method_1` | `method_2` |
|---|---|---|
| `method_name` | Prior geométrico | Línea base por argmax |
| `berm_crest_jitter_px` | **9.94** | 17.46 |
| `average_fps` | 12.7 | **17.2** |
| `proximity_alerts` | 12 | 12 |

Los números difieren porque los métodos difieren; las alertas coinciden porque el
módulo de proximidad es común a ambos y no depende de cómo se segmente el terreno.

En `output_ejemplo/` del repositorio están los artefactos reales de esa corrida:
los cuatro videos con el Método 1, y `video_02` también con el Método 2 para que el
contraste se pueda ver y no sólo leer.

---

## 1. Qué se comparó, y qué no

Se midieron **tres ejes**:

| Eje | Método A | Método B | Estado |
|---|---|---|---|
| **Detección de maquinaria** | Preentrenado COCO + zero-shot open-vocabulary | Fine-tuning sobre el dominio | Completo |
| **Segmentación del pretil** | Camino óptimo por programación dinámica | Máximo de gradiente por columna | Completo |
| **Despliegue** | CPU | CUDA | Completo |

**Lo que no está implementado:** un segmentador **neural** del pretil, que era el
Método 2 del plan original. Se priorizaron, en este orden, el despliegue Docker
verificado, el pipeline que no falla sobre material desconocido, y el módulo de
proximidad —que estaba bloqueado por la detección—. Se declara como trabajo pendiente
en la sección 8, no como algo omitido por descuido.

El eje de segmentación quedó por tanto entre dos formulaciones que comparten
preprocesado, exclusión de maquinaria y banda de búsqueda, y difieren **sólo** en
cómo se decide la cresta. Es un contraste más estrecho que el planificado, pero
aislado: la diferencia medida es atribuible a la formulación y a nada más.

---

## 2. Elección de métricas

| Métrica | Qué mide | Por qué se eligió |
|---|---|---|
| **mAP 0.5 por clase** | Calidad de detección | 0.5 y no 0.5:0.95 porque para disparar una alerta de proximidad importa *detectar* el equipo, no bordearlo al píxel. **Por clase y no global**: con un desbalance de 7:1 el promedio queda dominado por la mayoritaria |
| **Precisión y recall** | Errores de cada tipo | En seguridad los dos errores no son equivalentes: un equipo no detectado es un riesgo no vigilado; un falso positivo es una alarma espuria. Un mAP no los separa |
| **Jitter de la cresta (px)** | Estabilidad temporal del perfil | Es la métrica que el enunciado pide sin nombrarla al penalizar el «parpadeo». Y es la **única** medida de calidad de segmentación disponible sin ground truth |
| **ms/frame y percentil 95** | Costo computacional | El p95 acompaña a la media porque un pipeline de video se percibe por sus peores frames |
| **Desglose por etapa** | Dónde está el cuello de botella | Un FPS agregado no dice qué optimizar |
| **Dispersión de la altura entre condiciones** | Error de la medición métrica | Ver sección 6. Es una cota inferior del error obtenida de una invariancia física, sin necesidad de ground truth |

**Métrica descartada tras medirla: la cobertura del pretil.** Se planificó como medida
de calidad de la segmentación y resultó no medir la escena. El detalle está en la
sección 5.3.

**No hay mIoU ni ground truth anotado del pretil.** Etiquetar a mano el perfil de la
cresta en frames estratificados era el plan, y el presupuesto de anotación se destinó
íntegro al conjunto de entrenamiento del detector, que era el bloqueo del pipeline. Es
la limitación más importante de este reporte: **sin ground truth no se mide exactitud,
sólo estabilidad y costo.**

---

## 3. Eje 1 — Detección: zero-shot contra fine-tuning

Reproducible con:

```bash
python scripts/probe_detector.py data/raw --stride 12
python scripts/probe_openvocab.py --conf 0.05
python scripts/eval_detector.py
```

### 3.1 Preentrenado en COCO

COCO no contiene maquinaria minera. Sobre el material, YOLO11s activa `truck` de forma
consistente (confianza media 0.55–0.69) y nada más de forma relevante. Las clases `bus`
y `train`, incluidas preventivamente en el mapeo inicial, **no se activan en ningún
frame**; se eliminaron.

El conteo agregado sugería éxito. La inspección visual del frame 120 de `video_02`,
donde un CAEX y un bulldozer aparecen contiguos, mostró lo contrario:

```
imgsz=640,  conf>=0.20  ->  1 deteccion
    truck  0.74  x=[138,759]  ancho=622px   <- ambas maquinas en una caja
```

Aumentar la resolución de inferencia a 1280 y 1920 no separa; produce cajas
superpuestas e inestables.

### 3.2 Open-vocabulary zero-shot

YOLO-World (`yolov8s-worldv2`), siete conjuntos de prompts, `imgsz=1280`:

| Prompts | Detecciones | Resultado |
|---|---|---|
| `["truck","bulldozer"]` | 1 | `truck` 0.56, 683 px — fusionada |
| `["mining haul truck","bulldozer","excavator"]` | 1 | 0.34, 648 px — fusionada |
| `["large yellow mining dump truck","yellow tracked bulldozer with blade"]` | 1 | 0.20, 693 px — fusionada |
| `["bulldozer"]`, umbral 0.05 | **0** | — |
| `["tracked bulldozer"]`, umbral 0.05 | **0** | — |
| `["crawler dozer with blade"]`, umbral 0.05 | **0** | — |

La confianza **baja** al aumentar la especificidad del prompt, que es lo contrario de
lo esperado si el modelo entendiera la descripción.

Bajando el umbral a 0.03 se comprobó que la descomposición correcta **existe** en el
conjunto de propuestas pero pierde:

```
truck  0.562  x=[130,814]  ancho=683   <- fusionada, gana
truck  0.141  x=[135,555]  ancho=420   <- el CAEX solo
truck  0.089  x=[515,763]  ancho=248   <- el bulldozer
```

**Son dos fallos independientes, y la distinción decide si el enfoque es recuperable.**
El primero es de ranking: la caja fusionada domina, y en principio se podría atacar
ajustando umbrales. El segundo no lo es: sin competencia alguna y con el umbral casi en
el suelo, tres redacciones distintas devuelven cero. Si el embedding de texto no
coincide con la evidencia visual, ningún hiperparámetro lo hace activar.

La causa de fondo es brecha de dominio: dos máquinas ocres, parcialmente superpuestas,
sobre suelo ocre, con polvo y a media distancia. El vocabulario abierto condiciona las
features mediante texto, pero la propuesta de región sigue dominada por el backbone
visual.

### 3.3 Fine-tuning sobre el dominio

131 frames de entrenamiento anotados a mano, 40 de validación (`video_04` íntegro),
partición **por video y no aleatoria**. Detalle en `docs/resultados_deteccion.md`.

| Clase | Instancias (train) | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---|---|---|---|---|
| `caex` | 270 | **0.812** | 0.585 | 0.297 | **1.000** |
| `bulldozer` | 33 | **0.306** | 0.188 | 0.464 | 0.455 |
| global | | 0.559 | 0.387 | 0.381 | 0.727 |

### 3.4 Veredicto del eje

| | COCO | Zero-shot | Fine-tuned |
|---|---|---|---|
| Separa CAEX de bulldozer en cajas distintas | No | No | **Sí** |
| Etiqueta correctamente la clase minoritaria | No | No | Débilmente (0.306) |
| Datos etiquetados necesarios | 0 | 0 | 171 frames (~2 h) |
| Recall sobre la clase mayoritaria | — | — | 1.000 |

**El fine-tuning resolvió el bloqueo del pipeline aunque falló en la etiqueta.** La
razón de ser del entrenamiento no era la clase, sino que el modelo base fusionaba las
dos máquinas en una caja y eliminaba la magnitud que la proximidad necesita medir:

```
COCO:         1 caja de 622 px
detector_v1:  285 px (bulldozer) + 549 px (CAEX)
```

Localización y clasificación se aprendieron de forma muy desigual. 33 instancias no
alcanzan para una categoría visual nueva, pero entrenar con cajas separadas sí enseñó
a **no fusionar máquinas adyacentes**, y el problema bloqueante era ese.

**Un hiperparámetro pesó tanto como los pesos:** con el NMS por defecto de Ultralytics
(iou=0.7) el modelo emite cuatro cajas solapadas; con el `iou=0.45` de
`configs/method_1.yaml` quedan exactamente las dos correctas.

---

## 4. Eje 2 — Segmentación del pretil

Reproducible con `python main.py --input data/raw --output output --method all`.

### 4.0 La diferencia, en una frase

El pretil es una estructura **continua**: su cresta no puede saltar de una columna a la
siguiente. Ese hecho es un prior fuerte, y los dos métodos lo usan en momentos
distintos.

| | Cómo decide la cresta |
|---|---|
| **Método 1** — camino óptimo | Busca el recorrido de máxima respuesta a través de **todas** las columnas a la vez, con el salto vertical acotado entre columnas contiguas. Ninguna columna puede elegir un valor incompatible con sus vecinas, porque la continuidad es una **restricción de la búsqueda** |
| **Método 2** — argmax por columna | Cada columna toma su propio máximo **de forma independiente**, sin mirar a las demás, y después un filtro intenta reparar el perfil resultante |

Y de ahí sale la predicción que el benchmark confirma: **el filtrado no puede recuperar
lo que el `argmax` descartó.** Cuando el filtro actúa, la información de continuidad ya
se perdió — cada columna eligió su máximo local sin saber nada de sus vecinas, y
suavizar una secuencia de decisiones malas no produce una buena.

Todo lo demás es **idéntico** entre ambos: CLAHE sobre luminancia, gradiente vertical
positivo a escala gruesa, exclusión de las cajas de maquinaria, banda de búsqueda entre
el horizonte y la rasante, y el mismo estimador de altura. Es deliberado — con el
preprocesado compartido, la diferencia medida es atribuible a la formulación y a nada
más.

### 4.1 Resultados agregados

| | Método 1 (camino óptimo) | Método 2 (argmax + filtro) |
|---|---|---|
| **Jitter de la cresta** | **14.57 px** | 26.17 px |
| Cobertura media | 89.3 % | 86.5 % |
| Costo de la etapa | 55.0 ms/frame | **38.1 ms/frame** |
| Pipeline completo | 10.7 fps | **13.9 fps** |
| ms/frame (p95) | 117.7 | 97.8 |
| Altura mediana | 0.38 m | 0.37 m |

**El trade-off es explícito: el Método 1 reduce el jitter un 44 % a cambio de 1.44× el
costo.** Ambas cifras son consecuencia directa de la formulación. La búsqueda de camino
óptimo evalúa toda la rejilla con una restricción de continuidad, lo que cuesta más y
produce un perfil que no puede saltar; el `argmax` decide cada columna en una operación
trivial y el filtrado posterior sólo atenúa lo que ya se rompió.

Las cifras son posteriores al ADR 0007, y los tiempos de ambos métodos provienen de la
misma corrida, sin otros procesos compitiendo por la GPU: la comparación entre ellos es
válida. No se comparan con los tiempos de corridas anteriores, que se hicieron en
sesiones distintas de un equipo portátil donde la variación entre sesiones es real.

Las alturas medianas coinciden (0.38 vs 0.37 m) porque ambos alimentan el mismo
estimador métrico. Lo que cambia no es el valor central sino su estabilidad.

### 4.2 Desglose por condición lumínica, y una inversión que resultó ser un artefacto

| Condición | Jitter M1 (px) | Jitter M2 (px) | n |
|---|---|---|---|
| día | **15.30** | 30.26 | 424 |
| crepúsculo | **17.04** | 42.35 | 94 |
| noche | **19.21** | 33.46 | 473 |

El Método 1 es más estable en las tres condiciones: 2.0× de día, 2.5× en crepúsculo y
1.7× de noche.

#### La inversión nocturna, y el experimento que la explicó

La primera medición de este eje decía otra cosa: de noche el Método 2 parecía el más
estable, con 17.58 px contra 22.10. La lectura que se registró entonces fue que eso no lo
hacía mejor, porque **el jitter mide estabilidad, no exactitud**: un `argmax` que se
bloquea frame tras frame sobre el mismo máximo local produce jitter bajo y una respuesta
consistentemente equivocada. Sin ground truth no había forma de distinguir «estable y
correcto» de «estable y equivocado».

La revisión visual de los videos nocturnos identificó el objetivo equivocado. No eran
principalmente los penachos de polvo iluminados, como suponía la primera versión de este
reporte, sino **la frontera entre el terreno oscuro y el valle iluminado del fondo**, que
es el borde horizontal más fuerte de la escena nocturna y quedaba dentro de la banda de
búsqueda. El ADR 0007 corrigió la banda anclándola a la maquinaria también por arriba, y
con ello sacó ese borde de la región posible.

El resultado es el experimento que valida la hipótesis:

| Noche | Antes del ADR 0007 | Después |
|---|---|---|
| Método 1 | 22.10 px | 19.21 px |
| Método 2 | **17.58 px** | **33.46 px** |

Sin el horizonte a su alcance, el jitter nocturno del Método 2 **casi se duplica**: su
estabilidad anterior era la de estar anclado a un objetivo equivocado. La hipótesis se
planteó sin ground truth y se verificó con un cambio controlado, que es lo más cerca de una
medición de exactitud a que llega este reporte.

La corrección no es gratuita, y dejó dos defectos a la vista:

- Donde la banda no contiene un borde claro —junto al camión, en el frame nocturno de
  referencia— el camino baja hasta el límite inferior de la banda en lugar de seguir una
  estructura.
- Cuando el detector emite una caja demasiado grande, el límite inferior de la banda se
  desplaza hacia el primer plano. En el frame 130 de `video_04` la caja del CAEX llega a la
  fila 664, la banda se extiende hasta la 693, y la curva se asienta sobre las huellas de
  neumático del primer plano en lugar de sobre el banco que está detrás del bulldozer. Antes
  del ADR 0007 el error de ese mismo frame iba hacia arriba, hacia la llanura del fondo;
  ahora va hacia abajo.

El segundo es la interacción de dos limitaciones ya declaradas —la precisión del detector y
la dominancia del gradiente del primer plano—, y es un recordatorio de lo que advierte la
propia sección: un jitter menor es un perfil más estable, no uno más correcto. Ambos
defectos están descritos en el ADR 0007.

### 4.3 La cobertura no es una métrica utilizable

Se planificó como medida de calidad de la segmentación. Se probaron dos criterios de
validación de las columnas del perfil:

| Criterio | Cobertura medida |
|---|---|
| Percentil 55 de la respuesta del camino | 43 % en los 4 videos y las 3 condiciones |
| 3 × la mediana de la banda de búsqueda | 92–95 % en los 4 videos y las 3 condiciones |

**Ninguna de las dos describe el pretil.** La primera es el complemento del percentil
elegido: un percentil selecciona por rango y por tanto acepta siempre la misma
fracción. La segunda es alta y uniforme porque la respuesta de gradiente a escala
gruesa es suave y no nula en casi todas las columnas.

Que la cifra sea **insensible a la condición lumínica**, cuando el contraste varía dos
órdenes de magnitud entre día y noche, es la señal de que mide el umbral y no la escena.
Se reporta con esta advertencia en lugar de presentarse como calidad.

Tras el ADR 0007 la cobertura bajó unos cuatro puntos —de 93.6 a 89.3 % en el Método 1, y a
87 % de noche—, como es esperable al estrechar la banda. No cambia la conclusión: sigue
dominada por el criterio de validación y no por la escena.

---

## 5. Eje 3 — Despliegue: CPU contra CUDA

Medido dentro del contenedor, con el comando de referencia del enunciado y la imagen
construida desde este mismo código. La fila de 720p agrega los tres videos de esa
resolución, con el rango entre videos entre paréntesis; la de 1080p corresponde a
`video_01`, el único en esa resolución.

| Resolución | Sin `--gpus` (CPU) | Con `--gpus all` | Ganancia |
|---|---|---|---|
| 1280×720 | 8.1 fps (7.5–9.1) | 15.0 fps (14.8–15.2) | **1.8×** |
| 1920×1080 | 5.2 fps | 7.3 fps | **1.4×** |

**La ganancia es de 1.8×, no de un orden de magnitud, y el desglose por etapa explica
por qué.** Sobre `video_02`, en ms/frame:

| Etapa | CPU | GPU |
|---|---|---|
| detección | 60.2 | **15.6** |
| pretil (NumPy/OpenCV) | 35.7 | 36.4 |
| luminancia y horizonte | 3.6 | 3.9 |
| renderizado del OSD | 2.9 | 3.0 |
| corte de toma | 0.9 | 1.0 |
| altura, proximidad y tracking | < 0.5 | < 0.5 |

La GPU divide por casi cuatro la detección y deja todo lo demás igual. En CPU la
detección es la etapa dominante; con GPU pasa a serlo el pretil, que cuesta 2.3 veces lo
que la detección y no se acelera, de modo que acota la ganancia total. A 1080p el efecto
se acentúa: el pretil sube a 74 ms en ambos modos y la ganancia cae a 1.4×. Es la ley de
Amdahl, y tiene dos consecuencias operativas:

1. **El modo CPU es utilizable** —8 fps a 720p y 5 fps a 1080p—, lo que respalda la
   decisión de degradar en lugar de exigir GPU (ADR 0001).
2. **Optimizar el detector daría retorno marginal.** El trabajo rendidor sería llevar la
   búsqueda de camino óptimo a GPU, o reducir su resolución de trabajo.

Tres observaciones sobre la medición:

- **En CPU, la detección varió entre 60 y 123 ms por frame** entre videos de la misma
  resolución, con el mismo modelo. Su costo no depende del contenido, así que la variación
  es del equipo: repetido en aislamiento, `video_04` bajó de 123 a 79 ms. Es el régimen
  térmico de un portátil bajo carga sostenida; la tabla usa esa repetición, y por eso las
  cifras de CPU se dan con su rango.
- **CPU y GPU no producen artefactos idénticos bit a bit.** La GPU infiere en fp16 y la
  CPU en fp32; en `video_01` difieren en una alerta de proximidad (33 contra 34) y en los
  otros tres videos coinciden.
- **Estas cifras no se comparan con las de las secciones 0 y 4**, tomadas en el host
  Windows en otra sesión, por la razón que da la sección 4.1.

Medir el desglose antes de optimizar evitó invertir esfuerzo en la etapa equivocada.

---

## 6. Rigor de la medición métrica

La altura se obtiene sin calibración de cámara. El modelo está en
`bermguard/analytics/height.py`; su propiedad útil es que **la focal se cancela**:

```
H = (y_base − y_cresta) · h / (y_base − y_horizonte)
```

de modo que la altura no depende del campo de visión asumido, que es el parámetro más
incierto de la cadena. Sólo necesita el horizonte y la altura de montaje, y esta última
se deriva del ancho nominal de un CAEX detectado.

### 6.1 Una cota del error sin ground truth

La altura medida, desglosada por condición lumínica:

| Condición | Altura mediana M1 |
|---|---|
| día | 0.33 m |
| crepúsculo | 0.36 m |
| noche | 0.42 m |

**Un pretil físico no cambia de altura al atardecer.** Toda esa dispersión es error de
medición, y como la invariancia tiene que cumplirse por física, la dispersión observada
es una **cota inferior del error del método** que no requiere ningún dato etiquetado:

> ±12 % en torno a la mediana global de 0.38 m, sólo por el cambio de iluminación.

Antes del ADR 0007 esta misma cota era de **±26 %**, con la noche en 0.63 m. La mayor parte
de esa dispersión la producía la confusión con el horizonte: medir la cresta sobre la
frontera del valle alarga la extensión vertical hasta la base e infla la altura. Corregir
la banda redujo la cota a la mitad, que es la otra forma en que este reporte mide una
mejora sin necesidad de ground truth.

Es una cota inferior porque un error sistemático común a las tres condiciones —el ancho
nominal supuesto, por ejemplo— no aparecería en esta dispersión.

### 6.2 El sesgo sistemático

La normativa referencia la altura mínima del pretil al radio de rueda del equipo mayor,
del orden de **1.5–2 m**. La medición da una mediana de **0.38 m**: baja por un factor
próximo a 4.

El factor era de 3 antes del ADR 0007. La corrección no empeoró la medición: eliminó un
error que casualmente inflaba las alturas nocturnas, y con ello el subregistro de fondo
quedó más a la vista.

No se ajustó ningún parámetro para acercarla a la cifra esperada. Las hipótesis, sin
resolver:

1. **La base se detecta demasiado alta.** El estimador la busca donde la respuesta de
   gradiente vuelve al nivel de fondo, y en un talud de pendiente suave ese punto puede
   quedar muy por encima del pie real. Sería el sospechoso principal.
2. **La estructura seguida no es el pretil normativo** sino el quiebre general de la
   plataforma, que es un rasgo del terreno de menor relieve.
3. **El ancho nominal supuesto es incorrecto** para esta maquinaria, lo que escalaría
   todas las alturas por el mismo factor.

Distinguirlas requiere ground truth, y por tanto queda fuera del alcance de esta
entrega. **La medición absoluta no debe usarse para verificar cumplimiento normativo.**

### 6.3 Dónde el sistema sí es confiable

El error dominante es de escala, y una escala equivocada es **común a todas las
mediciones de la misma toma**. Por lo tanto:

- **Detectar que el pretil se degrada respecto a su propia línea base es confiable**,
  porque un factor de escala constante se cancela en la comparación.
- **Afirmar que el pretil mide 0.38 m no lo es.**

Y ése es además el caso de uso operativo: a un supervisor le importa que la altura esté
disminuyendo, no el valor absoluto con dos decimales.

---

## 7. Proximidad

Con el detector especializado el módulo pasa a ser posible, porque hay dos entidades
entre las que medir. Las alertas se cuentan como **escaladas de nivel** y no como frames
en riesgo: un equipo diez segundos en zona crítica es un evento, no doscientas alertas.

| Video | Alertas | Distancia mínima registrada | Mediana |
|---|---|---|---|
| `video_01` | 39 | 0.6 m | 17.4 m |
| `video_02` | 12 | 2.9 m | 23.7 m |
| `video_03` | 16 | 4.3 m | 19.4 m |
| `video_04` | 13 | 0.9 m | 19.3 m |

**Una parte de estas alertas es espuria, y las propias figuras lo delatan.** La matriz
de distancias mínimas restringida a los ocho equipos con mayor permanencia no contiene
ningún par por debajo de 14 m, mientras el mapa de dispersión muestra puntos en nivel
crítico. Las dos cosas son correctas, y su desacuerdo localiza el problema: las alertas
críticas provienen de identificadores **efímeros**, es decir de cajas duplicadas sobre
una misma máquina.

Es la consecuencia aguas abajo de la precisión 0.297 del detector. Se mitigó con dos
mecanismos del tracker:

| Mecanismo | Efecto medido |
|---|---|
| Confirmación tras 3 detecciones consecutivas | Elimina los falsos positivos aislados sobre polvo |
| Supresión de duplicados por contención | Alertas 43→39 en `video_01`; distancia mínima 0.6→2.9 m en `video_02` |

La supresión usa **intersección sobre el área menor y no IoU**: dos cajas desplazadas
sobre el mismo camión tienen IoU moderado —del orden de 0.35, indistinguible de dos
equipos reales— y contención alta.

Mitigación parcial, no solución: `video_01` conserva un mínimo de 0.6 m. La causa raíz
es la precisión del detector, y por tanto un problema de datos.

---

## 8. Limitaciones y trabajo futuro

En orden de impacto sobre lo que este reporte no pudo medir o resolver.

**1. No hay ground truth del pretil.** Es la limitación central. Sin él no se mide
exactitud, sólo estabilidad y costo. La inversión nocturna del jitter quedó explicada por el
ADR 0007, pero con un experimento controlado y no con una medición de error: sigue sin
conocerse cuánto se aparta la cresta del pretil real. Anotar su perfil en unos 30 frames
estratificados permitiría calcular ese error absoluto y decidir el eje de segmentación con
exactitud y no sólo con estabilidad.

**2. La clase `bulldozer` está aprendida pero es frágil** (mAP 0.306, recall 0.455). La
causa es de datos: 33 instancias frente a 270. La vía más eficiente no es un dataset
externo —se evaluaron tres y se descartaron por brecha de dominio, ver ADR 0004— sino
**copy-paste augmentation** con las 44 máscaras poligonales ya anotadas: son del dominio
exacto por construcción, y permitirían multiplicar la clase minoritaria sin anotar más.

**3. El detector sobre-detecta** (precisión 0.297 con recall 1.000). Subir el umbral de
confianza costaría el recall que la clase minoritaria no tiene margen de perder; la vía
correcta es endurecer la persistencia exigida en el tracker y medir el efecto sobre las
alertas espurias.

**4. El sesgo sistemático de la altura** (factor ~4) tiene tres hipótesis planteadas en
la sección 6.2 y ninguna descartada.

**5. Falta el segmentador neural del pretil.** Con SAM2 promptado por el camino óptimo
del Método 1 —los métodos apoyándose uno en otro— el eje del benchmark pasaría de
comparar dos formulaciones clásicas a comparar prior geométrico contra representación
aprendida, que es el contraste que el enunciado propone.

**6. La segmentación del pretil es el cuello de botella** con GPU (36 ms/frame a 720p
y 74 ms a 1080p, frente a 16 y 29 ms de la detección). Llevar la búsqueda de camino óptimo a GPU, o ejecutarla a resolución
reducida e interpolar, es la única optimización con retorno real.

**7. Exportación a ONNX Runtime** y medición del speedup frente a PyTorch en FP32 y
FP16, que el enunciado valora y que no se llegó a medir.

**8. Los umbrales están calibrados sobre cuatro videos.** Los cortes de clasificación
lumínica, el umbral de detección de cortes y el campo de visión asumido se ajustaron
sobre el material de muestra. El conjunto de evaluación es ciego: su generalización es un
supuesto declarado, no un hecho verificado. Todos son configurables en `configs/`.

**9. Anotador único, sin acuerdo entre anotadores.** Las métricas de detección arrastran
una incertidumbre propia que no se ha cuantificado.
