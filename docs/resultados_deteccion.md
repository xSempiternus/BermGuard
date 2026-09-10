# Resultados del detector especializado

Análisis del fine-tuning decidido en el ADR 0002 y ejecutado con los datos del ADR 0004.

Reproducible con:

```bash
python scripts/split_dataset.py data/export --out data/dataset --val video_04
python scripts/train_detector.py --data data/dataset/data.yaml --epochs 80
```

Artefactos en `docs/entrenamiento/`. Pesos entregados en `weights/detector_v1.pt`.

---

## 1. Conjunto de entrenamiento

171 frames anotados a mano, extraídos del material de muestra con muestreo estratificado por
condición lumínica y separados por video para la partición.

| Partición | Imágenes | `caex` | `bulldozer` |
|---|---|---|---|
| train (`video_01`, `02`, `03`) | 131 | 270 | 33 |
| val (`video_04` completo) | 40 | 39 | 11 |
| **Total** | **171** | **309** | **44** |

**El desbalance de clases es de 7:1, y la clase minoritaria tiene 44 instancias en todo el
conjunto.** No es una consecuencia del muestreo sino del material: en `video_01` hay cuatro o
cinco CAEX simultáneos y un solo bulldozer, y en los otros clips el bulldozer aparece más
pequeño, más ocluido por el polvo y con frecuencia fuera de cuadro.

La partición es por video y no aleatoria. Con partición aleatoria, dos frames separados por
seis posiciones —misma escena, misma luz, máquinas casi en la misma posición— caerían a lados
distintos, y la métrica de validación premiaría la memorización. Reservar un clip íntegro mide
lo que interesa: el comportamiento sobre material no visto.

La validación se asignó a `video_04` y no a `video_03` tras comprobar que este último aporta
sólo 4 instancias de bulldozer. Un mAP calculado sobre 4 objetos no es una medición: fallar uno
lo mueve un 25 %.

## 2. Entrenamiento

Partiendo de `yolo11s.pt` (preentrenado en COCO), 80 épocas solicitadas, `batch=8`,
`imgsz=640`, precisión mixta, semilla fija en 0.

El entrenamiento se detuvo por paciencia en la **época 42**, con el mejor resultado en la
**época 22**. La curva muestra un colapso transitorio alrededor de la época 6 —el mAP cae a
0.002 y se recupera— característico de un conjunto pequeño frente a la capacidad del modelo.

## 3. Métricas globales

| Métrica | Valor |
|---|---|
| mAP@0.5 | 0.559 |
| mAP@0.5:0.95 | 0.388 |

Para calibrar: un detector bien entrenado sobre un conjunto amplio y limpio se sitúa entre
0.80 y 0.90 de mAP@0.5. El resultado es moderado y consistente con 131 imágenes de
entrenamiento.

## 4. El resultado que el promedio esconde

El desglose por clase sobre el conjunto de validación:

| Clase | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---|---|---|---|
| `caex` | 0.812 | 0.585 | 0.297 | **1.000** |
| `bulldozer` | **0.306** | 0.188 | 0.464 | 0.455 |

**El mAP global de 0.559 describe dos comportamientos muy distintos promediados.** La clase
mayoritaria alcanza 0.812, un valor utilizable; la minoritaria se queda en 0.306, que no lo es.
Reportar sólo la cifra global habría ocultado exactamente el problema que este entrenamiento
existía para resolver.

### Una discrepancia que conviene explicar

La matriz de confusión que genera el entrenamiento
(`docs/entrenamiento/confusion_matrix_normalized.png`) muestra la fila de `bulldozer`
completamente vacía, como si el modelo nunca predijera esa clase. El mAP de 0.306 dice que sí
la predice.

Ambas cosas son ciertas y la diferencia está en el umbral. La matriz de confusión se calcula a
un **umbral de confianza fijo**, mientras el mAP integra la curva precisión-recall **sobre todos
los umbrales**. La lectura conjunta es que el modelo emite predicciones de `bulldozer`, pero
casi todas por debajo del umbral de operación: a 0.25 —el valor que fija
`configs/method_1.yaml`— prácticamente desaparecen.

Es una distinción con consecuencia práctica: la clase está aprendida, aunque débilmente, y no
ausente. Bajar el umbral la haría aparecer, a costa de los falsos positivos que se describen
abajo. Y es un recordatorio de que una métrica agregada y una matriz de confusión no responden
la misma pregunta.

### El problema de precisión, que no estaba previsto

`caex` tiene **recall 1.000 con precisión 0.297**: el modelo encuentra todos los camiones del
conjunto de validación, pero cerca del 70 % de sus detecciones no corresponde a ninguno.

Sobre-detecta. Es coherente con lo observado al inspeccionar frames sueltos: con el umbral de
NMS por defecto de Ultralytics (0.7) el modelo emite cuatro cajas solapadas sobre la misma
escena. El `iou=0.45` de la configuración recorta buena parte de esa duplicación, pero la
precisión medida indica que quedan falsos positivos sobre terreno y polvo.

Tiene una consecuencia aguas abajo que hay que declarar: **cada falso positivo es una entidad
fantasma para el módulo de proximidad**, y por tanto una fuente de alertas espurias. Un
sistema de seguridad que alerta sin causa produce fatiga de alarma, que es el mismo problema
que la histéresis existe para evitar. Mitigarlo pasa por subir el umbral de confianza —a costa
del recall de `bulldozer`, ya frágil— o por exigir persistencia temporal en el tracker antes de
considerar un track como equipo real.

## 5. Y sin embargo, el entrenamiento resolvió el bloqueo

La razón de ser del fine-tuning (ADR 0002) no era la etiqueta, sino que el modelo COCO
**fusionaba el CAEX y el bulldozer en una sola caja**, lo que elimina la magnitud que el módulo
de proximidad necesita medir.

Sobre el frame 120 de `video_02`, donde ambas máquinas aparecen contiguas:

| Modelo | Detecciones | Cajas |
|---|---|---|
| `yolo11s` COCO | 1 | `truck` 0.74, x=[138,759], **622 px — ambas máquinas** |
| `detector_v1` | 2 | `caex` 0.48, x=[480,765], 285 px — **el bulldozer**<br>`caex` 0.43, x=[65,614], 549 px — **el CAEX** |

**Las máquinas quedan separadas.** El sistema pasa de una entidad a dos, que es la condición
necesaria para medir distancia entre equipos.

La interpretación es que el entrenamiento sobre datos del dominio, con cajas anotadas
separadas, enseñó al modelo a **no fusionar máquinas adyacentes**, aunque 33 instancias no
alcanzaran para aprender una categoría visual sólida. **Localización y clasificación se
aprendieron de forma muy desigual**, y el problema bloqueante era el primero: la separación en
entidades independientes es lo que el módulo de proximidad necesita, y la etiqueta correcta es
deseable pero no imprescindible para medir una distancia.

Nótese que en este frame ambas cajas salen como `caex`, incluida la del bulldozer. Es
consistente con la sección 4: al umbral de operación de 0.25 la clase minoritaria casi no se
emite.

## 6. Estado y limitaciones declaradas

**Lo que funciona.** Recall perfecto sobre `caex` en el conjunto de validación (1.000) y
mAP@0.5 de 0.812 para esa clase. Separación de máquinas contiguas en cajas independientes,
que era el bloqueo del pipeline.

**Lo que no, y en qué grado.**

- **La clase `bulldozer` está aprendida pero es frágil**: mAP@0.5 de 0.306, recall 0.455. Al
  umbral de operación de 0.25 casi no se emite, de modo que en la práctica el OSD etiqueta
  toda máquina como `caex`. Las etiquetas de clase de los artefactos **no deben interpretarse
  como identificación de tipo de equipo**.
- **La precisión sobre `caex` es baja** (0.297): el modelo sobre-detecta, y cada falso positivo
  se convierte en una entidad fantasma para el módulo de proximidad y por tanto en una alerta
  espuria potencial.

**Causa identificada.** 33 instancias de bulldozer en entrenamiento, frente a 270 de CAEX. Es
un problema de datos, no de arquitectura ni de hiperparámetros. El desbalance explica ambos
síntomas: la clase minoritaria se aprende mal, y el modelo aprende a apostar por la mayoritaria
ante la duda, lo que infla el recall a costa de la precisión.

**Nota sobre el conjunto exportado.** La validación advirtió `len(segments)=18, len(boxes)=50`:
dieciocho anotaciones se hicieron con herramienta de polígono en lugar de caja. Ultralytics
descarta los polígonos y usa las cajas envolventes, de modo que no hay pérdida de información
para detección, pero conviene homogeneizar la herramienta de anotación si el conjunto se amplía.

**Camino de corrección, en orden de coste-beneficio:**

1. **Etiquetado dirigido.** Extraer frames restringidos a los tramos donde el bulldozer está
   visible, reduciendo la separación mínima entre muestras, y anotar unas sesenta más. Se
   estima que duplicaría la clase minoritaria en cerca de una hora de trabajo.
2. **Pérdida ponderada por clase o sobremuestreo** de los frames con bulldozer, para compensar
   el desbalance sin datos nuevos. Más barato, y de efecto más limitado.
3. **Datos externos.** Los tres datasets públicos evaluados en el ADR 0004 se descartaron por
   brecha de dominio. Un preentrenamiento sobre ellos seguido de afinado sobre los datos
   propios sigue siendo una hipótesis medible, no descartada.

**Métrica sin cuantificar.** El conjunto lo anotó una sola persona, sin acuerdo entre
anotadores, de modo que estas cifras arrastran una incertidumbre propia que no se ha medido.
