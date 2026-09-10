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

La matriz de confusión normalizada sobre el conjunto de validación:

| | predicho `caex` | predicho `bulldozer` | no detectado |
|---|---|---|---|
| **`caex` real** | 0.97 | 0.00 | 0.03 |
| **`bulldozer` real** | 0.45 | **0.00** | 0.55 |

**El modelo no predice la clase `bulldozer` en ninguna ocasión.** La fila entera está vacía.
Un bulldozer real se etiqueta como `caex` el 45 % de las veces y se pierde el 55 % restante.

El mAP global de 0.559 es por tanto engañoso: describe un sistema que resuelve bien una clase
e ignora por completo la otra. Reportar sólo esa cifra habría ocultado exactamente el problema
que este entrenamiento existía para resolver.

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
alcanzaran para aprender una categoría visual nueva. Localización y clasificación se
aprendieron de forma desigual, y el problema bloqueante era el primero.

Una observación práctica: con el umbral de NMS por defecto de Ultralytics (0.7) el modelo emite
cuatro cajas solapadas sobre la misma escena. Con el `iou=0.45` que fija `configs/method_1.yaml`
quedan exactamente las dos correctas. El valor de configuración importa tanto como los pesos.

## 6. Estado y limitaciones declaradas

**Lo que funciona.** Detección de CAEX con recall de 0.97 sobre validación. Separación de
máquinas contiguas en cajas independientes.

**Lo que no.** La clasificación `caex` / `bulldozer` no es utilizable: toda detección se emite
como `caex`. El OSD y los artefactos reflejan esa etiqueta, y no debe interpretarse como una
identificación de tipo de equipo.

**Causa identificada.** 33 instancias de bulldozer en entrenamiento, frente a 270 de CAEX. Es
un problema de datos, no de arquitectura ni de hiperparámetros.

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
