# Resultados del detector

Resultados del fine-tuning (ADR 0002) con los datos del ADR 0004.

```bash
python scripts/split_dataset.py data/export --out data/dataset --val video_04
python scripts/train_detector.py --data data/dataset/data.yaml --epochs 80
python scripts/eval_detector.py
```

Las curvas y métricas del entrenamiento están en `docs/entrenamiento/`, y los pesos en
`weights/detector_v1.pt`.

## 1. Datos

171 frames anotados a mano, estratificados por condición de luz y separados por video.

| Partición | Imágenes | `caex` | `bulldozer` |
|---|---|---|---|
| train (`video_01`, `02`, `03`) | 131 | 270 | 33 |
| val (`video_04` completo) | 40 | 39 | 11 |
| **Total** | **171** | **309** | **44** |

**Hay 7 CAEX por cada bulldozer**, y solo 44 bulldozers en total. Viene del material: en
`video_01` hay cuatro o cinco CAEX a la vez y un solo bulldozer, y en los otros clips el
bulldozer se ve más chico, tapado por el polvo o fuera de cuadro.

Separé por video y no al azar porque dos frames a seis posiciones de distancia son casi la
misma imagen, y con una partición aleatoria la validación premiaría memorizar. Dejé
`video_04` para validación y no `video_03` porque este solo tiene 4 bulldozers: con 4
objetos, fallar uno mueve el resultado un 25 %.

## 2. Entrenamiento

Desde `yolo11s.pt` (COCO), con un máximo de 80 épocas, `batch=8`, `imgsz=640`, precisión
mixta y semilla 0. Se detuvo por paciencia en la época 42 y la mejor fue la 22. Cerca de la
época 6 el mAP cayó a 0.002 y se recuperó, algo típico con pocos datos para el tamaño del
modelo.

## 3. Métricas

| Clase | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---|---|---|---|
| `caex` | 0.812 | 0.585 | 0.297 | **1.000** |
| `bulldozer` | **0.306** | 0.188 | 0.464 | 0.455 |
| global | 0.559 | 0.387 | 0.381 | 0.727 |

El mAP global de 0.559 es moderado (un detector bien entrenado con muchos datos anda entre
0.80 y 0.90), pero promedia dos cosas muy distintas: `caex` en 0.812, que es usable, y
`bulldozer` en 0.306, que no lo es. Si reportara solo el global, escondería el problema
principal.

### La matriz de confusión y el mAP no se contradicen

La matriz de confusión (`docs/entrenamiento/confusion_matrix_normalized.png`) muestra la
fila de `bulldozer` vacía, como si nunca se predijera, pero el mAP es 0.306. La diferencia
es el umbral: la matriz usa un umbral de confianza fijo y el mAP recorre todos. El modelo sí
predice `bulldozer`, pero casi siempre bajo 0.25, el umbral de `configs/method_1.yaml`, así
que en operación casi no aparece. La clase está aprendida, aunque débil.

### Precisión baja en `caex`

`caex` tiene recall 1.000 y precisión 0.297: encuentra todos los camiones, pero ~70 % de sus
cajas no corresponden a ninguno. Con el NMS por defecto (0.7) salen cuatro cajas solapadas
por escena. El `iou=0.45` de la configuración quita buena parte, pero quedan falsos
positivos sobre terreno y polvo.

Cada falso positivo es un equipo fantasma para la proximidad, o sea, una posible falsa
alarma. El tracker lo mitiga exigiendo 3 detecciones seguidas antes de confirmar un track y
suprimiendo cajas duplicadas (reporte, sección 7).

## 4. Aun así, resolvió el problema

El fine-tuning no se hizo por la etiqueta, sino porque el modelo COCO juntaba el CAEX y el
bulldozer en una caja, y sin dos entidades no hay distancia que medir. En el frame 120 de
`video_02`:

| Modelo | Detecciones | Cajas |
|---|---|---|
| `yolo11s` COCO | 1 | `truck` 0.74, x=[138,759], **622 px, las dos máquinas** |
| `detector_v1` | 2 | `caex` 0.48, x=[480,765], 285 px, **el bulldozer**<br>`caex` 0.43, x=[65,614], 549 px, **el CAEX** |

Las máquinas quedan separadas. Con 33 ejemplos el modelo no aprendió bien la clase
`bulldozer`, pero entrenar con cajas separadas le enseñó a no juntar máquinas vecinas, y eso
era lo que bloqueaba. Las dos cajas salen como `caex`, lo que calza con lo anterior.

## 5. Limitaciones

- **`bulldozer` es débil** (mAP@0.5 de 0.306, recall 0.455), y en operación casi todo sale
  como `caex`. Las etiquetas de clase de la salida no identifican el tipo de equipo.
- **`caex` sobre-detecta** (precisión 0.297).
- Las dos cosas vienen del desbalance: 33 bulldozers contra 270 CAEX. El modelo aprende mal
  la clase chica y, ante la duda, apuesta por la grande, lo que sube el recall y baja la
  precisión.
- Los 44 bulldozers y 118 de los 309 CAEX los anoté con polígono en vez de caja.
  Ultralytics usa la caja envolvente, así que para detección no se pierde nada, pero
  conviene usar una sola herramienta si se amplía el conjunto.
- Anoté yo solo, sin acuerdo entre anotadores.

Cómo mejorarlo, de más a menos rentable:

1. **Anotar más bulldozers**, extrayendo frames solo de los tramos donde aparece. Unas
   sesenta anotaciones más duplicarían la clase en cerca de una hora.
2. **Ponderar la pérdida por clase o sobremuestrear** los frames con bulldozer. Es más
   barato, pero con menos efecto.
3. **Preentrenar con un dataset externo y afinar con los datos propios.** Los del ADR 0004
   se descartaron como fuente principal, pero esta combinación sigue siendo una hipótesis
   por probar.
