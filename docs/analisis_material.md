# Análisis del material de entrada

Mediciones del material de muestra, hechas antes de diseñar el pipeline:

```bash
python scripts/probe_videos.py data/raw
python scripts/probe_detector.py data/raw --stride 12
python scripts/probe_openvocab.py --conf 0.05
```

Los supuestos habituales (misma resolución, toma continua, luz constante por archivo) no se
cumplen, y cada uno tiene una consecuencia en el diseño.

## 1. Formato

| Video | Resolución | FPS | Frames | Luminancia mín / máx / media |
|---|---|---|---|---|
| `video_01` | 1920×1080 | 30 | 302 | 49 / 136 / 102 |
| `video_02` | 1280×720 | 24 | 240 | 23 / 163 / 82 |
| `video_03` | 1280×720 | 24 | 240 | 19 / 154 / 79 |
| `video_04` | 1280×720 | 24 | 240 | 28 / 147 / 76 |

La resolución y los FPS cambian entre videos, así que el pipeline no puede asumir ninguno.
El detector trabaja a un tamaño fijo con letterbox, y `metadata.json` guarda la resolución
original y la analizada.

Son unos 1.022 frames en total. Una corrida completa toma minutos incluso en CPU, así que
pude priorizar la calidad de la medición por sobre la velocidad.

## 2. Estructura temporal

`video_01` tiene tres tomas separadas por cortes, y la cámara cambia de posición entre
ellas:

| Instante | Frame | Luminancia antes → después |
|---|---|---|
| 1.50 s | 45 | 119 → 49 |
| 7.13 s | 214 | 134 → 94 |
| 7.17 s | 215 | 94 → 67 |

`video_02`, `video_03` y `video_04` no tienen cortes, pero pasan gradualmente de noche a día
con polvo y de vuelta a noche en diez segundos.

Son dos casos distintos:

- **Corte**: invalida los tracks, la estimación del suelo y los filtros temporales. Hay que
  reiniciar el estado.
- **Transición gradual**: no invalida el estado, pero en los extremos de luz el detector y
  el pretil empeoran.

## 3. Condición de luz

Clasificación por frame según la luminancia media (`night` < 70, `day` > 110, `dusk` entre
ambos):

| Video | night | dusk | day |
|---|---|---|---|
| `video_01` | 30 % | 12 % | 58 % |
| `video_02` | 53 % | 7 % | 40 % |
| `video_03` | 53 % | 13 % | 34 % |
| `video_04` | 57 % | 10 % | 34 % |

Más de la mitad de los clips de 720p es de noche, así que un promedio global quedaría
dominado por la noche. Como la luz cambia dentro de cada archivo, el benchmark se desglosa
por frame y no por video. La misma clasificación ajusta el realce de contraste del pretil.

## 4. Detector preentrenado (YOLO11s, COCO)

Clases detectadas, tomando un frame de cada doce:

| Video | Clase principal | Detecciones | Confianza media | Por frame |
|---|---|---|---|---|
| `video_01` | `truck` | 56 | 0.61 | 2.15 |
| `video_02` | `truck` | 20 | 0.69 | 1.00 |
| `video_03` | `truck` | 21 | 0.55 | 1.05 |
| `video_04` | `truck` | 18 | 0.58 | 0.90 |

Las demás clases (`car`, `airplane`, `kite`, `person`) aparecen con confianza de 0.28 a
0.41 y menos de 0.3 veces por frame: son falsos positivos sobre polvo y sobre el horizonte.
`bus` y `train` no aparecen nunca, así que las saqué del mapeo.

### Las máquinas se fusionan

Un `truck` por frame en `video_02` a `video_04` parece correcto, pero no lo es. En el frame
120 de `video_02`, con un CAEX y un bulldozer separados:

```
imgsz=640,  conf>=0.20  ->  1 detección
    truck  0.74  x=[138,759]  ancho=622px
```

Una sola caja cubre las dos máquinas. Subir la resolución no lo arregla:

```
imgsz=1280, conf>=0.15  ->  truck 0.58 x=[127,742] | truck 0.15 x=[131,551]
imgsz=1920, conf>=0.10  ->  truck 0.51 x=[132,550] | truck 0.32 x=[261,817]
```

Esto afecta a tres módulos:

1. **Proximidad**: sin dos entidades no hay distancia que medir.
2. **Vista en planta**: el punto de contacto de la caja fusionada no es el de ninguna
   máquina real.
3. **Escala**: la caja fusionada no mide lo que mide un CAEX, así que la escala
   píxel-metro que sale de ella no sirve.

## 5. Detector zero-shot (YOLO-World)

`yolov8s-worldv2` sobre el mismo frame, con `imgsz=1280`.

Con las dos máquinas en escena:

| Prompts | Detecciones | Caja |
|---|---|---|
| `["truck", "bulldozer"]` | 1 | `truck` 0.56, x=[130,814], 683 px |
| `["mining haul truck", "bulldozer", "excavator"]` | 1 | 0.34, x=[136,783], 648 px |
| `["large yellow mining dump truck", "yellow tracked bulldozer with blade"]` | 1 | 0.20, x=[141,834], 693 px |

Siempre una caja con las dos máquinas, y menos confianza mientras más específico es el
prompt.

Con umbral 0.03 aparecen las cajas correctas, pero pierden:

```
truck  0.562  x=[130,814]  ancho=683   <- fusionada
truck  0.141  x=[135,555]  ancho=420   <- solo el CAEX
truck  0.089  x=[515,763]  ancho=248   <- el bulldozer
```

La del CAEX la elimina el NMS por solaparse con la fusionada, y la del bulldozer queda en
0.089, al nivel del ruido.

Con el bulldozer solo, sin otra clase:

```
["bulldozer"]                 conf>=0.05  ->  0 detecciones
["tracked bulldozer"]         conf>=0.05  ->  0 detecciones
["crawler dozer with blade"]  conf>=0.05  ->  0 detecciones
["construction vehicle"]      conf>=0.05  ->  1 detección, otra vez fusionada
```

Hay dos fallas distintas. Que gane la caja fusionada se podría intentar arreglar con
umbrales o NMS, aunque quedaría frágil. Que el prompt del bulldozer no active nada no se
arregla con hiperparámetros. El texto guía al modelo, pero las propuestas de región las
sigue decidiendo la parte visual, y si esta propone una sola masa el prompt no la separa.

La causa de fondo es la brecha de dominio: dos máquinas ocres, superpuestas, sobre suelo
ocre, con polvo y a media distancia, cuando estos modelos aprendieron con maquinaria
fotografiada de cerca y bien iluminada. Por eso descarté el zero-shot (ADR 0002).
