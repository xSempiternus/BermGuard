# Análisis del material de entrada

Caracterización medida del material de muestra, realizada antes de diseñar el pipeline.
Todos los números de este documento se reproducen con:

```bash
python scripts/probe_videos.py data/raw
python scripts/probe_detector.py data/raw --stride 12
python scripts/probe_openvocab.py --conf 0.05
```

El propósito es explícito: los supuestos habituales sobre material de video —resolución
uniforme, toma continua, condición lumínica constante por archivo— **no se cumplen acá**, y
cada incumplimiento tiene una consecuencia de diseño concreta.

---

## 1. Propiedades de contenedor

| Video | Resolución | FPS | Frames | Luminancia min / máx / media |
|---|---|---|---|---|
| `video_01` | 1920×1080 | 30 | 302 | 49 / 136 / 102 |
| `video_02` | 1280×720 | 24 | 240 | 23 / 163 / 82 |
| `video_03` | 1280×720 | 24 | 240 | 19 / 154 / 79 |
| `video_04` | 1280×720 | 24 | 240 | 28 / 147 / 76 |

**Resolución y tasa de frames no son uniformes.** El pipeline no puede fijar ninguna de las
dos. La inferencia se realiza a un tamaño canónico mediante letterbox, y ese tamaño se
registra en `metadata.json` junto a la resolución de origen.

**El material total son ~1.022 frames.** El costo computacional no es el eje del diseño: una
corrida completa es cuestión de minutos incluso en CPU. Esto descarta la necesidad de un
perfil de ejecución reducido y permite priorizar calidad de medición por sobre throughput.

---

## 2. Estructura temporal

`video_01` **no es una toma continua**. Contiene tres tomas separadas por cortes duros:

| Instante | Frame | Luminancia antes → después |
|---|---|---|
| 1.50 s | 45 | 119 → 49 |
| 7.13 s | 214 | 134 → 94 |
| 7.17 s | 215 | 94 → 67 |

La cámara además se reposiciona entre tomas.

`video_02`, `video_03` y `video_04` no tienen cortes duros, pero recorren una transición
**gradual** noche → día pleno con polvo → noche dentro de los mismos diez segundos.

Estos son dos regímenes distintos y ambos deben manejarse:

- **Corte duro** → invalida identidades de tracking, calibración del plano de suelo y
  cualquier filtro temporal. Requiere reinicio de estado.
- **Transición gradual** → no invalida el estado, pero degrada al detector en ambos extremos
  del rango dinámico. Requiere acondicionamiento adaptativo del frame.

Un sistema que sólo contemple uno de los dos falla en la mitad del material.

---

## 3. Condición lumínica

Clasificación por frame sobre la luminancia media (`night` < 70, `day` > 110, `dusk` entre
ambos):

| Video | night | dusk | day |
|---|---|---|---|
| `video_01` | 30 % | 12 % | 58 % |
| `video_02` | 53 % | 7 % | 40 % |
| `video_03` | 53 % | 13 % | 34 % |
| `video_04` | 57 % | 10 % | 34 % |

**Más de la mitad del material de los tres clips de 720p es nocturno.** Un promedio global de
cualquier métrica quedaría dominado por el caso nocturno mientras aparenta describir el
sistema completo.

Y la condición cambia *dentro* de un mismo archivo, por lo que una etiqueta a nivel de archivo
no significa nada. La estratificación del benchmark es **por frame**, no por video.

Consecuencia favorable: la clasificación por frame que exige el reporte es el mismo
mecanismo que alimenta el acondicionamiento adaptativo del pipeline. Un solo desarrollo
cubre ambos requisitos.

---

## 4. Detección de maquinaria con modelos preentrenados

### 4.1 YOLO11s sobre COCO

Histograma de clases, muestreando un frame de cada doce:

| Video | Clase dominante | Aciertos | Confianza media | Por frame |
|---|---|---|---|---|
| `video_01` | `truck` | 56 | 0.61 | 2.15 |
| `video_02` | `truck` | 20 | 0.69 | 1.00 |
| `video_03` | `truck` | 21 | 0.55 | 1.05 |
| `video_04` | `truck` | 18 | 0.58 | 0.90 |

El resto de las clases activadas (`car`, `airplane`, `kite`, `person`) aparecen con
confianzas de 0.28 a 0.41 y frecuencias por debajo de 0.3 por frame: son falsos positivos
sobre penachos de polvo y sobre el horizonte.

Las clases `bus` (5) y `train` (6), incluidas preventivamente en el mapeo inicial, **no se
activan nunca**. Se eliminan del mapeo.

### 4.2 El hallazgo que condiciona el diseño

El conteo de `truck` cercano a 1.0 por frame en `video_02` a `video_04` es engañoso. La
inspección visual del frame 120 de `video_02`, donde hay un CAEX y un bulldozer claramente
separados, muestra el problema:

```
imgsz=640,  conf>=0.20  ->  1 deteccion
    truck  0.74  x=[138,759]  ancho=622px
```

**Una sola caja engloba ambas máquinas.** El bulldozer no se pierde: se fusiona.

Aumentar la resolución de inferencia no lo resuelve, produce cajas superpuestas e
inestables:

```
imgsz=1280, conf>=0.15  ->  truck 0.58 x=[127,742] | truck 0.15 x=[131,551]
imgsz=1920, conf>=0.10  ->  truck 0.51 x=[132,550] | truck 0.32 x=[261,817]
```

### 4.3 Por qué esto invalida tres módulos

1. **Proximidad.** Medir la distancia entre equipos requiere dos entidades. Con una caja
   fusionada no hay par que comparar.
2. **Homografía.** El punto de contacto con el suelo de la caja fusionada no corresponde al
   de ninguna máquina real, así que la proyección a vista cenital devuelve una posición
   inventada.
3. **Ancla métrica.** La altura de la caja fusionada no es la altura del CAEX, de modo que la
   escala píxel-metro derivada de ella carece de significado.

No es un problema de precisión de detección. Es un problema que se propaga a la analítica.

---

## 5. Detección open-vocabulary (zero-shot)

Se evaluó YOLO-World (`yolov8s-worldv2`) sobre el mismo frame, con prompts de especificidad
creciente, `imgsz=1280`.

### 5.1 Con ambas máquinas compitiendo

| Prompts | Detecciones | Caja |
|---|---|---|
| `["truck", "bulldozer"]` | 1 | `truck` 0.56, x=[130,814], 683 px |
| `["mining haul truck", "bulldozer", "excavator"]` | 1 | 0.34, x=[136,783], 648 px |
| `["large yellow mining dump truck", "yellow tracked bulldozer with blade"]` | 1 | 0.20, x=[141,834], 693 px |

Siempre una sola caja, siempre englobando ambas máquinas, y con la confianza **cayendo** a
medida que el prompt se vuelve más específico.

### 5.2 Aislamiento de la causa

Bajando el umbral a 0.03 con ambos prompts, la descomposición correcta **sí existe** en el
conjunto de propuestas:

```
truck  0.562  x=[130,814]  ancho=683   <- fusionada
truck  0.141  x=[135,555]  ancho=420   <- el CAEX solo
truck  0.089  x=[515,763]  ancho=248   <- el bulldozer
```

Pero pierde. La caja del CAEX se suprime por NMS al solaparse fuertemente con la fusionada;
la del bulldozer sobrevive al NMS pero queda en 0.089, indistinguible del ruido.

Y con el bulldozer aislado, sin ninguna clase que le compita:

```
["bulldozer"]                 conf>=0.05  ->  0 detecciones
["tracked bulldozer"]         conf>=0.05  ->  0 detecciones
["crawler dozer with blade"]  conf>=0.05  ->  0 detecciones
["construction vehicle"]      conf>=0.05  ->  1 deteccion, nuevamente fusionada
```

### 5.3 Interpretación

Son **dos fallos independientes**, y la distinción determina si el enfoque es recuperable:

- **La caja fusionada gana el ranking.** En teoría atacable ajustando umbrales o el NMS,
  aunque el resultado sería frágil.
- **El embedding de texto no activa.** Sin competencia alguna y con el umbral casi en el
  suelo, tres redacciones distintas devuelven cero. Esto **no es ajustable por
  hiperparámetros**: si la representación textual no coincide con la visual, ningún umbral
  la hace aparecer.

El vocabulario abierto condiciona las features mediante el texto, pero la propuesta de
región sigue dominada por el backbone visual. Cuando el backbone propone una masa única, el
prompt no la separa.

La causa de fondo es brecha de dominio: dos máquinas ocres, parcialmente superpuestas, sobre
suelo ocre, con polvo en suspensión y a media distancia. La distribución de entrenamiento de
un modelo vision-language contiene maquinaria de construcción fotografiada de cerca y bien
iluminada.

**Conclusión:** el enfoque zero-shot queda descartado para detección de maquinaria, con la
causa aislada y medida. Ver `docs/adr/0002-deteccion-de-maquinaria.md`.
