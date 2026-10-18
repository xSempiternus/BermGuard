# ADR 0004: Entrenar con frames del propio material en vez de datasets públicos

- Estado: aceptada
- Fecha: 2026-09-08
- Depende de: ADR 0002

## Contexto

El ADR 0002 decidió entrenar un detector. Faltaba decidir con qué datos.

El enunciado sugiere datasets públicos y nombra Roboflow Universe. Evalué tres, fijándome
sobre todo en qué tan parecidos son a este material, porque lo que hizo fallar a YOLO-World
fue justamente la distancia entre sus datos de entrenamiento y estos videos.

| Dataset | Clases | Imágenes | Licencia | Problema |
|---|---|---|---|---|
| `brancos-workspace/mining-truck-detection-w53tl` | 1 (`haul_truck`) | 264 | CC BY 4.0 | No tiene bulldozer. Solo aporta la clase que el modelo base ya detecta, y parte de las imágenes son aumentos de un mismo frame |
| `mohamads-workspace-rl4ui/bulldozer-lpoe2` | 7, incluye `Bull_dozer` | 1.324 | CC BY 4.0 | Tiene las clases, pero son fotos de catálogo: la máquina llena el encuadre, bien iluminada y sin oclusión |
| `rf100-vl/apoce-...-construction-equipment` | 7, incluye `bulldozer` | 928 | MIT | Fotos cenitales de dron sobre obras urbanas, con objetos muy pequeños |

Ninguno se parece a estos videos: vista oblicua desde altura, distancia media, desierto con
polvo y más de la mitad de los frames de noche.

## Decisión

**Entrenar con frames anotados del propio material.** Los extraje con
`scripts/prepare_labeling_set.py`, estratificados por condición de luz, y anoté 171.

Creo que ese volumen alcanza por dos razones:

- El entrenamiento parte de pesos preentrenados, que ya saben de bordes, texturas y
  vehículos. Lo que falta enseñar es cómo se ven estas dos máquinas desde esta cámara.
- El material es homogéneo: misma faena, mismas máquinas y encuadres parecidos. En otro
  contexto sería un riesgo de sobreajuste, pero lo más probable es que el conjunto de
  evaluación venga del mismo lugar.

**Las clases son `caex` y `bulldozer`**, que son roles en el botadero. Los datasets
públicos etiquetan `truck` o `vehicle`, que no es la distinción que necesito.

## Alternativas descartadas

- **`bulldozer-lpoe2` como fuente principal.** Enseñaría a esperar máquinas grandes,
  centradas y bien iluminadas, lo contrario de este material. Queda como experimento:
  preentrenar con él y afinar con los frames propios.
- **APOCE.** Una vista cenital y una oblicua dan siluetas distintas de la misma máquina.
- **Juntar los tres datasets sin datos propios.** Más volumen no corrige el sesgo: 2.500
  imágenes fuera de dominio siguen fuera de dominio.

## Consecuencias

- Anotar me tomó cerca de dos horas.
- El modelo queda muy especializado y no va a generalizar a otras faenas ni a otros
  ángulos de cámara.
- La validación **no puede ser aleatoria**. Frames del mismo video son casi iguales, y una
  partición aleatoria mediría memoria y no generalización. Separé por video y dejé
  `video_04` completo para validación.

## Revisión (2026-10-18): preentrenar con datos públicos, como variante medida

Volví a buscar datasets públicos, ahora para el experimento que quedó abierto: preentrenar
con datos públicos y afinar con los frames propios. Revisé las imágenes de cada uno en
Roboflow Universe y conté, con el filtro por clase, cuántas tienen bulldozer:

| Dataset | Imágenes con bulldozer | Vista | Veredicto |
|---|---|---|---|
| [`mines-project/mines`](https://universe.roboflow.com/mines-project/mines-jo13g) (CC BY 4.0) | **68 de 261** (~75 dozers), más `dumper` | Fotos en una mina a cielo abierto, a media distancia, varias fechas entre 2022 y 2024; **polígonos** | **Usar.** El más parecido en escena y máquinas |
| [`bulldozer-lpoe2`](https://universe.roboflow.com/mohamads-workspace-rl4ui/bulldozer-lpoe2) (CC BY 4.0) | **1.324 de 1.324** | Mezcla de catálogo y fotos de faena a media distancia | **Usar.** Volumen y variedad de la forma del bulldozer |
| [`ccl-kl4nk/dji_65`](https://universe.roboflow.com/ccl-kl4nk/dji_65) (CC0) | 19 de 136 (20 dozers) | Dron oblicuo sobre una mina | **Descartado.** Los 19 frames son casi seguidos de un mismo vuelo (272–290 s y 212–219 s): en la práctica, una sola máquina |
| [`ws1/m`](https://universe.roboflow.com/ws1-i1agt/m-oi9pf) (CC BY 4.0) | 102 (+153 `dozer_night`) | Cámara de tablero de un camión minero, a nivel del suelo | **Descartado.** Otra perspectiva, dozers de pocos píxeles, y las nocturnas son versiones sintéticas de las diurnas (`*_day2night.jpg`) |
| [`rockaware/conference`](https://universe.roboflow.com/rockaware/conference-x8lzn) | — | Maquetas de juguete en una oficina | **Descartado** |
| [`minitruck/mining-truck-v1.4`](https://universe.roboflow.com/minitruck/mining-truck-v1.4) | 0 | Cámara fija, CAEX en primer plano | Sólo CAEX, que ya se detecta bien |
| [`igensol/mines-vehicles`](https://universe.roboflow.com/igensol/mines-vehicles) | 0 | CCTV nocturno en una mina | Sin bulldozer |
| [mineCareerDB](https://data.mendeley.com/datasets/c5s76mj4bm/5) (Mendeley) | 0 | Dron cenital a ~400 m | Sin bulldozer |

Corrijo lo que dije de `bulldozer-lpoe2` en el contexto: no son sólo fotos de catálogo,
también hay muchas de faena con varias máquinas a media distancia.

Sigue sin existir un dataset con este encuadre, y la conclusión sobre la fuente principal
no cambia. Pero el preentrenamiento no necesita el mismo dominio: sirve si le enseña al
modelo la forma de un bulldozer, que es la clase con 33 instancias propias. La pregunta es
empírica, así que queda como una tercera variante de la validación cruzada del ADR 0009:

- `scripts/prepare_pretrain.py` junta los exports YOLO de Roboflow y traduce las clases a
  `caex` y `bulldozer` (`bermguard/training/pretrain.py`). Las clases sin traducción, como
  `excavator` o `drill`, se descartan: forzarlas a una de las dos enseñaría una confusión.
- `--require bulldozer` conserva sólo las imágenes con al menos un bulldozer: las demás no
  aportan a la clase que falta. En las imágenes que quedan, los camiones (`dumper`,
  `Dumb_truck`) se etiquetan como `caex`; sin etiqueta, el modelo aprendería que un camión
  es fondo.
- `scripts/cross_validate.py --variants pretrain` preentrena una sola vez y afina cada fold
  con la misma partición e hiperparámetros que `base`. Lo único que cambia es el punto de
  partida, así que la diferencia con `base` mide el preentrenamiento. Como ninguna imagen
  pública viene de estos videos, no hay fuga entre folds.

Pasos para correrla, con la GPU libre:

1. En cada página de Roboflow: *Download Dataset* → formato **YOLOv8** → *download zip*
   (pide una cuenta gratuita). Descomprimir en `data/externos/mines/` y
   `data/externos/bulldozer-lpoe2/`.
2. Revisar los nombres en el `data.yaml` de cada uno y preparar:

   ```bash
   python scripts/prepare_pretrain.py \
       --source data/externos/mines --source data/externos/bulldozer-lpoe2 \
       --map dozer=bulldozer bull_dozer=bulldozer dumper=caex dumb_truck=caex \
       --require bulldozer
   ```

   El script imprime, por fuente, cuántas imágenes quedaron, cuántas descartó por no tener
   bulldozer y cuántas instancias hay de cada clase; el resumen queda en
   `data/pretrain/resumen.json`. Los polígonos de `mines` se conservan tal cual.
3. `python scripts/cross_validate.py --variants base copypaste pretrain --batch 4`.

Un riesgo a mirar en el resultado: `bulldozer-lpoe2` tiene 20 veces más imágenes que
`mines`, así que el preentrenamiento queda dominado por él. Si la variante no mejora, el
siguiente intento es preentrenar sólo con `mines`.

Los datasets descargados quedan fuera del repo (`data/externos/`, `data/pretrain/`).
