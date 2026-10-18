# ADR 0008: Validar el pretil contra el ruido real y medir el jitter por columna

- Estado: aceptada
- Fecha: 2026-10-17
- Modifica: ADR 0006
- Implementación: `bermguard/vision/berm/band.py` (`noise_floor`, `hugs_mask`),
  `bermguard/pipeline/orchestrator.py` (`_salto_por_columna`),
  `bermguard/benchmark/synthetic.py`

## Contexto

Hasta acá el pretil solo se podía evaluar por estabilidad: el material no trae el perfil
real anotado. Para tener una prueba con respuesta conocida escribí un generador de escenas
sintéticas (`bermguard/benchmark/synthetic.py`). La cresta se dibuja como un escalón de poco
contraste en una fila conocida, con huellas de neumático más contrastadas debajo y ruido
encima: los tres hechos medidos que hacen difícil el problema (`docs/analisis_material.md`).

Los tests sobre esas escenas encontraron tres defectos que medir solo estabilidad no podía
mostrar.

**1. La validación aceptaba ruido puro.** El camino óptimo siempre existe, así que cada
columna se valida comparando su respuesta con el suelo de ruido de la banda. Ese suelo era
la mediana de la banda, pero la respuesta es un gradiente rectificado: en una zona sin
estructura la mitad de los píxeles vale cero, y la mediana también. El corte caía entonces
al piso de respaldo, el 5 % del pico, que deja pasar casi todo.

En una escena con la cresta **fuera** de la banda de búsqueda, es decir, donde dentro solo
hay ruido:

| Criterio | Cresta dentro de la banda | Cresta fuera de la banda |
|---|---|---|
| 3 × mediana de la banda (antes) | 82–85 % de cobertura | 81–87 % |
| 3 × mediana de la respuesta positiva fuera de la maquinaria | 71–81 % | **0 %** |

El criterio anterior daba la misma cobertura con pretil y sin pretil. Es la explicación
mecánica de lo que la sección 4.3 del reporte ya había observado: que la cobertura "acepta
casi todo" y no distingue el día de la noche.

**2. Las columnas tapadas por una máquina se medían igual.** Las cajas de maquinaria se
anulan en la respuesta, pero el camino puede pasar justo por debajo de la máscara, por la
primera fila sin anular, donde solo queda ruido. En una escena sintética lo hizo en 59 de
las 190 columnas ocultas. Detrás de una máquina el pretil no es observable.

**3. El jitter mezclaba cobertura con movimiento.** Se calculaba sobre la mediana de la
cresta en cada frame, y esa mediana se mueve cuando cambia *qué* columnas son válidas,
aunque el pretil esté quieto. En una secuencia sintética donde la cresta se mueve 0.08 px
por frame, la métrica daba 2.2 px.

## Decisión

1. **El suelo de ruido es la mediana de la respuesta positiva, fuera de la maquinaria**
   (`noise_floor`). El corte sigue siendo 3 veces ese nivel, con el mismo piso del 5 % del
   pico.
2. **Una columna cuya cresta queda a 2 px o menos de la máscara de maquinaria se declara
   sin medición** (`hugs_mask`).
3. **La métrica principal de estabilidad pasa a ser el jitter por columna**
   (`berm_crest_jitter_col_px`): la mediana del cambio de la cresta, columna a columna, entre
   dos frames seguidos de la misma toma, contando solo las columnas medidas en ambos, y
   promediada sobre los pares. El jitter anterior queda en `berm_crest_jitter_px` para
   comparar con los reportes previos.

Los dos métodos del benchmark reciben exactamente los mismos cambios, así que la
comparación sigue aislando la formulación (camino óptimo contra argmax).

## Lo que muestra el benchmark sintético

`scripts/benchmark_berm_synthetic.py` recorre el contraste de la cresta y la cantidad de
huellas, con 20 escenas por punto. Fracción de columnas a 5 px o menos de la cresta
verdadera (`docs/experimentos/pretil_sintetico.json`):

| Contraste | Huellas | Camino óptimo | Argmax + filtro |
|---|---|---|---|
| 18 | 0 | 100 % | 100 % |
| 18 | 6 | **90 %** | 75 % |
| 18 | 12 | **60 %** | 49 % |
| 12 | 4 | **71 %** | 59 % |
| 12 | 12 | 17 % | 20 % |
| 8 | 6 | 17 % | 21 % |

El camino óptimo gana cuando la cresta es visible: con contraste alto la ventaja crece con
las huellas, hasta 11–15 puntos. Cuando la cresta es más débil que los distractores no gana
ninguno, y el camino óptimo puede quedar algo peor, porque si se engancha a una huella larga
la sigue entera. La continuidad ayuda a no saltar a una huella corta, no a encontrar una
cresta que no se ve.

Es la primera medición de **exactitud** del proyecto. Es sobre escenas controladas, no
sobre el material, así que mide la formulación y no el sistema.

## En el material

Con el mismo detector, en la misma máquina, antes y después del cambio
(`docs/experimentos/pretil_antes_despues.json`):

| Video | Cobertura M1 | Jitter por mediana M1 | Jitter por columna M1 | Jitter por columna M2 |
|---|---|---|---|---|
| `video_01` | 0.91 → 0.65 | 25.2 → 40.1 px | 5.52 px | 8.80 px |
| `video_02` | 0.90 → 0.50 | 9.9 → 15.1 px | 0.49 px | 0.68 px |
| `video_03` | 0.86 → 0.45 | 9.9 → 13.3 px | 0.89 px | 1.73 px |
| `video_04` | 0.90 → 0.54 | 13.2 → 16.3 px | 2.10 px | 2.39 px |

Tres lecturas:

1. **La cobertura baja a la mitad.** Es lo esperado: antes contaba como pretil el ruido de
   la banda. Que ahora varíe entre videos (0.45 a 0.65) es la señal de que vuelve a medir la
   escena.
2. **El jitter por mediana empeoró en el Método 1.** No lo escondo, pero tampoco lo leo como
   una pérdida de estabilidad: con menos columnas válidas, la mediana de cada frame depende
   más de *qué* columnas entran, que es justo el defecto de esa métrica. En el Método 2 bajó.
3. **Con la métrica por columna, el camino óptimo es más estable que el argmax en los
   cuatro videos**, entre 1.1 y 1.9 veces. Es la conclusión del ADR 0006, ahora sobre una
   métrica que no mezcla cobertura con movimiento.

La altura mediana por condición de luz casi no cambia, y las alertas de proximidad son las
mismas, porque no dependen del pretil.

## Alternativas descartadas

- **Percentil de la respuesta del camino.** Ya descartado en el ADR 0006: acepta siempre la
  misma fracción.
- **MAD sobre la banda.** La mediana de la desviación absoluta también es cero sobre un
  gradiente rectificado; en las pruebas rechazó todas las columnas, con y sin cresta.
- **Percentil 99 de la banda.** Distingue algo (23–63 % con cresta, 0–30 % sin ella), pero
  peor que la mediana positiva y con más dependencia del ruido.
- **Rellenar las columnas ocultas por interpolación.** Sería reportar un pretil que no se
  vio.

## Consecuencias

- **La cobertura vuelve a significar algo**: una banda sin pretil da cobertura baja. Sobre
  el material baja de ~0.9 a 0.45–0.65; no es que el método empeore, es que deja de contar
  ruido como pretil.
- Las cifras de jitter de los reportes anteriores no son comparables con
  `berm_crest_jitter_col_px`.
- El benchmark sintético queda como test de regresión (`tests/test_berm.py`): el camino
  óptimo tiene que superar al argmax con huellas, y ninguno de los dos puede devolver pretil
  sobre una banda de ruido.
