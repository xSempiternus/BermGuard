# ADR 0006: La cresta del pretil como camino óptimo

- Estado: aceptada, modificada por el ADR 0007
- Fecha: 2026-09-09
- Implementación: `bermguard/vision/berm/classical.py`

## Contexto

El Método 1 tiene que encontrar la cresta y la base del pretil sin un modelo entrenado. La
idea inicial era la habitual: gradiente vertical, argmax por columna y suavizado. Al
medirlo sobre el material vi que no funcionaba, por tres razones:

- **El gradiente vertical es más fuerte en el primer plano que en el pretil.** Por bandas
  de filas, la energía crece hacia abajo hasta 379 en el último décimo del frame, contra 223
  en la banda de la cresta. Las huellas de neumático y las sombras responden más que el
  pretil.
- **Exigir coherencia horizontal no ayuda.** Promediar el gradiente en ventanas de 31, 81 y
  161 px deja el orden de las filas igual, porque las huellas son tan horizontales como el
  pretil.
- **El pretil tiene poco contraste.** Es tierra oscura sobre suelo del mismo color.

La primera versión, con argmax, filtro de mediana y Savitzky-Golay, daba saltos de cientos
de píxeles entre columnas vecinas, con la curva pegada a vehículos y polvo.

## Decisión

**Buscar la cresta como el camino de mayor respuesta acumulada, con programación dinámica
y un salto vertical máximo entre columnas vecinas.**

El argmax decide cada columna por separado, y cuando el filtro actúa la continuidad ya se
perdió. Con el camino óptimo la continuidad pasa a ser una restricción del problema: se
maximiza la suma de respuestas a lo largo de un camino que cruza todas las columnas, con un
salto máximo de `max_step_px` (3 por defecto). Cada paso se resuelve con un filtro de máximo
deslizante, así que el costo es una pasada vectorizada por columna.

Además:

- **Gradiente a escala gruesa y solo positivo.** Se suaviza antes de derivar para que
  sobreviva el escalón del pretil y se cancele la textura fina. Me quedo con el signo
  positivo porque la cresta oscurece hacia abajo, mientras que la textura cambia de signo.
- **Se excluye la maquinaria.** Las cajas detectadas, algo agrandadas, se anulan antes de
  buscar. Una máquina tiene bordes mucho más marcados que un banco de tierra.
- **Banda de búsqueda acotada.** Es la franja de filas donde se busca el pretil. Por arriba
  se excluye el cielo, detectado por varianza por fila. Por abajo, el punto de contacto más
  bajo de la maquinaria, porque en la imagen el pretil siempre está sobre la rasante. El
  ADR 0007 le agregó un límite superior.
- **Se valida el camino.** El camino óptimo siempre existe, incluso sobre suelo liso. Las
  columnas cuya respuesta no supera 3 veces la mediana de la banda se marcan como `NaN`.
- **Suavizado.** El perfil se suaviza con Savitzky-Golay y se promedia en el tiempo con una
  media móvil exponencial (α=0.3).

## Alternativas descartadas

- **Argmax por columna con filtrado.** Lo implementé y lo descarté midiendo: el perfil salta
  entre estructuras sin relación. Quedó como Método 2, la línea base del benchmark.
- **Coherencia horizontal con ventanas anchas.** Probada con tres anchos; no cambia el orden
  de las filas.
- **Umbralizar y buscar componentes conexas.** Con tan poco contraste, cualquier umbral que
  capte el pretil capta también la textura.
- **Ajustar una recta o parábola con RANSAC.** El pretil sigue el borde del botadero, que es
  irregular. Un modelo rígido trataría como outliers justo las zonas hundidas, que son lo
  que interesa detectar.

## Consecuencias

- **La cobertura no sirve como métrica.** El camino pasa por todas las columnas, así que la
  cobertura depende solo del criterio de validación (sección 4.3 del reporte). Por eso la
  calidad se mide con el jitter temporal de la cresta.
- **Es la etapa más cara con GPU**: 36 ms por frame a 720p y 74 ms a 1080p, porque recorre
  toda la rejilla.
- **Depende de que haya maquinaria detectada.** Sin ella la banda es un recorte grueso del
  frame y la confianza reportada baja a la mitad.
- `IBermSegmenter.segment` recibe las detecciones del frame. Es un acoplamiento entre la
  rama de maquinaria y la de terreno que al principio quería evitar, pero es físico: el
  terreno necesita saber dónde están las máquinas para excluirlas y ubicar la rasante. Es
  opcional, así que un segmentador puede ignorarlas.
