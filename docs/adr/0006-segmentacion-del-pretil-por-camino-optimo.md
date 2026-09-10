# ADR 0006 — Segmentación del pretil: búsqueda de camino óptimo

- **Estado:** aceptada
- **Fecha:** 2026-09-09
- **Evidencia:** `docs/analisis_material.md`, sección 1
- **Implementación:** `bermguard/vision/berm/classical.py`

## Contexto

El Método 1 debe localizar la cresta y la base del pretil sin ningún modelo entrenado. El
plan inicial contemplaba el enfoque habitual: gradiente vertical, `argmax` por columna, y
suavizado posterior del perfil.

La medición sobre el material mostró que ese enfoque no es viable tal cual, por tres razones
que se comprobaron una por una:

**La energía de gradiente vertical es máxima en el primer plano, no en el pretil.** Por bandas
de filas, la energía media crece monótonamente hacia abajo hasta 379 en el último décimo del
frame, frente a 223 en la banda donde está la cresta. Las huellas de neumático y las sombras
largas producen respuestas más fuertes que el propio pretil, de modo que un `argmax` global
por columna elige textura del terreno.

**Exigir coherencia horizontal no separa nada.** Promediar el gradiente sobre ventanas de 31,
81 y 161 píxeles de ancho deja el orden de las filas por energía **exactamente igual**: las
huellas de neumático son tan horizontales como el pretil.

**El contraste del pretil es bajo.** Es un banco de tierra oscura sobre suelo del mismo color,
no un borde limpio.

Una primera implementación con `argmax` por columna, filtro de mediana y Savitzky-Golay
produjo un perfil inutilizable: saltos de cientos de píxeles entre columnas contiguas, con la
curva enganchada a los vehículos y a las nubes de polvo.

## Decisión

**Buscar la cresta como un camino de máxima respuesta acumulada mediante programación
dinámica, con el salto vertical acotado entre columnas contiguas.**

El diagnóstico del fracaso inicial es que el `argmax` por columna decide cada columna de forma
independiente. La continuidad del pretil es un prior fuerte y disponible, pero un
post-procesado no puede recuperarla: cuando el filtro actúa, la información ya se perdió al
elegir el máximo local de cada columna por separado.

La formulación como camino óptimo convierte ese prior en **una restricción del problema**. Se
maximiza la suma de respuestas a lo largo de un camino que recorre todas las columnas, sujeto
a que el salto vertical entre columnas contiguas no supere `max_step_px` (3 por defecto). El
paso hacia adelante se resuelve con un filtro de máximo deslizante, porque «mejor predecesor
dentro de una ventana» es exactamente esa operación, y eso deja el coste en una pasada
vectorizada por columna.

Acompañan tres decisiones necesarias para que la búsqueda tenga sentido:

**Gradiente a escala gruesa y con signo.** Se suaviza con una gaussiana antes de derivar
—equivalente a una derivada de gaussiana— de modo que el escalón estructural sobrevive y la
textura de alta frecuencia se cancela. Se conserva sólo el signo positivo, porque la cresta
oscurece hacia abajo mientras la textura oscila de signo.

**Exclusión de la maquinaria.** Las cajas de las detecciones se anulan en la respuesta antes
de buscar, dilatadas para cubrir el desenfoque de movimiento y el polvo adyacente. El pretil
es terreno; una máquina produce bordes mucho más marcados que un banco de tierra.

**Banda de búsqueda acotada.** Por arriba se excluye el cielo, que es el borde horizontal más
nítido de la escena, detectándolo por varianza por fila. Por abajo se usa el punto de contacto
más bajo de la maquinaria detectada: el pretil está siempre por encima de la rasante en el
espacio imagen.

**Validación explícita del camino.** El camino óptimo existe siempre, también sobre suelo
liso. Sin validar, el método devolvería una línea inventada con aspecto convincente. La
respuesta a lo largo del camino se compara contra un umbral relativo, y las columnas que no
lo superan se marcan `NaN`.

## Alternativas consideradas

**`argmax` por columna con filtrado posterior.** Implementada y descartada por medición: el
perfil resultante salta entre estructuras no relacionadas y la curva se engancha a los
vehículos. El filtrado no lo corrige porque actúa cuando la información de continuidad ya se
perdió.

**Coherencia horizontal por filtrado de ventana ancha.** Probada con tres anchos distintos.
No altera el orden de las filas por energía, porque la textura del primer plano es tan
horizontal como el pretil.

**Umbralizado del gradiente y análisis de componentes conexas.** Descartado por el bajo
contraste: cualquier umbral que capture el pretil captura también la textura del terreno, y
las componentes resultantes se fusionan.

**Ajuste de una recta o parábola por RANSAC.** Descartado porque impone una forma que el
pretil no tiene: sigue el borde del botadero, que es curvo e irregular por diseño. Un modelo
paramétrico rígido descartaría como outliers precisamente las variaciones que interesa medir
—una zona hundida del pretil es la señal, no el ruido.

## Consecuencias

### La cobertura no es una métrica utilizable con este método

Es el hallazgo más incómodo y conviene declararlo antes que nada, porque la cobertura era la
métrica prevista para cuantificar la calidad de la segmentación sin ground truth.

El camino óptimo recorre **todas** las columnas por construcción, de modo que la cobertura
depende enteramente del criterio con que se validan sus columnas. Y se probaron dos:

| Criterio de validación | Cobertura medida |
|---|---|
| Percentil 55 de la respuesta del propio camino | 43 % en los 4 videos y las 3 condiciones |
| 3 × la mediana de la banda de búsqueda | 92–95 % en los 4 videos y las 3 condiciones |

**Ninguna de las dos cifras describe el pretil.** La primera es el complemento del percentil
elegido, y sale idéntica porque un percentil selecciona por rango: siempre acepta la misma
fracción. La segunda es alta y uniforme porque la respuesta de gradiente a escala gruesa es
suave y no nula en casi todas las columnas, de modo que cualquier umbral referido al ruido las
acepta.

Que el número sea insensible a la condición lumínica —cuando el contraste varía dos órdenes de
magnitud entre día y noche— es la señal de que mide el umbral y no la escena.

Se conserva el criterio referido al ruido, por ser el más defendible de los dos, y **la
cobertura se reporta con esta advertencia explícita en lugar de presentarse como medida de
calidad**. La métrica que sí sería informativa sin ground truth es el **jitter temporal** de la
cresta —la desviación estándar de su variación entre frames consecutivos—, que es además lo que
el enunciado penaliza al hablar de parpadeo. Queda pendiente.

### Otras consecuencias

- El método no delinea el pretil completo. En las escenas nocturnas la inspección visual
  muestra que la curva se engancha a penachos de polvo iluminados por los faros, y eso no lo
  refleja ninguna de las dos cifras de cobertura.
- El coste es de 57 ms por frame en 720p y 120 ms en 1080p, comparable o superior al de la
  inferencia del detector. Es el precio de una búsqueda global sobre toda la rejilla.
- La calidad depende de que haya maquinaria detectada. Sin ella, la banda de búsqueda es un
  recorte grosero del frame y la confianza reportada se penaliza a la mitad.
- **El comportamiento nocturno está sin verificar** al momento de escribir este ADR. La
  medición por condición lumínica es parte del reporte de benchmark.
- El contrato `IBermSegmenter.segment` pasó a recibir las detecciones del frame. Es un
  acoplamiento entre la rama de maquinaria y la de terreno que el diseño inicial evitaba
  deliberadamente. Se acepta porque la dependencia es física y no accidental: la segmentación
  de terreno necesita saber dónde están las máquinas para excluirlas y para situar la rasante.
  Sigue siendo opcional, de modo que un segmentador puede ignorarlas.
