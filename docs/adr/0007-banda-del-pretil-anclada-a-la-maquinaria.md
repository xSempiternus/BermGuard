# ADR 0007 — La banda de búsqueda del pretil se ancla a la maquinaria por arriba

- **Estado:** aceptada
- **Fecha:** 2026-09-11
- **Modifica:** ADR 0006
- **Implementación:** `bermguard/vision/berm/band.py`

## Contexto

La revisión visual de los videos nocturnos mostró que la curva del pretil se confundía
con el horizonte. En el frame 30 de `video_02` la mitad izquierda de la cresta corría
sobre la frontera entre el terreno oscuro y el valle iluminado del fondo —donde están las
luces del poblado—, y el camino óptimo la conectaba con el pretil real de la mitad derecha
mediante un salto vertical.

El diagnóstico es geométrico. La banda de búsqueda del ADR 0006 usaba la maquinaria sólo
como límite **inferior** —el pretil está por encima de la rasante— y tomaba como límite
superior la primera fila con textura, que en ese frame caía en y=175. De noche, el borde
horizontal más fuerte y continuo de la escena no es el pretil, que en el lado izquierdo
no recibe iluminación, sino esa frontera del valle. Estaba dentro de la banda, y el camino
óptimo la eligió.

Las cifras del frame:

| | Fila |
|---|---|
| Punto de contacto del CAEX | 560 |
| Alto de su caja | ~190 px |
| Pretil real | 490–520 |
| Frontera del valle (lo que se medía) | ~295 |
| Límite superior de la banda anterior | 175 |

## Decisión

**Acotar la banda también por arriba, con la maquinaria:** el límite superior pasa a ser
el punto de contacto menos una altura de caja.

El fundamento es físico. El pretil del botadero está al borde de la plataforma por la que
circulan los equipos, es decir, aproximadamente a su misma profundidad. En el espacio
imagen no puede aparecer mucho más arriba que el techo de un camión que opera junto a él.
Con un múltiplo de 1.0 el criterio es generoso —un pretil real mide del orden de un tercio
de la altura de un CAEX— y en el frame del contexto deja el pretil dentro (378–588) y la
frontera del valle fuera.

Cuando hay varios equipos se toma la cota más alta que admite cualquiera de ellos, porque
el pretil puede estar junto a cualquiera; descartarlo por estar cerca de un equipo lejano
sería perder pretil real. Sin maquinaria no hay ancla, y se conserva el criterio anterior.

La lógica se extrajo a un módulo compartido por ambos segmentadores. La comparación del
benchmark sólo es válida si los dos métodos buscan en la misma región, y con la lógica
duplicada un cambio aplicado a uno y olvidado en el otro habría alterado la comparación
sin que nada lo indicara.

## Alternativas consideradas

**Margen fijo por debajo del horizonte.** Resuelve este frame, pero es un número sin
justificación: en una toma más abierta, con el pretil lejano, recortaría pretil real.

**Penalizar las líneas perfectamente rectas.** El horizonte es recto y un pretil es
irregular, así que discrimina bien en este material. Se descarta por frágil: un pretil
lejano también se proyecta como una línea casi recta.

**Enmascarar las luces puntuales cálidas del poblado.** Ataca el síntoma en estos videos
concretos y no la causa, que es geométrica.

## Consecuencias

Medidas sobre los 4 videos, con ambos métodos:

| | Jitter antes | Jitter después | Cambio |
|---|---|---|---|
| Método 1 | 20.03 px | 14.57 px | **−27 %** |
| Método 2 | 32.84 px | 26.17 px | **−20 %** |

### La inversión nocturna desapareció, y eso confirma un diagnóstico anterior

El reporte de benchmark señalaba que de noche el Método 2 parecía más estable que el 1
(17.58 contra 22.10 px), y advertía que eso no significaba que fuese mejor: el jitter mide
estabilidad y no exactitud, y un `argmax` que se bloquea sobre el mismo máximo local frame
tras frame produce jitter bajo y una respuesta consistentemente equivocada.

Esta corrección es el experimento que lo verifica. Al sacar la frontera del valle de la
banda, el Método 2 ya no tiene dónde bloquearse:

| Noche | Antes | Después |
|---|---|---|
| Método 1 | 22.10 px | 19.21 px |
| Método 2 | **17.58 px** | **33.46 px (+90 %)** |

Su jitter nocturno casi se duplica — su estabilidad anterior era la de estar anclado a un
objetivo equivocado. El Método 1 pasa a ser más estable en las tres condiciones.

### La dispersión de la altura entre condiciones se redujo a la mitad

| | Día | Crepúsculo | Noche | Dispersión |
|---|---|---|---|---|
| Antes | 0.38 m | 0.47 m | 0.63 m | ±26 % |
| Después | 0.33 m | 0.36 m | 0.42 m | **±12 %** |

Un pretil físico no cambia de altura al atardecer, así que esa dispersión es error de
medición. La mayor parte venía de la noche: medir la cresta sobre la frontera del valle
alarga la extensión vertical hasta la base, y la altura resultante sale inflada.

### Lo que empeoró o quedó pendiente

- **La altura mediana bajó**, y con ella el sesgo sistemático respecto a la referencia
  normativa se hace más visible. La corrección eliminó un error que casualmente inflaba
  las alturas nocturnas; el subregistro de fondo —cuyas hipótesis plantea el reporte— no
  cambió.
- **La cobertura bajó** unos 4 puntos (93.6 → 89.3 % en el Método 1), como es esperable al
  estrechar la banda. No afecta a ninguna conclusión, porque la cobertura ya estaba
  documentada como no discriminativa.
- **Aparece un artefacto nuevo.** Donde la banda no contiene un borde claro —junto al
  camión, en el frame del contexto—, el camino óptimo baja hasta el límite inferior de la
  banda en lugar de seguir una estructura. Es menos grave que el error corregido, porque
  queda confinado a la región físicamente plausible, pero es un defecto y así se declara.
- **Con una caja sobredimensionada, el error cambia de dirección.** El límite inferior de
  la banda se deriva del punto de contacto más bajo, así que una caja que se extiende de más
  hacia abajo arrastra la banda al primer plano. En el frame 130 de `video_04` la caja del
  CAEX llega a la fila 664 y la curva termina sobre las huellas de neumático, cuando antes
  de esta corrección el error de ese frame iba hacia la llanura del fondo. La causa es la
  precisión del detector (0.297), no la banda en sí. Tomar la mediana de los puntos de
  contacto en lugar del máximo lo atenuaría, y queda como trabajo pendiente.
