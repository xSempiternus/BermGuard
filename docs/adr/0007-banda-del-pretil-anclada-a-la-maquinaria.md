# ADR 0007: Limitar la búsqueda del pretil con la altura de las máquinas

- Estado: aceptada
- Fecha: 2026-09-11
- Modifica: ADR 0006
- Implementación: `bermguard/vision/berm/band.py`

## Contexto

Revisando los videos nocturnos vi que la curva del pretil se iba al horizonte. En el frame
30 de `video_02`, la mitad izquierda de la curva seguía el borde del valle iluminado del
fondo (las luces del pueblo) y después bajaba de golpe hasta el pretil real.

El pretil no se busca en toda la imagen, sino en una franja horizontal (la "banda de
búsqueda" en el código). Esa franja terminaba abajo en las máquinas, porque el pretil
siempre está más arriba que el suelo que pisan, pero arriba llegaba casi hasta el horizonte.
De noche la línea más marcada de la imagen no es el pretil, que en ese lado no tiene luz,
sino el borde del valle. Como quedaba dentro de la franja, el método lo eligió.

En píxeles contados desde el borde superior de la imagen (menos es más arriba):

| Frame 30 de `video_02` | Altura en la imagen |
|---|---|
| Donde el CAEX toca el suelo | 560 |
| Alto de la caja del CAEX | ~190 px |
| Pretil real | 490–520 |
| Borde del valle (lo que se medía) | ~295 |
| Borde superior de la franja, antes | 175 |

## Decisión

**La franja termina arriba a una altura de camión sobre el punto donde el equipo toca el
suelo.**

El pretil está en el borde de la plataforma por donde circulan los equipos, así que en la
imagen no puede aparecer mucho más arriba que el techo de un camión que pasa a su lado. Una
altura de camión es un margen generoso, porque un pretil mide cerca de un tercio de un CAEX.
En este frame deja el pretil dentro de la franja (378–588) y el valle fuera.

Si hay varios equipos, uso el que deja la franja más amplia, porque el pretil puede estar
junto a cualquiera. Si no hay ninguno, se mantiene el criterio anterior. Los dos métodos
usan el mismo código para esto, así el benchmark los compara buscando en la misma zona.

## Alternativas descartadas

- **Cortar a una distancia fija bajo el horizonte.** Arregla este frame, pero es un número
  arbitrario: en una toma más abierta, con el pretil lejos, lo dejaría fuera.
- **Castigar las líneas muy rectas**, porque el horizonte es recto y el pretil no. Pero un
  pretil lejano también se ve casi recto.
- **Tapar las luces del pueblo.** Arregla estos videos, no la causa.

## Consecuencias

El jitter mide cuánto tiembla la curva de un frame al siguiente. En los 4 videos:

| | Jitter antes | Jitter después | Cambio |
|---|---|---|---|
| Método 1 | 20.03 px | 14.57 px | **−27 %** |
| Método 2 | 32.84 px | 26.17 px | **−20 %** |

**Confirmó una sospecha del reporte.** De noche el Método 2 parecía más estable que el 1
(17.58 contra 22.10 px), y yo había anotado que probablemente estaba quieto porque se
quedaba pegado a la línea equivocada. Al sacar el valle de la franja, su jitter nocturno
subió a 33.46 px (+90 %) y el del Método 1 bajó a 19.21. Ahora el Método 1 es más estable
de día, al atardecer y de noche.

**La altura medida varía la mitad entre día y noche:**

| | Día | Crepúsculo | Noche | Variación |
|---|---|---|---|---|
| Antes | 0.38 m | 0.47 m | 0.63 m | ±26 % |
| Después | 0.33 m | 0.36 m | 0.42 m | **±12 %** |

Un pretil no cambia de altura cuando oscurece, así que esa variación es error. Casi todo
venía de la noche: con la curva en el valle, el pretil parecía mucho más alto.

**Lo que empeoró o quedó pendiente:**

- Las alturas bajaron, y se nota más que el sistema mide el pretil más bajo de lo que pide
  la normativa (unas 4 veces, antes 3). El arreglo quitó un error que inflaba la noche, pero
  no el de fondo.
- La parte de la curva que el sistema da por válida bajó unos 4 puntos (de 93.6 % a
  89.3 %). No cambia nada, porque esa cifra ya estaba descartada como métrica.
- Donde la franja no tiene un borde claro, como junto al camión en este frame, la curva baja
  hasta el límite inferior de la franja.
- Si el detector dibuja una caja más grande que el camión, la franja baja hacia el primer
  plano. En el frame 130 de `video_04` la caja del CAEX llega hasta 664 y la curva termina
  sobre las huellas de neumático. Viene de la precisión del detector (0.297). Usar la
  posición típica de los equipos, en vez de la del que está más abajo, lo atenuaría; queda
  pendiente.
