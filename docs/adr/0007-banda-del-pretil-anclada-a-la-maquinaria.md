# ADR 0007: La banda de búsqueda del pretil se limita por arriba con la maquinaria

- Estado: aceptada
- Fecha: 2026-09-11
- Modifica: ADR 0006
- Implementación: `bermguard/vision/berm/band.py`

## Contexto

Revisando los videos nocturnos vi que la curva del pretil se iba al horizonte. En el frame
30 de `video_02` la mitad izquierda de la cresta seguía la línea entre el terreno oscuro y el
valle iluminado del fondo, donde están las luces del pueblo, y el camino óptimo la unía con
el pretil real de la mitad derecha con un salto vertical.

La causa estaba en la banda de búsqueda. El ADR 0006 usaba la maquinaria solo como límite
**inferior**, y como límite superior tomaba la primera fila con textura, que en ese frame
era la 175. De noche el borde horizontal más fuerte no es el pretil (el lado izquierdo no
tiene luz), sino esa línea del valle, y estaba dentro de la banda.

| Frame 30 de `video_02` | Fila |
|---|---|
| Punto de contacto del CAEX | 560 |
| Alto de su caja | ~190 px |
| Pretil real | 490–520 |
| Línea del valle (lo que se medía) | ~295 |
| Límite superior anterior | 175 |

## Decisión

**Limitar la banda también por arriba con la maquinaria: el límite superior pasa a ser el
punto de contacto menos una altura de caja.**

El pretil está al borde de la plataforma por donde circulan los equipos, más o menos a su
misma profundidad. En la imagen no puede estar mucho más arriba que el techo de un camión
que pasa a su lado. Con un múltiplo de 1.0 el margen es amplio (un pretil mide del orden de
un tercio del alto de un CAEX), y en el frame de arriba deja el pretil dentro (378–588) y el
valle fuera.

Con varios equipos se usa el límite más alto que permita cualquiera, porque el pretil puede
estar junto a cualquiera de ellos. Sin maquinaria se mantiene el criterio anterior.

La lógica quedó en un módulo que usan los dos métodos. La comparación del benchmark solo
vale si ambos buscan en la misma banda, y con el código duplicado un cambio en uno solo
habría alterado la comparación sin que se notara.

## Alternativas descartadas

- **Un margen fijo bajo el horizonte.** Arregla este frame, pero es un número arbitrario:
  en una toma más abierta, con el pretil lejos, lo cortaría.
- **Penalizar líneas perfectamente rectas.** El horizonte es recto y el pretil irregular,
  pero un pretil lejano también se ve casi recto.
- **Enmascarar las luces del pueblo.** Ataca el síntoma en estos videos y no la causa, que
  es geométrica.

## Consecuencias

Medido en los 4 videos con los dos métodos:

| | Jitter antes | Jitter después | Cambio |
|---|---|---|---|
| Método 1 | 20.03 px | 14.57 px | **−27 %** |
| Método 2 | 32.84 px | 26.17 px | **−20 %** |

**Confirmó una hipótesis del reporte.** De noche el Método 2 parecía más estable que el 1
(17.58 contra 22.10 px), y yo había anotado que probablemente estaba pegado de forma
estable a un borde equivocado. Sin el valle en la banda, su jitter nocturno subió a
33.46 px (+90 %), mientras que el del Método 1 bajó a 19.21. Ahora el Método 1 es más
estable en las tres condiciones de luz.

**La dispersión de la altura entre condiciones bajó a la mitad:**

| | Día | Crepúsculo | Noche | Dispersión |
|---|---|---|---|---|
| Antes | 0.38 m | 0.47 m | 0.63 m | ±26 % |
| Después | 0.33 m | 0.36 m | 0.42 m | **±12 %** |

Un pretil no cambia de altura con la luz, así que esa dispersión es error. Casi todo venía
de la noche: con la cresta en el valle, la distancia hasta la base se alargaba e inflaba la
altura.

**Lo que empeoró o quedó pendiente:**

- La altura mediana bajó y el sesgo contra la normativa se nota más (factor ~4 en vez de
  3). El arreglo quitó un error que inflaba las alturas nocturnas; el subregistro de fondo
  sigue igual.
- La cobertura bajó unos 4 puntos (de 93.6 a 89.3 % en el Método 1). No cambia ninguna
  conclusión, porque la cobertura ya estaba descartada como métrica.
- Donde la banda no tiene un borde claro (junto al camión, en el frame de arriba), la curva
  baja hasta el límite inferior. Es menos grave que el error corregido porque queda dentro
  de la zona plausible, pero es un defecto.
- Con una caja demasiado grande, el error cambia de dirección. El límite inferior sale del
  punto de contacto más bajo, así que una caja que se extiende de más hacia abajo arrastra
  la banda al primer plano. En el frame 130 de `video_04` la caja del CAEX llega a la fila
  664 y la curva termina sobre las huellas de neumático; antes del arreglo, ese frame se iba
  hacia la llanura del fondo. Viene de la precisión del detector (0.297). Usar la mediana de
  los puntos de contacto en vez del máximo lo atenuaría, y queda pendiente.
