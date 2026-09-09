# Guía de anotación

Protocolo seguido para etiquetar el conjunto de entrenamiento del detector (ADR 0004).

Existe por una razón concreta: la consistencia importa más que el volumen. Un criterio
aplicado de forma uniforme a doscientos frames produce mejor modelo que uno aplicado a ojo
sobre cuatrocientos, porque las reglas contradictorias no se promedian — se aprenden como
ruido. Este documento también hace auditable el conjunto: quien lea las métricas puede saber
qué se consideró un acierto.

## Clases

El orden define el índice numérico en los archivos de etiquetas y debe respetarse.

| Índice | Clase | Qué es |
|---|---|---|
| 0 | `caex` | Camión de extracción (*haul truck*). Bastidor rígido, tolva basculante sobre el chasis, ruedas de gran diámetro. Silueta compatible con un Caterpillar 793 |
| 1 | `bulldozer` | Tractor de oruga (*crawler dozer*). Orugas en lugar de ruedas, hoja empujadora al frente, habitualmente ripper en la parte trasera |

Son categorías **operativas**, no visuales: distinguen dos roles dentro de la plataforma de
descarga. Es la distinción que los datasets públicos no ofrecen, ya que etiquetan `truck` o
`vehicle` de forma genérica.

## Extensión de la caja

**Ajustada a los extremos visibles de la máquina.** Los bordes de la caja tocan el píxel más
externo del equipo, sin margen de holgura.

Qué queda **dentro**:

- CAEX: tolva, cabina, chasis y ruedas completas, hasta el punto de contacto con el suelo.
- Bulldozer: hoja delantera, cuerpo, cabina, orugas y ripper trasero si es visible.

Qué queda **fuera**:

- El penacho de polvo que la máquina levanta. Es ambiente, no equipo.
- La sombra proyectada.
- El halo de los focos en las secuencias nocturnas. Se etiqueta el cuerpo de la máquina, no
  la luz que emite.

El borde inferior de la caja merece atención particular: de él se deriva el punto de contacto
con el suelo, que es el único punto proyectable a la vista cenital y el origen de toda
distancia métrica del sistema. Una caja cuyo borde inferior queda alto desplaza la máquina
hacia el fondo de la escena y corrompe la medición de proximidad.

## Casos límite

**Máquina cortada por el borde del frame.** Se etiqueta la porción visible, con la caja
terminando en el borde de la imagen.

**Oclusión parcial por polvo.** Si el contorno se infiere con claridad a través del polvo, se
incluye la extensión completa. Si el polvo es denso y el límite es adivinado, se etiqueta sólo
lo que se distingue.

**Dos máquinas superpuestas.** Se dibuja una caja por máquina, aunque se solapen. Es
exactamente el caso que el modelo base no resuelve y la razón de ser de este entrenamiento.

**Tamaño mínimo: 20 px de ancho.** Por debajo de eso no se etiqueta. Una máquina de doce
píxeles en el horizonte no es identificable de forma fiable ni siquiera para el anotador, y
etiquetarla enseña ruido.

**Tipo indistinguible.** Si a la distancia no puede determinarse si es un CAEX o un bulldozer,
no se etiqueta. Una etiqueta adivinada es peor que una ausente: la primera enseña una
asociación falsa, la segunda sólo omite un ejemplo.

**Frames sin ninguna máquina.** Se conservan con archivo de etiqueta vacío. Enseñan cómo se ve
una escena sin equipos y reducen falsos positivos sobre penachos de polvo y formaciones del
terreno. Eliminarlos desperdicia información.

## Pre-anotaciones

`scripts/prepare_labeling_set.py` propone cajas usando el modelo preentrenado. Son un punto de
partida, no un resultado.

Como documenta `docs/analisis_material.md`, ese modelo **fusiona el CAEX con el bulldozer en
una sola caja** cuando aparecen próximos. Toda caja propuesta se revisa, y las fusionadas se
dividen en dos. Ninguna pre-anotación identifica bulldozers: todas llegan como clase `caex` y
hay que corregirlas.

## Limitaciones declaradas

- **Anotador único.** No hay acuerdo entre anotadores, de modo que las métricas calculadas
  contra este conjunto arrastran una incertidumbre propia que no está cuantificada.
- **Los umbrales son juicio del anotador.** El mínimo de 20 px y el criterio de
  distinguibilidad se aplicaron a ojo, no mediante medición.
- **El material proviene de una sola faena.** El conjunto no cubre otras configuraciones de
  botadero, otros modelos de equipo ni otros ángulos de cámara.
