# Guía de anotación

Criterios que usé para anotar el conjunto de entrenamiento (ADR 0004). Los escribí para
aplicar siempre la misma regla, porque el modelo aprende las reglas inconsistentes como
ruido, y para que quede claro qué cuenta como acierto en las métricas.

## Clases

El orden define el índice en los archivos de etiquetas.

| Índice | Clase | Qué es |
|---|---|---|
| 0 | `caex` | Camión de extracción. Chasis rígido, tolva basculante y ruedas grandes. Silueta tipo Caterpillar 793 |
| 1 | `bulldozer` | Tractor de oruga. Orugas, hoja al frente y a veces ripper atrás |

Son roles en el botadero, no categorías genéricas como `truck` o `vehicle`.

## La caja

Ajustada a los extremos visibles de la máquina, sin margen.

- **Dentro:** en el CAEX, tolva, cabina, chasis y ruedas hasta el suelo. En el bulldozer,
  hoja, cuerpo, cabina, orugas y ripper si se ve.
- **Fuera:** el polvo que levanta, la sombra y el halo de los focos de noche.

El borde inferior es el más importante. De ahí sale el punto de contacto con el suelo, que
es lo que se proyecta a la vista en planta y el origen de toda distancia. Si queda alto, la
máquina aparece más lejos de lo que está.

## Casos límite

- **Cortada por el borde del frame:** se anota la parte visible.
- **Tapada en parte por polvo:** si el contorno se ve claro, se anota completa; si hay que
  adivinarlo, solo lo que se distingue.
- **Dos máquinas superpuestas:** una caja por máquina, aunque se solapen. Es justo lo que
  el modelo base no resolvía.
- **Menos de 20 px de ancho:** no se anota. A esa distancia ni yo puedo identificarla con
  seguridad.
- **Tipo indistinguible:** no se anota. Una etiqueta adivinada enseña algo falso, y una que
  falta solo pierde un ejemplo.
- **Sin máquinas:** el frame se deja con el archivo de etiquetas vacío. Ayuda a reducir los
  falsos positivos sobre polvo y terreno.

## Pre-anotaciones

`scripts/prepare_labeling_set.py` propone cajas con el modelo COCO. Son solo un punto de
partida: ese modelo junta el CAEX y el bulldozer cuando están cerca, así que revisé todas,
separé las fusionadas y corregí la clase de los bulldozers, que llegan todos como `caex`.

## Auditoría (octubre de 2026)

La regla de corregir la clase de las pre-anotaciones no se cumplió del todo. 191 de las 309
etiquetas de CAEX seguían en formato caja, es decir, sin pasar por la corrección, y entre
ellas había **29 bulldozers etiquetados como `caex`**. Los encontré revisando las 191 en
hojas de contacto y los corregí con `scripts/apply_label_audit.py`; el detalle está en
`data/auditoria_etiquetas.json` y el efecto en las métricas, en el ADR 0009.

Para no repetirlo:

- **Una pre-anotación aceptada sin cambios sigue siendo una pre-anotación.** Si queda en
  formato caja, es señal de que nadie la miró.
- **Revisar por clase en hojas de contacto** antes de entrenar: 191 recortes se revisan en
  minutos y un error de clase salta a la vista. Frame por frame no se ve.
- **Cuatro casos quedaron abiertos** y están listados en `pendientes` dentro del archivo de
  auditoría: una caja que junta un CAEX y un bulldozer lejano, que hay que partir, y tres
  máquinas ambiguas, dos de ellas híbridas inventadas por el generador de video.

## Limitaciones

- Anoté yo solo, sin acuerdo entre anotadores.
- El mínimo de 20 px y el criterio de "distinguible" los apliqué a ojo.
- Todo el material es de una sola faena, con los mismos equipos y el mismo ángulo de
  cámara.
