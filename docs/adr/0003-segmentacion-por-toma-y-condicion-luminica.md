# ADR 0003: La toma como unidad de análisis y la luz clasificada por frame

- Estado: aceptada
- Fecha: 2026-09-07
- Evidencia: `docs/analisis_material.md`, secciones 2 y 3

## Contexto

El enunciado pide una curva de la altura del pretil en el tiempo y penaliza el parpadeo.
Las dos cosas suponen que el video es continuo, y el material no lo es:

- `video_01` tiene tres tomas, con cortes en t=1.50 s y t=7.13 s, y la cámara se mueve
  entre ellas.
- `video_02` a `video_04` no tienen cortes, pero pasan de noche a día y de vuelta a noche
  en diez segundos, con luminancia media entre 19 y 163.
- La luz cambia dentro de cada archivo: entre 53 % y 57 % de los frames de los clips de
  720p son nocturnos.

Un corte invalida los tracks, la estimación del plano de suelo y cualquier filtro temporal.
Si se ignora, aparecen cosas que parecen mediciones y no lo son: tracks que saltan de una
máquina a otra, distancias entre equipos que nunca estuvieron en la misma escena, una curva
que promedia pretiles distintos.

## Decisión

**Dividir cada video en tomas y analizar por toma.**

- Un corte es un salto de más de 20 unidades de luminancia media entre frames seguidos
  (`bermguard/vision/shots.py`). Los cortes de `video_01` superan ese valor con holgura y
  las transiciones graduales se mueven unas pocas unidades por frame.
- En cada corte el orquestador llama a `reset()` en el tracker, el segmentador, el
  estimador de altura y la proximidad. `reset()` es parte de las interfaces, así que
  arrastrar estado entre tomas rompe el contrato.
- La curva de altura se corta en cada cambio de toma en vez de unir escenas distintas.

**Clasificar la luz por frame** en `day`, `dusk` o `night` según la luminancia media:

- Es la unidad correcta, porque la luz cambia dentro del archivo.
- Permite desglosar las métricas por condición. Un promedio global lo dominaría la noche.
- Es la misma señal que ajusta el realce de contraste de la rama del pretil.

## Alternativas descartadas

- **Tratar cada archivo como una toma continua.** Es lo habitual y produce los errores
  descritos arriba.
- **Detectar cortes por cambio estructural**, como los detectores de escena habituales. En
  este material los cortes son entre tomas de encuadre parecido, que cambian poco la
  estructura y mucho la exposición; en `video_01` dos de los tres cortes quedaban justo en
  el umbral.
- **Clasificar la luz por archivo.** Ningún archivo tiene una sola condición. El más
  parejo, `video_01`, tiene 58 % día, 30 % noche y 12 % crepúsculo.

## Consecuencias

- Una toma corta da poca información para estimar la escala. Si no alcanza, la altura queda
  vacía en vez de arrastrarse desde la toma anterior.
- La curva de altura tiene huecos en los cortes. Es lo correcto: el hueco avisa que la
  escena cambió.
- Los umbrales de luz están en `configs/` y los ajusté con el material de muestra. No sé
  cómo se comportan con los videos de evaluación.
