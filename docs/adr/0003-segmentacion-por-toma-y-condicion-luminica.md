# ADR 0003 — Segmentación por toma y clasificación lumínica por frame

- **Estado:** aceptada
- **Fecha:** 2026-09-07
- **Evidencia:** `docs/analisis_material.md`, secciones 2 y 3

## Contexto

El enunciado pide una curva temporal de la altura del pretil a lo largo de la secuencia, y
penaliza explícitamente el parpadeo. Ambas cosas presuponen continuidad temporal.

La medición del material muestra que esa presunción no se sostiene:

- `video_01` contiene **tres tomas** separadas por cortes duros en t=1.50 s y t=7.13 s, con
  reposicionamiento de cámara entre ellas.
- `video_02` a `video_04` no tienen cortes, pero recorren una transición gradual noche → día
  pleno → noche dentro de diez segundos, con la luminancia media oscilando entre 19 y 163.
- La condición lumínica varía **dentro** de cada archivo. Entre el 53 % y el 57 % de los
  frames de los tres clips de 720p son nocturnos.

Un corte invalida las identidades de tracking, la calibración del plano de suelo y el estado
de cualquier filtro temporal. Ignorarlo produce artefactos que parecen mediciones: tracks que
saltan de una máquina a otra, distancias entre equipos que nunca estuvieron en la misma
escena, y una curva de altura continua que en realidad promedia pretiles distintos.

## Decisión

**Segmentar cada video en tomas y tratar la toma como la unidad de análisis.**

- Los cortes se detectan combinando salto de luminancia media entre frames consecutivos con
  cambio estructural. El primer criterio es el que decide en este material: un corte entre
  dos tomas de encuadre similar altera poco la estructura y mucho la exposición, de modo que
  un detector basado sólo en diferencia estructural lo pasa por alto.
- En cada frontera de toma, el orquestador invoca `reset()` sobre el tracker, el segmentador
  de pretil, el estimador de altura y el analizador de proximidad. Ese método forma parte de
  los `Protocol` correspondientes, de modo que arrastrar estado a través de un corte
  constituye una violación de contrato y no un descuido de implementación.
- La curva temporal de altura se reporta **por segmento**, con discontinuidad explícita en
  los cortes, en lugar de una línea continua que uniría escenas sin relación.

**Clasificar la condición lumínica por frame, no por archivo**, en `day`, `dusk` o `night`
según la luminancia media.

- Es la unidad correcta, porque la condición cambia dentro del archivo.
- Es el estratificador del benchmark: todas las métricas se reportan desglosadas por
  condición, ya que un promedio global quedaría dominado por el caso nocturno mientras
  aparenta describir el sistema completo.
- Es además la señal que gobierna el acondicionamiento adaptativo del frame previo a la
  inferencia.

## Alternativas consideradas

**Tratar cada archivo como una toma continua.** Es el supuesto por defecto y el que produce
los artefactos descritos. Se descarta por medición directa, no por principio.

**Detectar cortes sólo por cambio estructural** (el criterio de escena convencional). Se
probó sobre este material y no detecta las transiciones de `video_02` a `video_04`, además de
depender de un umbral que en `video_01` marca dos de tres cortes de forma marginal. La
luminancia resultó ser la señal discriminante acá.

**Etiquetar la condición lumínica a nivel de archivo.** Descartada porque ningún archivo tiene
una condición única: el más homogéneo, `video_01`, reparte 58 % día, 30 % noche y 12 %
crepúsculo.

## Consecuencias

- Una toma corta deja poco material para estimar el plano de suelo y la escala. Cuando la
  evidencia no alcanza, el estimador devuelve ausencia de medición en lugar de un valor
  extrapolado desde una toma anterior.
- La curva de altura presenta huecos en los cortes. Es el comportamiento correcto y se
  documenta como tal: un hueco declarado informa más que una interpolación que oculta que
  la escena cambió.
- Los cortes de clasificación lumínica se fijan en `configs/` y son revisables. Están
  calibrados sobre el material de muestra, y el set de evaluación es ciego, de modo que su
  generalización es un supuesto declarado y no un hecho verificado.
