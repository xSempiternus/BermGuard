# ADR 0005: El detector recibe el frame sin procesar

- Estado: aceptada
- Fecha: 2026-09-08
- Relacionado: ADR 0004

## Contexto

La luminancia media del material va de 19 a 163, y más de la mitad de los frames de tres
clips son de noche. Realzar el contraste (CLAHE, corrección gamma) mejora mucho la
visibilidad.

La pregunta es dónde aplicarlo. Un modelo tiene que ver en inferencia lo mismo que vio en
entrenamiento: si el pipeline realza el frame antes del detector, los datos de
entrenamiento tienen que estar realzados igual. Había que decidirlo antes de anotar, porque
define qué imágenes se anotan.

## Decisión

**El detector recibe el frame tal como sale del video.** El realce se aplica solo en la
rama del pretil, y por eso los frames del ADR 0004 se anotaron sin procesar.

Razones:

- **El aumento de datos cubre lo mismo.** El entrenamiento varía el brillo y la saturación
  más de lo normal (`hsv_v=0.6`, `hsv_s=0.5`), según el rango medido del material. Así el
  modelo aprende a tolerar los cambios de luz en vez de depender de un preprocesado.
- **Atar el detector al preprocesado lo hace frágil.**  El pretil sí necesita
  el realce, porque es un método de gradientes y sin contraste no tiene señal, pero esa
  dependencia no tiene por qué pasar al detector.

## Alternativas descartadas

- **Realzar antes del detector y entrenar con frames realzados.** Probablemente ayudaría de
  noche, pero ata el detector al preprocesado y agrega otra calibración que no puedo
  validar contra el conjunto ciego.
- **Realzar antes del detector sin reentrenar.** Crearía justo el desajuste entre
  entrenamiento e inferencia que se quiere evitar.
- **Dos detectores, uno de día y otro de noche**, elegidos con la clasificación de luz del
  ADR 0003. Podría rendir mejor, pero 171 frames divididos en dos no alcanzan para ninguno.

## Consecuencias

- Las máquinas que no se distinguen en el frame sin procesar no se anotan
  (`docs/guia_anotacion.md`). Si yo no las distingo en los mismos píxeles que ve el modelo,
  la etiqueta sería una suposición.
- El rendimiento nocturno del detector depende del aumento de datos. No alcancé a medir el
  detector por condición de luz, así que esta apuesta sigue sin verificarse.
- El pipeline mantiene dos versiones del frame: la original para el detector y el OSD, y la
  realzada para el pretil.
