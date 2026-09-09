# ADR 0005 — El detector opera sobre frames sin acondicionar

- **Estado:** aceptada
- **Fecha:** 2026-09-08
- **Relacionado:** ADR 0004 (los datos de entrenamiento salen de esta decisión)

## Contexto

El material recorre luminancias medias entre 19 y 163, y más de la mitad de los frames de tres
de los cuatro clips son nocturnos. El acondicionamiento de imagen —ecualización adaptativa de
contraste, corrección gamma— mejora sustancialmente la visibilidad en ese rango.

La pregunta es dónde aplicarlo, y no es una cuestión de gusto. Un modelo debe encontrar en
inferencia la misma distribución de entradas que vio en entrenamiento. Si el pipeline
acondiciona el frame antes del detector, el conjunto de entrenamiento tiene que estar
acondicionado del mismo modo.

La decisión debía tomarse **antes** de anotar, porque determina qué imágenes se anotan y
rehacer ese trabajo cuesta horas.

## Decisión

**El detector recibe el frame tal como sale del decodificador.** El acondicionamiento se
aplica únicamente en la rama de segmentación del pretil.

En consecuencia, el conjunto de entrenamiento del ADR 0004 se compone de frames sin
acondicionar, que es lo que `scripts/prepare_labeling_set.py` extrae.

Dos razones:

**La augmentación cubre lo que el acondicionamiento resolvería.** El entrenamiento aplica
variación de brillo y saturación por encima de los valores habituales (`hsv_v=0.6`,
`hsv_s=0.5`), calibrada contra el rango medido del material. Eso enseña invarianza a la
iluminación dentro del modelo, en lugar de imponerla desde fuera. Un modelo invariante es
preferible a uno que depende de que su preprocesado se comporte igual siempre.

**Acoplar el detector al preprocesado lo vuelve frágil.** Si el detector se entrena sobre
frames con una configuración concreta de ecualización, cambiar un parámetro de esa
configuración deja el modelo desalineado con su entrada y obliga a reentrenar. La rama del
pretil sí necesita ese acondicionamiento porque es un método clásico basado en gradientes, que
sin contraste no tiene señal; pero esa dependencia no debe propagarse al detector.

## Alternativas consideradas

**Acondicionar antes del detector y entrenar sobre frames acondicionados.** Probablemente
mejoraría la detección nocturna a corto plazo. Se descarta por el acoplamiento descrito, y
porque quedaría una segunda calibración implícita —la del preprocesado— que también habría que
validar contra el conjunto ciego sin poder medirla.

**Acondicionar antes del detector sin reentrenar.** Descartada sin discusión: introduciría un
desajuste entre entrenamiento e inferencia, que es precisamente el error que esta decisión
existe para evitar.

**Entrenar dos detectores, uno diurno y uno nocturno, seleccionados por la clasificación
lumínica del ADR 0003.** Es defendible y podría rendir mejor. Se descarta por presupuesto de
datos: dividir un conjunto de 171 frames anotados en dos lo deja sin material suficiente para
ninguno de los dos.

## Consecuencias

- Las máquinas que resultan indistinguibles en el frame sin acondicionar no se anotan, según
  establece `docs/guia_anotacion.md`. Es coherente: si el anotador no las distingue sobre los
  mismos píxeles que verá el modelo, la etiqueta sería una suposición.
- El rendimiento nocturno del detector depende enteramente de la augmentación durante el
  entrenamiento. Es una apuesta declarada, y su verificación es una fila del reporte de
  benchmark: las métricas se reportan segmentadas por condición lumínica precisamente para
  poder comprobarla o refutarla.
- El pipeline mantiene dos rutas de imagen desde el mismo frame: sin acondicionar hacia el
  detector, acondicionada hacia la segmentación del terreno. El orquestador debe conservar el
  frame original, y de hecho el renderizado del OSD también dibuja sobre él y no sobre la
  versión analizada.
