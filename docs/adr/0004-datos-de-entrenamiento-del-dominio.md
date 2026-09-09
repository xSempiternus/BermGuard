# ADR 0004 — Datos de entrenamiento propios por sobre datasets públicos

- **Estado:** aceptada
- **Fecha:** 2026-09-08
- **Depende de:** ADR 0002 (la decisión de hacer fine-tuning)

## Contexto

El ADR 0002 estableció que el detector debe especializarse en el dominio. Queda por decidir
con qué datos.

El enunciado del desafío fomenta explícitamente el uso de datasets públicos y nombra Roboflow
Universe. Se evaluaron tres candidatos, con el criterio de correspondencia de dominio que el
fracaso del enfoque zero-shot dejó como lección: lo que hundió a YOLO-World no fue la falta de
capacidad, sino la distancia entre su distribución de entrenamiento y este material.

| Dataset | Clases | Imágenes | Licencia | Evaluación |
|---|---|---|---|---|
| `brancos-workspace/mining-truck-detection-w53tl` | 1 (`haul_truck`) | 264 | CC BY 4.0 | No contiene bulldozer. Aporta únicamente la clase que el modelo base ya detecta con solvencia. Parte de las imágenes son augmentaciones del mismo frame |
| `mohamads-workspace-rl4ui/bulldozer-lpoe2` | 7, incluye `Bull_dozer` | 1.324 | CC BY 4.0 | Cobertura de clases correcta. Las imágenes son fotografías de catálogo: la máquina ocupa el encuadre, bien iluminada y sin oclusión |
| `rf100-vl/apoce-...-construction-equipment` | 7, incluye `bulldozer` | 928 | MIT | Fotografía aérea cenital desde dron sobre obra urbana. Los objetos aparecen vistos desde arriba y a escala muy reducida |

El material objetivo tiene características que ninguno de los tres reproduce: vista oblicua
desde posición elevada, distancia media, entorno desértico con polvo en suspensión, y entre
un 53 % y un 57 % de frames nocturnos según la medición de `docs/analisis_material.md`.

Las diferencias no son de matiz. El dataset de catálogo enseña máquinas grandes y centradas;
el aéreo enseña un ángulo de observación que este material no contiene en ningún frame.

## Decisión

**Entrenar con frames anotados del propio material, como fuente principal y suficiente.**

Se extraen y anotan aproximadamente 200 frames mediante
`scripts/prepare_labeling_set.py`, con muestreo estratificado por condición lumínica.

Dos consideraciones sostienen que ese volumen basta:

- El entrenamiento parte de pesos preentrenados. El aporte del preentrenamiento son las
  representaciones genéricas de bajo y medio nivel —bordes, texturas, la noción de vehículo—,
  no la lista de clases. Lo que hay que enseñar es la vista específica de dos máquinas
  concretas, no qué es un objeto.
- El material es homogéneo: una misma faena, las mismas máquinas, encuadres similares. En
  otro contexto sería una advertencia de sobreajuste; acá el conjunto de evaluación ciego
  proviene con alta probabilidad del mismo material, de modo que la estrechez del dominio
  juega a favor.

**Las clases son `caex` y `bulldozer`**, categorías operativas de faena y no categorías
visuales genéricas. Es la distinción que ningún dataset público ofrece: todos etiquetan
`truck` o `vehicle`, cuando lo que el sistema necesita separar son dos roles distintos dentro
de la plataforma de descarga.

## Alternativas consideradas

**Entrenar con `mohamads-workspace/bulldozer-lpoe2`.** Es el de mejor cobertura de clases. Se
descarta como fuente principal porque enseñaría a esperar máquinas grandes, centradas y bien
iluminadas, que es precisamente el sesgo que hay que evitar. Queda como experimento
secundario: preentrenar con él y afinar después con los frames propios es una comparación
medible y, de realizarse, constituye una fila del reporte de benchmark.

**Entrenar con APOCE.** Descartado por ángulo de observación. Una vista cenital y una vista
oblicua producen siluetas distintas de la misma máquina; entrenar con la primera para inferir
sobre la segunda es una transferencia que no tiene por qué darse.

**Combinar los tres datasets públicos sin datos propios.** Se descarta porque sumar volumen no
corrige el sesgo: 2.500 imágenes fuera de dominio siguen estando fuera de dominio, y el
resultado sería un modelo que rinde bien en una validación que no se parece al despliegue.

## Consecuencias

- El etiquetado manual es trabajo del desarrollador, del orden de dos horas. Es el costo
  aceptado a cambio de datos que sí corresponden al problema.
- El modelo resultante queda estrechamente especializado y no generaliza a otras faenas ni a
  otros ángulos de cámara. Es una limitación declarada, no descubierta: el sistema se entrega
  con el alcance de sus datos documentado.
- La partición de validación **no puede ser aleatoria**. Frames del mismo video están
  fuertemente correlacionados, y una partición aleatoria mediría memorización en lugar de
  generalización. Se separa por video: uno de los clips se reserva íntegro para validación.
- Los tres datasets evaluados y descartados se documentan en el reporte de benchmark con su
  motivo. Un descarte justificado por medición es información; una omisión silenciosa no lo es.
