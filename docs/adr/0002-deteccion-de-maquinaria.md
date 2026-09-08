# ADR 0002 — Detección de maquinaria: fine-tuning por sobre zero-shot

- **Estado:** aceptada
- **Fecha:** 2026-09-07
- **Evidencia:** `docs/analisis_material.md`, secciones 4 y 5

## Contexto

El sistema debe distinguir camiones de extracción (CAEX) de maquinaria de apoyo
(bulldozers). No es una distinción cosmética: el módulo de proximidad mide distancias
**entre equipos**, de modo que tratar dos máquinas como una sola entidad no degrada la
precisión, sino que elimina la magnitud a medir.

Se evaluaron dos enfoques sin entrenamiento antes de considerar cualquier alternativa que
requiriera etiquetado.

### Modelo preentrenado en COCO

COCO no contiene clases de maquinaria minera. Sobre el material de muestra, YOLO11s activa
`truck` de forma consistente (confianza media 0.55–0.69) y nada más de forma relevante.

El conteo agregado sugería éxito. La inspección visual mostró lo contrario: en el frame 120
de `video_02`, con un CAEX y un bulldozer visualmente separados, el modelo emite **una sola
caja de 622 px que engloba ambas máquinas**. Aumentar la resolución de inferencia produce
cajas superpuestas e inestables, no separación.

### Modelo open-vocabulary zero-shot

Se evaluó YOLO-World (`yolov8s-worldv2`) con siete conjuntos de prompts, de `"bulldozer"` a
`"yellow tracked bulldozer with blade"`. Resultado: siempre una caja, siempre fusionada, con
la confianza **decreciendo** al aumentar la especificidad del prompt (0.56 → 0.34 → 0.20).

Bajando el umbral a 0.03 se comprobó que la descomposición correcta existe en el conjunto de
propuestas —una caja sobre el CAEX a 0.141 y otra sobre el bulldozer a 0.089— pero la
fusionada la domina con 0.562.

Y con el bulldozer aislado, sin competencia, tres redacciones distintas devuelven **cero
detecciones** a umbral 0.05.

Son dos fallos independientes. El primero es de ranking y en principio atacable; el segundo
no lo es: si el embedding de texto no coincide con la evidencia visual, ningún ajuste de
hiperparámetros lo hace activar. El vocabulario abierto condiciona las features con texto,
pero la propuesta de región sigue dominada por el backbone visual.

## Decisión

**Fine-tuning de un detector sobre imágenes del dominio**, empleando datasets públicos de
minería y construcción.

El enunciado del desafío permite y fomenta explícitamente el uso de datasets públicos
adicionales para entrenamiento y transfer learning, nombrando Roboflow Universe y los
datasets de minería y construcción.

El resultado zero-shot **no se descarta**: constituye la primera columna del benchmark de
detección, con el eje `modelo fundacional zero-shot` contra `arquitectura especializada
fine-tuneada`. Es uno de los ejes que el propio enunciado propone como ejemplo.

## Alternativas consideradas

**Ajustar umbrales y NMS sobre el modelo COCO.** Las cajas correctas existen a 0.089 y 0.141,
por debajo de la fusionada. Recuperarlas exigiría umbrales tan bajos que el ruido —los falsos
positivos sobre penachos de polvo, ya medidos entre 0.28 y 0.41— pasaría en masa. Sería
además un ajuste calibrado sobre cuatro videos de muestra, sin garantía alguna sobre el set
de evaluación ciego.

**Ingeniería de prompts adicional.** Descartada por la evidencia del aislamiento: el fallo no
es de redacción. Tres formulaciones distintas, sin competencia y con el umbral casi en cero,
no producen ninguna detección.

**Aceptar sólo la detección de CAEX y documentarlo como limitación.** Descartada porque
suprimiría un requisito explícito del enunciado y, con él, el módulo de proximidad completo.
Documentar una limitación es legítimo cuando no existe camino viable; acá existe.

## Consecuencias

- Se incorpora una etapa de entrenamiento al proyecto, con su costo de tiempo y su necesidad
  de un conjunto de validación honesto.
- El modelo resultante queda especializado en este dominio y se degradará fuera de él. Es el
  intercambio aceptado: se gana control y se puede caracterizar el modo de falla, que en un
  sistema de seguridad industrial vale más que la generalidad.
- La sustitución del detector no requiere cambios en el orquestador ni en ningún otro módulo:
  la implementación nueva satisface el mismo `IDetector`. Este ADR es la primera comprobación
  práctica de esa decisión de arquitectura.
- El pipeline continúa desarrollándose entretanto con el detector COCO, que localiza el CAEX
  de forma fiable. El trabajo aguas abajo no queda bloqueado por el entrenamiento.
