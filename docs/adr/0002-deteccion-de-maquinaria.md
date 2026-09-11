# ADR 0002: Detección de maquinaria con fine-tuning en vez de zero-shot

- Estado: aceptada
- Fecha: 2026-09-07
- Evidencia: `docs/analisis_material.md`, secciones 4 y 5

## Contexto

El sistema tiene que distinguir camiones de extracción (CAEX) de bulldozers. La proximidad
mide distancias entre equipos, así que si dos máquinas salen en una sola caja no hay nada
que medir.

Antes de pensar en anotar, probé dos opciones sin entrenamiento.

**YOLO11s preentrenado en COCO.** COCO no tiene maquinaria minera. El modelo detecta
`truck` de forma consistente (confianza media 0.55–0.69), pero en el frame 120 de
`video_02`, con un CAEX y un bulldozer separados, pone una sola caja de 622 px sobre los
dos. Subir la resolución da cajas superpuestas e inestables, no separadas.

**YOLO-World (`yolov8s-worldv2`), zero-shot.** Probé siete conjuntos de prompts, desde
`"bulldozer"` hasta `"yellow tracked bulldozer with blade"`. Siempre sale una caja
fusionada, y la confianza baja mientras más específico es el prompt (0.56, 0.34, 0.20). Con
umbral 0.03 aparecen las cajas correctas (0.141 el CAEX, 0.089 el bulldozer), pero la
fusionada gana con 0.562. Y con el bulldozer solo, sin competencia, tres prompts distintos
dan cero detecciones con umbral 0.05.

Lo primero se podría intentar arreglar con umbrales. Lo segundo no: si el texto no calza con
lo que el modelo ve, ningún hiperparámetro lo hace aparecer.

## Decisión

Hacer fine-tuning de un detector con imágenes del dominio, algo que el enunciado permite.
Con qué datos se entrenó lo decide el ADR 0004.

Los resultados zero-shot quedan como la primera columna del benchmark de detección
(zero-shot contra fine-tuning), que es uno de los ejes que sugiere el enunciado.

## Alternativas descartadas

- **Bajar umbrales y ajustar el NMS del modelo COCO.** Las cajas correctas están en 0.089 y
  0.141. Con umbrales tan bajos entraría el ruido (los falsos positivos sobre polvo están
  entre 0.28 y 0.41), y además quedaría calibrado para cuatro videos.
- **Más ingeniería de prompts.** El problema no es la redacción: tres formulaciones sin
  competencia y con umbral casi cero no detectan nada.
- **Detectar solo CAEX y declararlo como limitación.** Eliminaría un requisito del
  enunciado y con él la proximidad, cuando había un camino viable.

## Consecuencias

- El proyecto suma una etapa de entrenamiento y necesita una validación honesta, separada
  por video (ADR 0004).
- El modelo queda especializado en este dominio y va a funcionar peor fuera de él. Bajo el contexto de seguridad
  industrial vale mas la pena conocer la falla que el estado general.
- Cambiar el detector no tocó el orquestador, porque el nuevo implementa el mismo
  `IDetector`.