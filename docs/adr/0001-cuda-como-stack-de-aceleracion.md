# ADR 0001 — CUDA como stack de aceleración, con degradación a CPU

- **Estado:** aceptada
- **Fecha:** 2026-09-06

## Contexto

El enunciado del desafío es ambiguo respecto al entorno de ejecución, y la ambigüedad es
interna al propio documento.

Por un lado, los requisitos tecnológicos exigen que el sistema *"funcione out-of-the-box y
use GPU (CUDA o ROCm o OPENVINO…)"*. Por otro, el único comando de referencia publicado es:

```bash
docker run --rm -v /ruta/local/test:/app/test -v /ruta/local/output:/app/output \
  imagen \
  python main.py --input /app/test --output /app/output --method 1
```

Ese comando no incluye `--gpus`, por lo que ejecutado literalmente levanta un contenedor sin
acceso al dispositivo NVIDIA. Como la evaluación se describe como automatizada sobre un set
de videos oculto, no hay forma de saber de antemano si el comando se ejecutará tal cual, si
se le añadirá `--gpus all`, o qué hardware tendrá la máquina evaluadora.

Hay además una asimetría de riesgo: `docker run --gpus all` **falla al arrancar** si el host
no tiene configurado el runtime de NVIDIA, mientras que una imagen construida sobre una base
CUDA se ejecuta sin problemas en una máquina sin GPU — simplemente `torch.cuda.is_available()`
devuelve `False`.

## Decisión

**CUDA es el stack de aceleración objetivo.**

- Imagen base `nvidia/cuda:*-cudnn-runtime-ubuntu22.04`.
- PyTorch instalado con wheels CUDA, versión fijada en `requirements.txt`.
- Inferencia en FP16 cuando hay GPU disponible. El hardware de desarrollo es Ampere
  (compute capability 8.6), que tiene Tensor Cores y por tanto se beneficia realmente de
  media precisión; no es una optimización declarativa.

**La ambigüedad del `--gpus` se resuelve por selección de dispositivo en runtime, no
duplicando implementaciones.** La CLI expone `--device auto|cuda|cpu`, con `auto` por
defecto: resuelve a CUDA si hay dispositivo disponible y cae a CPU registrando un `WARNING`
explícito en el log, de modo que en la salida quede constancia de que el modo degradado se
activó de forma consciente y no por un fallo silencioso.

Existe un único camino de código. El dispositivo es un parámetro, no una rama de diseño.

## Alternativas consideradas

**ROCm.** Descartada por ausencia de hardware AMD para desarrollar y validar. Ofrecer un
camino de ejecución que nunca se probó es peor que no ofrecerlo: introduce una promesa que el
sistema no puede sostener.

**OpenVINO.** Es el candidato más sólido si el objetivo declarado fuera CPU, y el enunciado lo
admite explícitamente. Se descarta por costo de oportunidad: introduce un runtime y un formato
de modelo distintos, sin experiencia previa del autor, dentro de una ventana de desarrollo de
cinco días. El riesgo de integración no se compensa con la ganancia esperada.

**CUDA estricto, sin degradación.** Descartada por la asimetría de costos descrita arriba. Si
el evaluador ejecuta el comando literal del enunciado, un contenedor que exige GPU no genera
ningún artefacto, lo que compromete por completo el criterio de robustez de despliegue. El
fallback se implementa en unas pocas líneas de selección de dispositivo.

**Perfil de ejecución reducido para CPU** (modelo más pequeño, resolución de análisis menor).
Descartada por innecesaria: los clips de trabajo son del orden de 250 frames, de modo que el
modo CPU completa una corrida en tiempos aceptables sin degradar la calidad del análisis.
Añadir un segundo perfil habría introducido una fuente de divergencia entre lo que se mide y
lo que se entrega, a cambio de nada.

## Consecuencias

- La imagen final es pesada, del orden de varios GB, por la base CUDA. Se documenta el tamaño
  y el tiempo de construcción en el README para que el evaluador sepa qué esperar.
- El modo CPU es sensiblemente más lento. El reporte de benchmark incluye ambas mediciones en
  columnas separadas, en lugar de reportar un único número sin contexto de hardware.
- El sistema no aprovecha aceleración específica de hardware no-NVIDIA. Es una limitación
  aceptada y declarada, no una omisión.
- La exportación a ONNX Runtime queda como línea de trabajo secundaria, orientada a comparar
  rendimiento dentro del benchmark (PyTorch FP32 → FP16 → ONNX), y no como camino de
  inferencia principal.

## Notas

La contradicción del enunciado se explicita en el README. El propio documento del desafío
instruye a *"asumir el criterio de ingeniería más sólido, seguro y justificado, documentando
las decisiones en el README.md"* ante ambigüedades; este ADR y su resumen en el README son
la respuesta a esa instrucción.
