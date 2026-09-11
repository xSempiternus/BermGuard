# ADR 0001: CUDA como aceleración, con respaldo en CPU

- Estado: aceptada
- Fecha: 2026-09-06

## Contexto

Los requisitos piden que el sistema *"funcione
out-of-the-box y use GPU (CUDA o ROCm o OPENVINO…)"*, pero el único comando de referencia
no incluye `--gpus`:

```bash
docker run --rm -v /ruta/local/test:/app/test -v /ruta/local/output:/app/output \
  imagen \
  python main.py --input /app/test --output /app/output --method 1
```

Ejecutado tal cual, el contenedor no ve la GPU. Como la evaluación es automática y con
videos ocultos, no sé si se usará ese comando, si se le agregará `--gpus all` ni qué
hardware habrá.

`--gpus all` hace fallar el arranque si el host no tiene el
runtime de NVIDIA, mientras que una imagen con base CUDA corre sin problemas en una máquina
sin GPU (`torch.cuda.is_available()` simplemente devuelve `False`).

## Decisión

Usar CUDA como aceleración:

- Imagen base `nvidia/cuda:*-cudnn-runtime-ubuntu22.04`.
- PyTorch con wheels de CUDA, con la versión fija en `requirements.txt`.
- Inferencia en FP16 cuando hay GPU. Mi equipo es Ampere (compute capability 8.6), con
  Tensor Cores, así que FP16 sí acelera.

El dispositivo se elige al ejecutar, sin duplicar código. La CLI tiene
`--device auto|cuda|cpu`. Con `auto`, que es el valor por defecto, usa CUDA si hay GPU y si
no pasa a CPU dejando un `WARNING` en el log, para que quede claro que el modo degradado
fue intencional y no un error. Hay un solo camino de código y el dispositivo es un
parámetro.

## Alternativas descartadas

- **ROCm.** No tengo hardware AMD para probarlo.
- **OpenVINO.** Sería la mejor opción si el objetivo fuera CPU. Lo descarté porque suma otro
  runtime y otro formato de modelo que no conozco, con cinco días de desarrollo.
- **Solo CUDA, sin respaldo.** Si el evaluador usa el comando literal, un contenedor que
  exige GPU no genera nada. El respaldo son unas pocas líneas.
- **Un perfil liviano para CPU** (modelo más chico, menor resolución). No hace falta: los
  clips son cortos y la CPU los procesa en un tiempo razonable segun lo medido.
## Consecuencias

- La imagen pesa 15.7 GB, casi todo por la base CUDA. El tamaño está en el README.
- En CPU es más lento. El reporte mide los dos modos por separado (sección 5): a 720p,
  8.1 fps en CPU y 15.0 en GPU.
- No aprovecha aceleradores que no sean NVIDIA. Es una limitación asumida.
- ONNX Runtime queda como trabajo futuro para el benchmark, no como camino principal.

El enunciado pide:  *"asumir el criterio de ingeniería más sólido, seguro
y justificado, documentando las decisiones en el README.md"*. Este ADR y su resumen en el
README responden a eso.
