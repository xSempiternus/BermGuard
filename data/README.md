# Datos

## Qué se versiona y qué no

```
data/
├── raw/                     videos de origen             — NO versionado
├── labeling/                frames extraidos para anotar — NO versionado
├── export/                  exportacion de la anotacion  — NO versionado
└── dataset/
    ├── labels/train, val    las anotaciones              — SI versionado
    └── images/train, val    los frames                   — NO versionado
```

El criterio es uno solo: **se versiona lo que no se puede recuperar.**

Las **anotaciones** son dos horas de trabajo manual. No se regeneran de ninguna forma
y pesan 166 KB, así que viajan con el repositorio.

Los **frames** sí se regeneran, de forma determinista: `prepare_labeling_set.py` con la
semilla fijada produce exactamente los mismos 171. Y son material del cliente — este
repositorio es público, y redistribuir su metraje no es una decisión que corresponda
tomar acá.

Con las anotaciones versionadas y los scripts, el entrenamiento es **reproducible sin
necesidad de publicar el metraje**.

## Reproducir el conjunto de entrenamiento

Partiendo de los videos de origen en `data/raw/`:

```bash
# 1. Regenera los 171 frames. La semilla los hace identicos a los anotados.
python scripts/prepare_labeling_set.py data/raw --count 200 --seed 0

# 2. Copia los frames junto a las anotaciones ya versionadas.
#    Los nombres coinciden: video_02_f00120.jpg <-> video_02_f00120.txt
```

Las anotaciones versionadas provienen de una exportación de la herramienta de
anotación, que reescribe los nombres añadiendo un hash
(`video_02_f00120_jpg.rf.<hash>.txt`). El emparejamiento se hace por el prefijo
`video_NN_fNNNNN`, que es lo que `split_dataset.py` usa para agrupar.

## La partición

`data/dataset` la produce `scripts/split_dataset.py`, y **no es aleatoria**:

```bash
python scripts/split_dataset.py data/export --out data/dataset --val video_04
```

`video_04` se reserva íntegro para validación. Frames del mismo video están
fuertemente correlacionados —el frame 60 y el 66 comparten escena, iluminación,
encuadre y máquinas casi en la misma posición—, de modo que una partición aleatoria
mediría memorización en lugar de generalización.

`video_04` y no `video_03` porque este último aporta sólo 4 instancias de bulldozer, y
un mAP calculado sobre 4 objetos no es una medición: fallar uno lo mueve un 25 %.

## El pipeline no lee de aquí

En producción recibe el directorio de entrada por `--input`, tal como documenta el
`README.md` de la raíz. `data/` existe únicamente para el desarrollo y el
entrenamiento.
