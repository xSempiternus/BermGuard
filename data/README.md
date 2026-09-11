# Datos

## Qué se versiona

```
data/
├── raw/                     videos originales             NO versionado
├── labeling/                frames extraídos para anotar  NO versionado
├── export/                  exportación de la anotación   NO versionado
└── dataset/
    ├── labels/train, val    anotaciones                   SÍ versionado
    └── images/train, val    frames                        NO versionado
```

La regla es versionar lo que no se puede regenerar.

Las **anotaciones** fueron dos horas de trabajo manual y pesan 166 KB, así que van en el
repositorio.

Los **frames** se regeneran de forma determinista con `prepare_labeling_set.py` y la
semilla fija. Además son material del cliente, y este repositorio es público, así que no me
corresponde redistribuirlos. Con las anotaciones y los scripts, el entrenamiento se puede
reproducir sin publicar los videos.

## Reproducir el conjunto de entrenamiento

Con los videos originales en `data/raw/`:

```bash
# 1. Regenera los mismos 171 frames que se anotaron.
python scripts/prepare_labeling_set.py data/raw --count 200 --seed 0

# 2. Copia los frames junto a las anotaciones versionadas.
#    Los nombres coinciden: video_02_f00120.jpg <-> video_02_f00120.txt
```

Al exportar, la herramienta de anotación agrega un hash a los nombres
(`video_02_f00120_jpg.rf.<hash>.txt`). El emparejamiento se hace por el prefijo
`video_NN_fNNNNN`, igual que en `split_dataset.py`.

## Partición

```bash
python scripts/split_dataset.py data/export --out data/dataset --val video_04
```

No es aleatoria. `video_04` queda completo para validación, porque los frames de un mismo
video son casi iguales y una partición aleatoria mediría memoria. Elegí `video_04` y no
`video_03` porque este solo tiene 4 bulldozers.

## El pipeline no lee de acá

En producción recibe la carpeta de entrada por `--input`. `data/` es solo para desarrollo y
entrenamiento.
