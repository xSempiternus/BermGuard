# Datos

## Qué se versiona

```
data/
├── raw/                     videos originales             NO versionado
├── labeling/                frames extraídos para anotar  NO versionado
├── export/                  exportación de la anotación   NO versionado
├── dataset/
│   ├── labels/train, val    anotaciones                   SÍ versionado
│   └── images/train, val    frames                        NO versionado
├── auditoria_etiquetas.json correcciones de etiquetas     SÍ versionado
├── cv/                      folds de la validación cruzada NO versionado
├── externos/                datasets públicos descargados NO versionado
│   ├── bulldozer/           bulldozer-lpoe2, export YOLO
│   └── mines/               Mines, armado con import_universe.py
├── pretrain_excluir.txt     imágenes públicas excluidas   SÍ versionado
└── pretrain/                set de preentrenamiento       NO versionado
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

## Etiquetas que mezclan cajas y polígonos

36 de los 40 frames de `video_01` tienen en el mismo archivo cajas (pre-anotaciones sin
corregir) y polígonos. Ultralytics 8.4 descarta esos frames enteros. Las etiquetas
versionadas se dejan así, porque la auditoría identifica cada corrección por su línea
original; se homogeneizan al copiarlas a la carpeta de entrenamiento
(`bermguard/training/etiquetas.py`). `cross_validate.py` ya lo hace en cada fold;
`train_detector.py` sobre `data/dataset/data.yaml` todavía no
([`docs/entrenamiento/sesion_gpu.md`](../docs/entrenamiento/sesion_gpu.md)).

## Datasets públicos para preentrenar

Se usan dos, los dos con licencia **CC BY 4.0**, para la variante `pretrain` de la
validación cruzada (ADR 0004, revisiones del 2026-10-18 y del 2026-10-20). No se
versionan: se descargan y se regeneran.

### `data/externos/bulldozer/`: bulldozer-lpoe2

[`mohamads-workspace-rl4ui/bulldozer-lpoe2`](https://universe.roboflow.com/mohamads-workspace-rl4ui/bulldozer-lpoe2/dataset/1),
versión 1. En Universe: *Download Dataset* → formato YOLO → *download zip* (pide una
cuenta gratuita). Descomprimir el zip (`Bulldozer.v1i.yolo26.zip`) en
`data/externos/bulldozer/`: queda un `data.yaml` y las carpetas `train`, `valid` y `test`,
1.324 imágenes en total.

### `data/externos/mines/`: Mines

[`mines-project/mines-jo13g`](https://universe.roboflow.com/mines-project/mines-jo13g)
no tiene ninguna versión generada, así que no hay zip. Las anotaciones de sus imágenes con
bulldozer se juntaron desde el navegador en `data/externos/mines/anotaciones_universe.json`
(cómo, en el ADR 0004, revisión del 2026-10-20). Ese JSON no está en el repo: hay que
conservarlo o volver a juntarlo. Con él:

```bash
python scripts/import_universe.py data/externos/mines/anotaciones_universe.json \
    --out data/externos/mines
```

Descarga las imágenes, les aplica la rotación EXIF a los píxeles y escribe las etiquetas
YOLO y el `data.yaml`. Resultado: 68 imágenes y 70 dozers.

### `data/pretrain/`

```bash
python scripts/prepare_pretrain.py \
    --source data/externos/mines --source data/externos/bulldozer \
    --map dozer=bulldozer bull_dozer=bulldozer dumper=caex dumb_truck=caex \
    --require bulldozer --exclude data/pretrain_excluir.txt --repeat mines=5
```

Deja 1.068 imágenes de train (245 de Mines, que son 49 repetidas cinco veces) y 300 de
val, con el detalle por fuente en `data/pretrain/resumen.json`.
`scripts/gpu_session.py --check` rechaza un `data/pretrain/` generado sin `--exclude` o sin
`--repeat`.

`data/pretrain_excluir.txt` **se versiona**: es el resultado de una revisión a mano
(duplicados, etiquetas erróneas, juguetes), con el motivo de cada línea, y no se puede
regenerar.

## El pipeline no lee de acá

En producción recibe la carpeta de entrada por `--input`. `data/` es solo para desarrollo y
entrenamiento.
