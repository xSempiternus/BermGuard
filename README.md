# BermGuard AI

Pipeline de visión por computadora para botaderos de minería a rajo abierto. Recibe una
carpeta de videos y, por cada uno, genera un video anotado, la segmentación y la altura del
pretil, la proximidad entre equipos y los metadatos de ejecución.

Para ver el resultado, parte por `output_ejemplo/video_04/method_1/video_04_osd.mp4`: es el
único video que el detector no vio durante el entrenamiento. La comparación de métodos y las
mediciones están en [`reporte_benchmark.md`](reporte_benchmark.md).

---

## Ejecución

El comando de referencia del enunciado funciona tal cual:

```bash
docker build -t bermguard:latest .

docker run --rm \
  -v /ruta/local/test:/app/test \
  -v /ruta/local/output:/app/output \
  bermguard:latest \
  python main.py --input /app/test --output /app/output --method 1
```

Si hay GPU, agrega `--gpus all` antes del nombre de la imagen. No es obligatorio: sin GPU el
pipeline deja una advertencia en el log y sigue en CPU (ver
[ADR 0001](docs/adr/0001-cuda-como-stack-de-aceleracion.md)).

| Argumento | Valores | Descripción |
|---|---|---|
| `--input` | ruta | Carpeta con los videos. Lo que no sea video se ignora |
| `--output` | ruta | Carpeta de salida. Se crea si no existe |
| `--method` | `1`, `2`, `all` | `1` camino óptimo, `2` línea base por argmax, `all` ambos |
| `--device` | `auto`, `cuda`, `cpu` | Por defecto `auto` |
| `--max-frames` | entero | Límite de frames por video, para pruebas rápidas |
| `--log-level` | `DEBUG`…`ERROR` | Por defecto `INFO` |

Los pesos del modelo van dentro de la imagen, así que no se descarga nada al ejecutar.

### Salida

```
output/
├── run_metadata.json              argumentos, dispositivo, versiones, commit, fallos
└── <video>/method_<n>/
    ├── <video>_osd.mp4            video anotado
    ├── berm_height.{png,svg}      altura del pretil en el tiempo, con banda de incertidumbre
    ├── vehicle_spatial.{png,svg}  posiciones en planta y distancias (si hubo equipos)
    ├── berm_profile.csv           serie por frame del pretil
    ├── proximity_events.csv       posición y nivel de riesgo por equipo y frame
    ├── distance_matrix.csv        distancia mínima por par de equipos
    └── metadata.json              FPS, ms por etapa, alertas, resoluciones, jitter
```

Cada gráfico va con el CSV que lo genera, para poder rehacerlo desde los datos.

---

## Arquitectura

```
main.py          CLI: lee los argumentos, arma el pipeline y lo ejecuta
bermguard/
├── core/        tipos del dominio, interfaces (Protocol), errores, configuración
├── io/          lectura y escritura de video, metadatos y CSV
├── vision/      detección, segmentación del pretil, condición de luz, cortes
├── analytics/   geometría, altura y proximidad
├── pipeline/    orquestador, factory y dibujo del OSD
└── benchmark/   métricas para comparar métodos
```

Tres ideas ordenan el código:

- **El orquestador solo conoce interfaces.** Usa los `Protocol` de `core/interfaces.py` y
  nunca una clase concreta. Un método nuevo se agrega implementando esas interfaces y
  registrándolo en `pipeline/factory.py`. Cambiar el detector COCO por el entrenado fue
  cambiar una ruta en `configs/`.
- **Los tipos del dominio son inmutables y no dependen de librerías.** `analytics/` trabaja
  con dataclasses y arrays, así que se prueba sin GPU, sin modelo y sin video.
- **Segmentar y medir están separados.** `IBermSegmenter` encuentra el pretil en píxeles e
  `IHeightEstimator` lo pasa a metros. Pueden fallar por separado: si hay pretil pero ningún
  vehículo que dé la escala, se conserva la segmentación y la altura queda vacía.

---

## Decisiones

Las decisiones importantes están en `docs/adr/`, una por archivo. Cada ADR (registro de
decisión de arquitectura) explica el contexto, lo que se decidió, las alternativas
descartadas y las consecuencias. El número es solo el orden en que las fui tomando.

| ADR | Decisión |
|---|---|
| [0001](docs/adr/0001-cuda-como-stack-de-aceleracion.md) | CUDA como aceleración, con respaldo en CPU |
| [0002](docs/adr/0002-deteccion-de-maquinaria.md) | Entrenar un detector propio, después de medir que el zero-shot no separa las máquinas |
| [0003](docs/adr/0003-segmentacion-por-toma-y-condicion-luminica.md) | La toma es la unidad de análisis y la luz se clasifica por frame |
| [0004](docs/adr/0004-datos-de-entrenamiento-del-dominio.md) | Entrenar con frames del propio material en vez de datasets públicos |
| [0005](docs/adr/0005-el-detector-opera-sobre-frames-crudos.md) | El detector recibe el frame sin procesar; el realce de contraste es solo para el pretil |
| [0006](docs/adr/0006-segmentacion-del-pretil-por-camino-optimo.md) | La cresta del pretil se busca como camino óptimo y no con argmax por columna |
| [0007](docs/adr/0007-banda-del-pretil-anclada-a-la-maquinaria.md) | La banda de búsqueda del pretil se limita por arriba con la maquinaria, para que no se vaya al horizonte |

Otros documentos:

- [`docs/analisis_material.md`](docs/analisis_material.md): mediciones del material de
  entrada y prueba de los detectores sin entrenar.
- [`docs/resultados.md`](docs/resultados.md): resultados por objetivo (detección y
  proximidad, altura del pretil), con sus conclusiones y lo que queda por hacer.
- [`docs/guia_anotacion.md`](docs/guia_anotacion.md): criterios que usé para anotar.
- [`data/README.md`](data/README.md): qué datos se versionan y cómo reproducir el
  entrenamiento.

### Una ambigüedad del enunciado

El enunciado pide usar GPU, pero el comando de referencia no incluye `--gpus`, así que
ejecutado tal cual el contenedor no ve la GPU. No sé cuál de los dos escenarios usará la
evaluación automática, y equivocarse no cuesta lo mismo en ambos casos: `--gpus all` hace
fallar el arranque en una máquina sin el runtime de NVIDIA, mientras que una imagen con base
CUDA corre sin problema en una sin GPU. Por eso el sistema funciona en los dos modos: CUDA si
está disponible y CPU si no.

---

## Rendimiento

Medido con el comando de referencia dentro del contenedor, con la imagen construida desde
este código. Equipo: portátil con RTX 3050 Ti (4 GB).

| Resolución | CPU (sin `--gpus`) | GPU (`--gpus all`) | Ganancia |
|---|---|---|---|
| 1280×720 | 8.1 fps | 15.0 fps | 1.8× |
| 1920×1080 | 5.2 fps | 7.3 fps | 1.4× |

La GPU solo acelera el detector. La segmentación del pretil corre en NumPy y OpenCV y tarda
lo mismo en ambos modos (36 ms por frame a 720p, 74 ms a 1080p), así que limita la ganancia
total. De ahí salen dos cosas: el modo CPU es usable (8 fps a 720p), y optimizar el detector
serviría poco; lo que rendiría es llevar el pretil a GPU o bajarle la resolución. El
desglose por etapa está en la sección 5 del reporte.

La imagen pesa 15.7 GB, casi todo por la base CUDA y las librerías de NVIDIA que trae
PyTorch.

---

## Estado y limitaciones

### Funciona

- Procesa una carpeta cualquiera de videos, con distintas resoluciones y FPS. Si un video no
  se puede leer, queda registrado en `run_metadata.json` y el resto sigue.
- Detecta los CAEX con recall 1.000 y mAP@0.5 de 0.812 en validación.
- Separa en cajas distintas las máquinas que están juntas, que es lo que se necesita para
  medir proximidad. El modelo COCO las juntaba en una sola caja.
- Detecta los cortes de escena y clasifica la luz de cada frame.
- Traza un perfil continuo del pretil y mide su altura con una incertidumbre explícita.

### Lo que no funciona o tiene límites

- **La clase `bulldozer` es débil**: mAP@0.5 de 0.306 y recall 0.455. Con el umbral de 0.25
  casi no aparece, así que en la práctica el OSD marca todo como `caex`. Las etiquetas de
  clase no deben leerse como el tipo de equipo.
- **El detector sobre-detecta**: precisión 0.297 en `caex`. Encuentra todos los camiones,
  pero ~70 % de sus cajas son falsas, y eso genera alertas de proximidad espurias que el
  tracker solo mitiga en parte. Las dos cosas vienen de lo mismo: 33 bulldozers contra 270
  CAEX en entrenamiento (ver [`docs/resultados.md`](docs/resultados.md)).
- **El pretil se evalúa por estabilidad, no por exactitud.** No tengo el perfil real
  anotado, así que puedo comparar el jitter y el costo entre métodos pero no el error de la
  cresta.
- **La altura absoluta está sesgada** por un factor cercano a 4: la mediana es 0.38 m y la
  normativa habla de 1.5–2 m. Además varía ±12 % solo por la luz. No sirve para verificar
  cumplimiento. Sí sirve para detectar que el pretil baja respecto de sí mismo, porque un
  error de escala constante se cancela.
- **El pretil no se traza bien en toda la escena.** Tiene poco contraste, y las huellas de
  neumático del primer plano dan más gradiente que la cresta. De noche la curva se iba al
  horizonte; el ADR 0007 lo corrigió, pero donde no hay un borde claro la curva baja al
  límite inferior de la banda de búsqueda.
- **Sin calibración de cámara, la escala depende de la maquinaria.** Uso el ancho nominal
  del CAEX como referencia, así que los errores de la caja pasan a la medición.
- **Los umbrales se ajustaron con cuatro videos.** Los de luz, cortes y campo de visión
  están en `configs/`. No sé cómo se comportan con videos nuevos.
- **Anoté yo solo.** No hay acuerdo entre anotadores, así que las métricas de detección
  tienen una incertidumbre que no medí.

---

## Desarrollo local

Python 3.10, la misma versión que usa la imagen (Ubuntu 22.04).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# Torch primero y desde el índice de CUDA; si se instala después de ultralytics,
# pip trae la versión de CPU.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e . --no-deps

python main.py --input data/raw --output output --method 1
```

Calidad:

```bash
ruff check . && ruff format --check .
mypy bermguard/
pytest
```

`scripts/` tiene herramientas que no son parte del pipeline, pero con las que generé la
evidencia de `docs/`:

| Script | Para qué |
|---|---|
| `probe_videos.py` | Mide resolución, cortes y luminancia del material |
| `probe_detector.py` | Cuenta qué clases dispara el detector COCO |
| `probe_openvocab.py` | Prueba el detector zero-shot con prompts de texto |
| `prepare_labeling_set.py` | Extrae frames para anotar, estratificados por luz |
| `split_dataset.py` | Separa entrenamiento y validación por video |
| `train_detector.py` | Entrena el detector |
| `eval_detector.py` | Calcula las métricas por clase |

---

## Trabajo futuro

La lista completa, ordenada por objetivo y con el motivo de cada punto, está en la sección 4
de [`docs/resultados.md`](docs/resultados.md). Lo más importante:

1. **Anotar más bulldozers.** Es la limitación principal y la más barata de resolver.
2. **Anotar el perfil real del pretil** en unos 30 frames, para medir exactitud y no solo
   estabilidad.
3. **Calibrar la cámara**, para que la altura absoluta sea confiable.
