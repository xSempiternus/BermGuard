# BermGuard AI

Pipeline de visión por computadora para faenas mineras a rajo abierto. Sobre una carpeta de
videos de botadero, produce por cada uno: un video anotado, la segmentación del pretil de
seguridad, la evaluación de proximidad entre maquinaria y los metadatos de ejecución.

**Por dónde empezar.** El resultado se ve en
`output_ejemplo/video_04/method_1/video_04_osd.mp4` — la única salida producida sobre
material que el detector nunca vio. Este README explica cómo se ejecuta y qué decisiones
lo sostienen; [`reporte_benchmark.md`](reporte_benchmark.md) contiene la comparación de
los métodos y las mediciones.

---

## Ejecución

El contrato de ejecución es el comando de referencia del enunciado, y funciona tal cual:

```bash
docker build -t bermguard:latest .

docker run --rm \
  -v /ruta/local/test:/app/test \
  -v /ruta/local/output:/app/output \
  bermguard:latest \
  python main.py --input /app/test --output /app/output --method 1
```

Con GPU disponible, añadir `--gpus all` antes del nombre de la imagen. **No es necesario:**
el pipeline detecta la ausencia de CUDA, registra una advertencia explícita y continúa en CPU.
Ver [ADR 0001](docs/adr/0001-cuda-como-stack-de-aceleracion.md) para el razonamiento.

### Argumentos

| Argumento | Valores | Descripción |
|---|---|---|
| `--input` | ruta | Directorio con los videos. Los archivos que no son video se ignoran |
| `--output` | ruta | Directorio de artefactos. Se crea si no existe |
| `--method` | `1`, `2`, `all` | `1` camino óptimo, `2` línea base por argmax, `all` ambos |
| `--device` | `auto`, `cuda`, `cpu` | Por defecto `auto` |
| `--max-frames` | entero | Tope de frames por video, para pruebas rápidas |
| `--log-level` | `DEBUG`…`ERROR` | Por defecto `INFO` |

Todos los pesos van horneados en la imagen. No se descarga nada en tiempo de ejecución.

### Artefactos de salida

```
output/
├── run_metadata.json                  # argumentos, dispositivo, versiones, commit, fallos
└── <video>/method_<n>/
    ├── <video>_osd.mp4                # video con la capa de anotación
    ├── metadata.json                  # FPS, ms/frame por etapa, alertas, resoluciones
    ├── berm_profile.csv               # serie temporal del pretil, datos crudos
    └── proximity_events.csv           # posiciones y nivel de riesgo por track
```

Cada gráfico se acompaña de su CSV a propósito: permite reproducir la figura desde los datos
en lugar de confiar en la imagen.

---

## Arquitectura

```
main.py                    CLI. Interpreta argumentos, construye y delega
bermguard/
├── core/                  Tipos del dominio, Protocols, excepciones, configuración
├── io/                    Lectura y escritura de video, metadatos y series
├── vision/                Detección, segmentación de terreno, condición lumínica, cortes
├── analytics/             Geometría, altura y proximidad
├── pipeline/              Orquestación, factory y renderizado del OSD
└── benchmark/             Métricas comparativas entre métodos
```

Tres decisiones estructurales sostienen el resto:

**El orquestador depende de abstracciones.** Conoce únicamente los `Protocol` de
`core/interfaces.py`, nunca una clase concreta. Incorporar un método nuevo consiste en
escribir sus implementaciones y registrarlas en `pipeline/factory.py`; ninguna línea del
orquestador cambia. Esto se verificó en la práctica al sustituir el detector preentrenado por
el especializado: cambió una ruta en `configs/`.

**Los tipos del dominio son inmutables y no saben de librerías.** `analytics/` opera sobre
dataclasses y arrays, de modo que se testea sin GPU, sin modelo y sin archivo de video.

**Segmentar y medir son etapas separadas.** Un `IBermSegmenter` dice dónde está el pretil en
píxeles; un `IHeightEstimator` lo convierte a metros. Fallan por separado —puede haber pretil
perfectamente delineado sin ningún vehículo que sirva de ancla métrica— y el sistema lo
refleja en vez de descartar una segmentación válida por falta de escala.

---

## Decisiones de diseño

Cada decisión no trivial está registrada como ADR, con su contexto, las alternativas
consideradas y sus consecuencias.

| ADR | Decisión |
|---|---|
| [0001](docs/adr/0001-cuda-como-stack-de-aceleracion.md) | CUDA como stack, con degradación a CPU por selección de dispositivo |
| [0002](docs/adr/0002-deteccion-de-maquinaria.md) | Fine-tuning del detector, tras descartar zero-shot con evidencia medida |
| [0003](docs/adr/0003-segmentacion-por-toma-y-condicion-luminica.md) | La toma es la unidad de análisis; la condición lumínica se clasifica por frame |
| [0004](docs/adr/0004-datos-de-entrenamiento-del-dominio.md) | Datos propios anotados por sobre tres datasets públicos evaluados |
| [0005](docs/adr/0005-el-detector-opera-sobre-frames-crudos.md) | El detector no ve acondicionamiento de imagen; la rama de terreno sí |
| [0006](docs/adr/0006-segmentacion-del-pretil-por-camino-optimo.md) | La cresta del pretil como camino óptimo, no como `argmax` por columna |

Documentos de respaldo:

- [`docs/analisis_material.md`](docs/analisis_material.md) — caracterización medida del
  material de entrada y de los dos enfoques de detección sin entrenamiento
- [`docs/resultados_deteccion.md`](docs/resultados_deteccion.md) — resultados del detector
  especializado, incluido lo que no funcionó
- [`docs/guia_anotacion.md`](docs/guia_anotacion.md) — protocolo de anotación
- [`reporte_benchmark.md`](reporte_benchmark.md) — comparación de los métodos, con las
  métricas que resultaron no medir lo que se esperaba

### Ambigüedad del enunciado, resuelta y declarada

Los requisitos exigen *«usar GPU (CUDA o ROCm o OPENVINO…)»*, mientras que el único comando de
referencia publicado no pasa `--gpus`, de modo que ejecutado literalmente levanta un contenedor
sin acceso al dispositivo. Como la evaluación se describe como automatizada, no hay forma de
saber de antemano cuál de los dos escenarios ocurrirá.

El costo de equivocarse es asimétrico: `docker run --gpus all` falla al arrancar si el host no
tiene el runtime de NVIDIA, mientras que una imagen construida sobre base CUDA se ejecuta sin
problemas en una máquina sin GPU. Se asumió por tanto que el sistema debe operar en ambos
modos, con CUDA como camino declarado y la degradación a CPU como seguro.

---

## Rendimiento medido

Ejecutando el comando de referencia dentro del contenedor, sobre el material de muestra.
Hardware: RTX 3050 Ti Laptop (4 GB) e Intel de portátil.

| Resolución | Sin `--gpus` (CPU) | Con `--gpus all` | Ganancia |
|---|---|---|---|
| 1280×720 | 8.2 fps | 15.4 fps | 1.9× |
| 1920×1080 | — | 1.5 fps | — |

**La GPU sólo acelera el detector.** La segmentación del pretil se ejecuta en NumPy y OpenCV
sobre CPU, y a 720p ya cuesta más que la inferencia. El resultado es que la aceleración
extremo a extremo se queda en 1.9×, muy por debajo del orden de magnitud que suele asumirse:
la etapa que no se acelera acota la ganancia total.

Tiene dos consecuencias prácticas. La primera es que **el modo CPU es perfectamente utilizable**
—8 fps sobre clips de diez segundos—, lo que respalda la decisión del ADR 0001 de degradar en
vez de exigir GPU. La segunda es que optimizar el detector sin tocar la rama de terreno daría
un retorno marginal; el trabajo rendidor sería llevar la búsqueda de camino óptimo a GPU o
reducir su resolución de trabajo.

La imagen pesa 15.7 GB, dominada por la base CUDA y las librerías de NVIDIA que arrastra
PyTorch. Es el costo de un despliegue con GPU disponible sin descargas en tiempo de ejecución.

---

## Estado y limitaciones

Esta sección declara qué funciona, qué no, y por qué. Es deliberadamente explícita.

### Funciona

- **Ejecución extremo a extremo** sobre un directorio arbitrario, con resolución y tasa de
  frames heterogéneas. Un video ilegible se registra en `run_metadata.json` y no interrumpe
  el lote.
- **Detección de CAEX** con recall de 1.000 y mAP@0.5 de 0.812 sobre el conjunto de
  validación.
- **Separación de máquinas contiguas** en cajas independientes, que es la condición necesaria
  para medir proximidad. El modelo preentrenado las fusionaba en una sola caja.
- **Segmentación del pretil** con cobertura en torno al 60 % de las columnas, con continuidad
  garantizada por construcción.
- **Segmentación por tomas** y clasificación lumínica por frame.

### No funciona, y por qué

**La clase `bulldozer` está aprendida pero es frágil.** mAP@0.5 de 0.306 y recall de 0.455,
frente a 0.812 y 1.000 para `caex`. Al umbral de operación de 0.25 la clase minoritaria casi no
se emite, de modo que en la práctica el OSD etiqueta toda máquina como `caex`. **Las etiquetas
de clase de los artefactos no deben interpretarse como identificación de tipo de equipo.**

**El detector sobre-detecta.** La precisión sobre `caex` es de 0.297 con recall 1.000: encuentra
todos los camiones, pero cerca del 70 % de sus detecciones no corresponde a ninguno. Cada falso
positivo será una entidad fantasma para el módulo de proximidad, y por tanto una alerta espuria
potencial.

Ambos síntomas tienen la misma causa, y es de datos: 33 instancias de bulldozer en
entrenamiento frente a 270 de CAEX. El análisis completo está en
[`docs/resultados_deteccion.md`](docs/resultados_deteccion.md).

**La segmentación del pretil se mide por estabilidad, no por exactitud.** No hay ground
truth anotado del perfil, de modo que el jitter temporal y el costo son comparables entre
métodos pero el error absoluto de la cresta no se conoce. Es la limitación central del
`reporte_benchmark.md`.

**La altura absoluta tiene un sesgo sistemático de factor ~3** — mediana medida 0.49 m
frente a los 1.5–2 m que referencia la normativa — y una dispersión de ±26 % atribuible
sólo al cambio de iluminación. **No debe usarse para verificar cumplimiento normativo.**
En términos relativos, detectar que el pretil se degrada respecto a su propia línea base
sí es confiable, porque un factor de escala constante se cancela en la comparación.

**El pretil no se delinea completo.** La cobertura media ronda el 60 % de las columnas. El
material presenta contraste bajo entre el banco de tierra y el suelo circundante, y textura de
primer plano —huellas de neumático, sombras largas— cuya respuesta de gradiente supera a la de
la propia cresta. Las alternativas probadas y descartadas están en el ADR 0006.

**Sin calibración de cámara no hay medición métrica confiable.** Una imagen no contiene escala:
un pretil de dos metros cerca y uno de seis lejos ocupan los mismos píxeles. El ancla
disponible son las dimensiones nominales de la maquinaria detectada, lo que traslada al
resultado el error de la caja y la dispersión del supuesto.

**El rendimiento no es de tiempo real en 1080p.** La segmentación del pretil cuesta unos
120 ms por frame a esa resolución, comparable o superior a la inferencia del detector. Es el
precio de una búsqueda global sobre toda la rejilla.

**Los umbrales están calibrados sobre el material de muestra.** Los cortes de clasificación
lumínica y el umbral de detección de cortes se ajustaron sobre cuatro clips. El conjunto de
evaluación es ciego: su generalización es un supuesto declarado, no un hecho verificado. Todos
los umbrales son configurables en `configs/`.

**Anotador único.** El conjunto de entrenamiento y validación lo anotó una sola persona, sin
acuerdo entre anotadores. Las métricas arrastran una incertidumbre propia no cuantificada.

---

## Desarrollo local

Requiere Python 3.10, la misma versión que trae Ubuntu 22.04 dentro de la imagen.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# Torch primero y desde el indice de CUDA: instalarlo despues de ultralytics
# arrastra la compilacion de CPU desde PyPI.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e . --no-deps

python main.py --input data/raw --output output --method 1
```

### Calidad

```bash
ruff check . && ruff format --check .
mypy bermguard/
pytest
```

### Herramientas de desarrollo

`scripts/` contiene utilidades que no forman parte del pipeline y que documentan cómo se
construyó. Producen la evidencia citada en `docs/`.

| Script | Qué produce |
|---|---|
| `probe_videos.py` | Caracterización del material: resolución, cortes, luminancia |
| `probe_detector.py` | Histograma de clases que dispara un detector preentrenado |
| `probe_openvocab.py` | Evaluación del enfoque zero-shot con prompts de texto |
| `prepare_labeling_set.py` | Extracción estratificada de frames para anotar |
| `split_dataset.py` | Partición por video, no aleatoria |
| `train_detector.py` | Fine-tuning del detector |

---

## Trabajo futuro

En orden de impacto esperado sobre las limitaciones declaradas:

1. **Etiquetado dirigido de bulldozer.** Es la limitación dominante y la más barata de
   atacar: extraer frames restringidos a los tramos donde la máquina es visible y anotar unas
   sesenta más duplicaría la clase minoritaria en cerca de una hora.
2. **Calibración de cámara.** Un tablero de ajedrez, o los metadatos de montaje —focal, altura,
   ángulo—, llevarían el error de la medición de altura de un orden de ±20 % a menos del 5 %.
3. **Reportar la altura de forma adimensional**, como cociente entre la altura del pretil y el
   radio de rueda del equipo mayor. La normativa define el criterio de cumplimiento
   exactamente en esos términos, y el cociente cancela el sesgo global de escala, que es la
   fuente dominante de error.
4. **Segmentación neural del pretil** como Método 2, para contrastar contra el prior
   geométrico explícito.
5. **Exportación a ONNX Runtime** y medición del speedup frente a PyTorch en FP32 y FP16.
6. **Monitoreo de deriva en producción:** si la distribución de confianzas del detector cae,
   alertar antes de que el sistema falle en silencio.
