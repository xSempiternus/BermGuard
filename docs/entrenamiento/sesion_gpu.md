# Sesión de GPU: validación cruzada LOVO

Guía para correr la validación cruzada dejando un video fuera (ADR 0009, revisiones del
2026-10-20 y del 2026-10-21) con las tres variantes: `base`, `copypaste` y `pretrain`, más
el control `base2x` si hace falta. La GPU de desarrollo
(RTX 3050 Ti, 4 GB) se comparte con el escritorio, así que esto se corre en una ventana en
que la máquina está libre. Todo se lanza con un solo comando, `scripts/gpu_session.py`.

## Antes

1. **Cerrar Blender, los juegos, OBS** y cualquier otro programa que use la GPU. Con el
   escritorio cargado, Windows pagina la VRAM a RAM y cada época pasa de segundos a
   minutos.
2. **Tener los datos listos** (ver [`data/README.md`](../../data/README.md)):
   - `data/dataset/` con frames y etiquetas, y la auditoría aplicada
     (`python scripts/apply_label_audit.py --check` debe decir "Por aplicar: 0").
   - `data/pretrain/` generado con `--exclude` y `--repeat` (comando en el ADR 0004,
     revisión del 2026-10-20).
   - `weights/yolo11s.pt`, los pesos COCO locales.
3. **Chequear**, sin entrenar nada:

   ```bash
   python scripts/gpu_session.py --check
   ```

   Revisa el dataset, el set de preentrenamiento, los pesos, la VRAM libre (al menos
   2.800 MB), los procesos pesados en la GPU y el disco libre (al menos 5 GB). Sale con
   código 1 y una línea `!` por cada problema.

Para probar la cadena sin GPU (una época, un fold, en carpetas desechables, para no dejar
folds "terminados" falsos en `weights/`):

```bash
CUDA_VISIBLE_DEVICES=-1 python scripts/cross_validate.py --device cpu \
    --variants base copypaste pretrain --folds video_04 --epochs 1 --pretrain-epochs 1 \
    --weights-root <tmp>/weights --work <tmp>/cv --out <tmp>/lovo
python scripts/summarize_lovo.py --dir <tmp>/lovo
```

## Durante

```bash
python scripts/gpu_session.py
```

Corre los chequeos, `cross_validate.py --variants base copypaste pretrain --batch 4` y
`summarize_lovo.py`. Si el resumen queda `pendiente_control` (`copypaste` sería candidata
contra `base`), entrena además el control `base2x` y vuelve a resumir. `copypaste` hace el
doble de pasos que `base`, y `base2x` le iguala los pasos para saber si su mejora es de
augmentation o de cómputo (ADR 0009, revisión del 2026-10-21). Si un chequeo falla, se detiene; `--force` sigue igual (por ejemplo,
si el aviso de VRAM es por un programa que se va a cerrar).

Orden y duración estimada en la GPU de desarrollo libre. Son estimaciones extrapoladas de
una prueba de humo en CPU, no mediciones: **unas 10 h en total, ± 25 %**, y unas 3,4 h más
si hace falta el control.

| Etapa | Qué entrena | Estimado |
|---|---|---|
| 1. Preentrenamiento | 1 modelo sobre `data/pretrain` (50 épocas, con paciencia) | hasta ~3,3 h |
| 2. `base` | 4 folds × 60 épocas | ~1,7 h |
| 3. `copypaste` | 4 folds × 60 épocas, el doble de frames de train (y de pasos) | ~3,5 h |
| 4. `pretrain` | 4 folds × 60 épocas desde el preentrenado | ~1,7 h |
| 5. Resumen | `summarize_lovo.py` | segundos |
| 6. `base2x`, sólo si queda `pendiente_control` | 4 folds × 120 épocas, y otro resumen | ~3,4 h |

El preentrenamiento va primero porque `cross_validate.py` lo necesita antes de empezar los
folds de `pretrain`, y se hace una sola vez para los cuatro. Puede terminar antes de 3,3 h
si la paciencia lo corta.

**Cortar y retomar.** Ctrl+C en cualquier momento libera la GPU. Para retomar, volver a
lanzar `python scripts/gpu_session.py`: cada entrenamiento terminado dejó su
`resumen_entrenamiento.json` (en `weights/pretrain/externo/` y `weights/cv/<variante>/<video>/`),
y esos se saltan y sólo se reevalúan. El que quedó a medias se reentrena desde cero, así
que conviene cortar justo después de que termina un fold. Los JSON por fold van quedando
en `docs/experimentos/lovo/<variante>_<video>.json`.

Para repartir la corrida en varias ventanas se puede limitar a algunas variantes:
`python scripts/gpu_session.py --variants base copypaste`. El resumen necesita `base`, y
una variante sólo puede ganar si tiene los cuatro folds. `gpu_session.py --variants` no
acepta `base2x`; para entrenar el control a mano:
`python scripts/cross_validate.py --variants base2x --batch 4` y después
`python scripts/summarize_lovo.py`. Si se corta durante `base2x`, volver a lanzar
`gpu_session.py` salta los folds terminados, ve el resumen todavía pendiente y retoma el
control.

## Después

1. **Leer `docs/experimentos/lovo/resumen.md`.** Tiene la media y la dispersión por
   variante, las diferencias pareadas contra `base` con su veredicto, la variante elegida
   por la regla y el umbral de bulldozer fuera de fold. La regla y los riesgos están en el
   ADR 0009 (revisiones del 2026-10-20 y del 2026-10-21); no cambiarla después de ver los
   números. Si el motivo de la decisión dice "Provisoria", faltan folds de `base2x` y la
   ganadora no vale todavía: entrenar el control y volver a resumir. Si dice "entrenar
   v2 con el doble de épocas" (`epocas_dobles` en `resumen.json`), `base2x` mejoró a
   `base`: 60 épocas eran pocas, y el v2 se entrena con 120.
2. **Revisar avisos del resumen**: un umbral marcado "borde de la grilla" (el óptimo puede
   estar bajo 0,01) o "no cumple el mínimo" (ningún umbral llega a precisión 0,5).

### Pasos siguientes

1. **Reentrenar `detector_v2`** con la variante ganadora sobre los cuatro videos, con el
   mismo protocolo que los folds: 60 épocas, `--patience 0 --no-val`, `last.pt`, y desde
   el preentrenado si ganó `pretrain`.

   **Pendiente de herramienta.** Hoy `train_detector.py --data data/dataset/data.yaml`
   sufre el mismo descarte que tenía LOVO: Ultralytics salta 36 de los 51 frames de
   `video_01` porque sus etiquetas mezclan cajas y polígonos. Hace falta un script que
   copie `data/dataset` a una carpeta de entrenamiento pasando cada etiqueta por
   `bermguard.training.etiquetas.homogeneizar` (como hace `preparar_fold` en
   `cross_validate.py`), con los cuatro videos en train y, si ganó `copypaste`, los frames
   aumentados. Todavía no existe.

   Con esa carpeta, el entrenamiento sería:

   ```bash
   python scripts/train_detector.py --data <carpeta>/data.yaml --model weights/yolo11s.pt \
       --epochs 60 --patience 0 --no-val --batch 4 --name detector_v2
   ```

   Copiar su `last.pt` a `weights/detector_v2.pt` y agregar `!weights/detector_v2.pt` al
   `.gitignore`, que sólo deja pasar los pesos que la imagen necesita.

   `detector_v2` entrena con los cuatro videos, así que no queda un video sin ver para
   evaluarlo: su rendimiento esperado es el de LOVO, no el de una evaluación sobre
   `video_04`.

2. **Fijar el umbral en `configs/`**: `detector.weights: weights/detector_v2.pt` y
   `detector.class_confidence: {bulldozer: <umbral de producción del resumen>}` en
   `configs/method_1.yaml` y `configs/method_2.yaml`.
3. **Regenerar las salidas de ejemplo** (`output_ejemplo/`, ver su README) con
   `python main.py --input data/raw --output output --method all`, y las figuras con
   `python scripts/make_showcase.py --raw data/raw`, que incluye la de LOVO.
4. **Actualizar la documentación con los números medidos**:
   - `docs/resultados.md`: pegar el bloque de `resumen.md` en la sección 1 y actualizar
     las conclusiones y la sección 4.
   - ADR 0009: el resultado y la decisión tomada.
   - `portfolio.json` con `python scripts/build_portfolio.py`, y después la landing según
     su `docs/SYNC.md`. Qué JSON pasar a `--detector` para `detector_v2` queda por
     decidir, porque no hay un video que no haya visto.
   - El README (cifras de la portada, "Lo que no funciona", trabajo futuro).
