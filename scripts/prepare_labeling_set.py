"""Extrae y pre-anota un conjunto de frames para etiquetado manual.

Herramienta de desarrollo. Prepara el material que se sube a la herramienta de
anotación para el fine-tuning descrito en el ADR 0002.

Dos decisiones de muestreo, ambas con motivo:

**Estratificación por condición lumínica.** Un muestreo uniforme sobre el eje del
tiempo entregaría un conjunto dominado por lo que más dura, y en este material eso
es la noche. Peor aún, el crepúsculo —que es la transición donde el detector más
sufre— representa cerca del 10 % de los frames y quedaría casi ausente. El
muestreo asigna una cuota por condición, no por duración.

**Separación mínima entre frames.** Frames consecutivos son casi idénticos:
etiquetar dos de ellos duplica el esfuerzo sin aportar información nueva, y además
sesga el conjunto de validación si ambos caen a lados distintos de la partición.

La pre-anotación usa el modelo COCO para adelantar trabajo, no para reemplazarlo.
Como documenta ``docs/analisis_material.md``, ese modelo **fusiona el CAEX con el
bulldozer en una sola caja**, así que las cajas propuestas hay que revisarlas
siempre y dividirlas cuando corresponda. Se emiten todas como clase ``caex``
porque es la única que el modelo base acierta; el bulldozer se agrega a mano.

Uso:
    python scripts/prepare_labeling_set.py data/raw --count 150
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import cv2

from bermguard.core.device import resolve_device
from bermguard.core.types import LightingCondition
from bermguard.io.video_reader import VideoReader, discover_videos
from bermguard.vision.lighting import classify_luma, mean_luma

logger = logging.getLogger(__name__)

CLASS_NAMES = ["caex", "bulldozer"]
"""Clases del dataset de fine-tuning. El orden define el índice en las etiquetas."""

CUOTA_POR_CONDICION = {
    LightingCondition.DAY: 0.40,
    LightingCondition.NIGHT: 0.40,
    LightingCondition.DUSK: 0.20,
}
"""Reparto del presupuesto de etiquetado.

El crepúsculo recibe una cuota muy por encima de su frecuencia real (~10 %) porque
es donde el detector se degrada: el conjunto de entrenamiento debe cubrir el caso
difícil, no reproducir la distribución del material.
"""

SEPARACION_MINIMA = 6
"""Frames mínimos entre dos muestras del mismo video."""


def elegir_frames(
    candidatos: dict[LightingCondition, list[tuple[Path, int]]],
    total: int,
    rng: random.Random,
) -> list[tuple[Path, int]]:
    """Reparte ``total`` muestras entre condiciones según :data:`CUOTA_POR_CONDICION`.

    Si una condición no tiene suficientes candidatos, su remanente se redistribuye
    entre las demás en vez de perderse.
    """
    seleccion: list[tuple[Path, int]] = []
    pendiente = 0

    for condicion, cuota in CUOTA_POR_CONDICION.items():
        disponibles = candidatos.get(condicion, [])
        objetivo = round(total * cuota)
        toman = min(objetivo, len(disponibles))
        pendiente += objetivo - toman
        seleccion.extend(rng.sample(disponibles, toman))
        logger.info(
            "%s: %d candidatos, %d seleccionados (objetivo %d)",
            condicion.value,
            len(disponibles),
            toman,
            objetivo,
        )

    if pendiente:
        restantes = [
            item for lista in candidatos.values() for item in lista if item not in set(seleccion)
        ]
        extra = rng.sample(restantes, min(pendiente, len(restantes)))
        seleccion.extend(extra)
        logger.info("Redistribuidos %d cupos no cubiertos", len(extra))

    return seleccion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directorio con los videos")
    parser.add_argument("--out", type=Path, default=Path("data/labeling"))
    parser.add_argument("--count", type=int, default=150, help="Frames a extraer")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sin-preanotar",
        action="store_true",
        help="Extrae los frames sin proponer cajas",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = random.Random(args.seed)

    dir_imagenes = args.out / "images"
    dir_etiquetas = args.out / "labels"
    dir_imagenes.mkdir(parents=True, exist_ok=True)
    dir_etiquetas.mkdir(parents=True, exist_ok=True)

    # --- Paso 1: catalogar candidatos por condición lumínica ---
    candidatos: dict[LightingCondition, list[tuple[Path, int]]] = defaultdict(list)
    for video in discover_videos(args.input):
        ultimo_tomado = -SEPARACION_MINIMA
        with VideoReader(video) as reader:
            for index, _timestamp, frame in reader.frames():
                if index - ultimo_tomado < SEPARACION_MINIMA:
                    continue
                candidatos[classify_luma(mean_luma(frame))].append((video, index))
                ultimo_tomado = index

    seleccion = sorted(elegir_frames(candidatos, args.count, rng))
    logger.info("Total seleccionado: %d frames", len(seleccion))

    # --- Paso 2: extraer y, opcionalmente, pre-anotar ---
    modelo = None
    if not args.sin_preanotar:
        from ultralytics import YOLO

        device = resolve_device(args.device)  # type: ignore[arg-type]
        modelo = YOLO(args.weights)
        modelo.to(device)

    por_video: dict[Path, set[int]] = defaultdict(set)
    for video, index in seleccion:
        por_video[video].add(index)

    escritos = 0
    cajas_propuestas = 0
    for video, indices in por_video.items():
        with VideoReader(video) as reader:
            alto, ancho = reader.info.height, reader.info.width
            for index, _timestamp, frame in reader.frames():
                if index not in indices:
                    continue

                nombre = f"{video.stem}_f{index:05d}"
                cv2.imwrite(str(dir_imagenes / f"{nombre}.jpg"), frame)
                escritos += 1

                lineas: list[str] = []
                if modelo is not None:
                    resultado = modelo.predict(
                        source=frame, conf=args.conf, classes=[7], verbose=False
                    )[0]
                    for caja in resultado.boxes or []:
                        x1, y1, x2, y2 = caja.xyxy[0].cpu().numpy()
                        # Formato YOLO: clase cx cy w h, todo normalizado a [0,1].
                        cx = (x1 + x2) / 2 / ancho
                        cy = (y1 + y2) / 2 / alto
                        w = (x2 - x1) / ancho
                        h = (y2 - y1) / alto
                        lineas.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        cajas_propuestas += 1

                # Con salto de linea final: es la convencion de los archivos de
                # texto POSIX y evita que al concatenar varios se fusionen la
                # ultima linea de uno con la primera del siguiente.
                (dir_etiquetas / f"{nombre}.txt").write_text(
                    "".join(f"{linea}\n" for linea in lineas), encoding="utf-8"
                )

    # --- Paso 3: descriptores del dataset ---
    # classes.txt se escribe DENTRO de labels/ y no sólo en la raíz. Es la
    # convención que esperan las herramientas de anotación: sin ese archivo junto
    # a los .txt, el índice numérico de cada caja no se puede traducir a un nombre
    # y la herramienta acaba creando una clase llamada literalmente "0".
    nombres = "\n".join(CLASS_NAMES) + "\n"
    (dir_etiquetas / "classes.txt").write_text(nombres, encoding="utf-8")
    (args.out / "classes.txt").write_text(nombres, encoding="utf-8")
    (args.out / "data.yaml").write_text(
        "# Generado por scripts/prepare_labeling_set.py\n"
        "# Las rutas train/val se completan tras dividir el conjunto anotado.\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n",
        encoding="utf-8",
    )

    print(f"\n{escritos} imagenes en {dir_imagenes}")
    print(f"{cajas_propuestas} cajas propuestas en {dir_etiquetas}")
    print(
        "\nRevisar TODAS las cajas propuestas: el modelo base fusiona el CAEX con el\n"
        "bulldozer en una sola caja. Dividir las fusionadas y etiquetar el bulldozer\n"
        f"como clase 1. Clases: {dict(enumerate(CLASS_NAMES))}"
    )


if __name__ == "__main__":
    main()
