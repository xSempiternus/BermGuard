"""Reporta qué clases de COCO dispara un detector de estantería sobre el material.

Herramienta de desarrollo, no forma parte del pipeline. Responde una pregunta con
evidencia en vez de con suposiciones: un modelo preentrenado no sabe nada de
maquinaria minera, así que antes de decidir cómo mapear sus etiquetas a CAEX y
bulldozer hay que ver qué reporta realmente.

El histograma de clases que produce es insumo para el reporte de benchmark.

Uso:
    python scripts/probe_detector.py data/raw --weights yolo11s.pt --stride 12
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from bermguard.core.device import resolve_device
from bermguard.io.video_reader import VideoReader, discover_videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directorio con los videos")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--stride", type=int, default=12, help="Muestrear cada N frames")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from ultralytics import YOLO

    device = resolve_device(args.device)  # type: ignore[arg-type]
    model = YOLO(args.weights)
    model.to(device)
    names: dict[int, str] = model.names

    for video in discover_videos(args.input):
        counts: defaultdict[int, int] = defaultdict(int)
        conf_sum: defaultdict[int, float] = defaultdict(float)
        sampled = 0

        with VideoReader(video) as reader:
            for index, _timestamp, frame in reader.frames():
                if index % args.stride:
                    continue
                sampled += 1
                result = model.predict(
                    source=frame, conf=args.conf, device=device, verbose=False
                )[0]
                if result.boxes is None:
                    continue
                for cls, conf in zip(
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                ):
                    counts[int(cls)] += 1
                    conf_sum[int(cls)] += float(conf)

        print(f"\n=== {video.name}  ({sampled} frames muestreados) ===")
        if not counts:
            print("  sin detecciones")
            continue
        for cls in sorted(counts, key=lambda c: -counts[c]):
            mean_conf = conf_sum[cls] / counts[cls]
            per_frame = counts[cls] / sampled
            print(
                f"  {names.get(cls, cls):<16} id={cls:<3} "
                f"aciertos={counts[cls]:<5} conf_media={mean_conf:.2f} "
                f"por_frame={per_frame:.2f}"
            )


if __name__ == "__main__":
    main()
