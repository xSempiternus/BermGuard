"""Evaluación rigurosa del detector: punto de operación, intervalos y desglose por luz.

Complementa a ``eval_detector.py`` (que reporta lo que calcula Ultralytics) con lo
que hace falta para afirmar algo sobre el sistema en operación. Ver
``bermguard/benchmark/detection.py`` para las definiciones.

Las predicciones se calculan una vez con umbral casi cero y se guardan, de modo que
el barrido de umbrales y el bootstrap no vuelven a correr el modelo.

Uso:
    python scripts/evaluate_detector.py --weights weights/detector_v1.pt \\
        --images data/dataset/images/val --labels data/dataset/labels/val \\
        --out docs/experimentos/detector_v1_val.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from bermguard.benchmark.detection import (
    Box,
    ImageEval,
    bootstrap_ci,
    class_metrics,
    read_yolo_labels,
    threshold_sweep,
)
from bermguard.vision.lighting import LightingThresholds, classify_luma

NOMBRES = {0: "caex", 1: "bulldozer"}
PATRON = re.compile(r"(video_\d+)_f(\d+)")


def etiqueta_de(imagen: Path, etiquetas: Path) -> Path:
    """La etiqueta comparte el prefijo ``video_NN_fNNNNN`` con la imagen."""
    exacta = etiquetas / f"{imagen.stem}.txt"
    if exacta.exists():
        return exacta
    prefijo = PATRON.search(imagen.stem)
    if prefijo:
        candidatos = sorted(etiquetas.glob(f"{prefijo.group(0)}*.txt"))
        if candidatos:
            return candidatos[0]
    return exacta


def predecir(args: argparse.Namespace) -> list[ImageEval]:
    from ultralytics import YOLO

    modelo = YOLO(str(args.weights))
    umbrales_luz = LightingThresholds(night_max=70.0, day_min=110.0)
    salida: list[ImageEval] = []
    for ruta in sorted(args.images.glob("*.jpg")):
        imagen = cv2.imread(str(ruta))
        alto, ancho = imagen.shape[:2]
        r = modelo.predict(
            imagen,
            conf=0.001,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
            device=args.device,
            max_det=100,
        )[0]
        preds = [
            Box(*map(float, xyxy), cls=int(c), conf=float(p))
            for xyxy, c, p in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                strict=True,
            )
        ]
        m = PATRON.search(ruta.stem)
        video, frame = (m.group(1), int(m.group(2))) if m else (ruta.stem, 0)
        luz = classify_luma(float(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY).mean()), umbrales_luz)
        salida.append(
            ImageEval(
                name=ruta.stem,
                truth=read_yolo_labels(etiqueta_de(ruta, args.labels), ancho, alto),
                preds=preds,
                # Bloques de `block` frames anotados consecutivos (cada 6 frames de
                # video): unos 2.5 s de material a 12 fps con block=5.
                group=f"{video}_{frame // (6 * args.block)}",
                tags={"video": video, "luz": luz.value},
            )
        )
    return salida


def resumir(
    imagenes: list[ImageEval], conf: float, replicas: int, por_clase: dict[str, float]
) -> dict[str, object]:
    reporte: dict[str, object] = {}
    for cls, nombre in NOMBRES.items():
        umbral = por_clase.get(nombre, conf)
        puntual = class_metrics(imagenes, cls, umbral)
        ic = bootstrap_ci(imagenes, cls, umbral, replicas=replicas)
        reporte[nombre] = {
            **{k: round(v, 4) for k, v in puntual.items()},
            "ic95": {k: [round(a, 4), round(b, 4)] for k, (a, b) in ic.items()},
        }
    return reporte


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de operación")
    parser.add_argument(
        "--class-conf",
        nargs="*",
        default=[],
        help="Umbral por clase, como en configs/: bulldozer=0.05",
    )
    parser.add_argument("--iou", type=float, default=0.45, help="NMS, igual que en configs/")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--replicas", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    por_clase = {k: float(v) for k, v in (par.split("=") for par in args.class_conf)}
    imagenes = predecir(args)
    reporte: dict[str, object] = {
        "pesos": str(args.weights),
        "imagenes": len(imagenes),
        "bloques": len({i.group for i in imagenes}),
        "umbral_operacion": args.conf,
        "umbral_por_clase": por_clase,
        "nms_iou": args.iou,
        "por_clase": resumir(imagenes, args.conf, args.replicas, por_clase),
        "barrido": {
            nombre: threshold_sweep(imagenes, cls, np.round(np.arange(0.05, 0.96, 0.05), 2))
            for cls, nombre in NOMBRES.items()
        },
        "por_luz": {},
    }
    for luz in sorted({i.tags["luz"] for i in imagenes}):
        subconjunto = [i for i in imagenes if i.tags["luz"] == luz]
        reporte["por_luz"][luz] = {  # type: ignore[index]
            "imagenes": len(subconjunto),
            **resumir(subconjunto, args.conf, max(200, args.replicas // 4), por_clase),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    for nombre, m in reporte["por_clase"].items():  # type: ignore[attr-defined]
        print(
            f"{nombre:10s} AP50 {m['ap50']:.3f} {m['ic95']['ap50']}  "
            f"P@{args.conf} {m['precision']:.3f}  R {m['recall']:.3f}  n={m['n_truth']:.0f}"
        )


if __name__ == "__main__":
    main()
