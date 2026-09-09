"""Fine-tuning del detector de maquinaria.

Herramienta de desarrollo. Ejecuta la decisión del ADR 0002: especializar un
detector en el dominio, tras haber descartado el enfoque zero-shot con evidencia.

Los valores por defecto están dimensionados para una GPU de 4 GB, que es la
restricción real del entorno de desarrollo. Los tres parámetros que la determinan
son ``--batch``, ``--imgsz`` y la precisión mixta, activada siempre en CUDA.

Sobre reproducibilidad: la semilla se fija y se registra, y el directorio de salida
queda bajo ``weights/``. Un modelo cuyo procedimiento de entrenamiento no se puede
repetir no es un resultado, es una anécdota.

Uso:
    python scripts/train_detector.py --data data/dataset/data.yaml --epochs 80
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bermguard.core.device import describe_runtime, resolve_device

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="data.yaml del dataset, en formato Ultralytics",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Checkpoint de partida. 'yolo11s-seg.pt' entrena mascaras en vez de cajas",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Bajar a 4 si aparece 'CUDA out of memory' con 4 GB",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolucion de entrenamiento. Debe coincidir con la de inferencia",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help=(
            "Capas del backbone a congelar. 10 congela el backbone completo: entrena "
            "mas rapido y sobreajusta menos con pocos datos, a costa de no adaptar las "
            "features de bajo nivel al dominio (polvo, contraste nocturno)"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Epocas sin mejora antes de detener. Evita sobreajustar un dataset chico",
    )
    parser.add_argument("--workers", type=int, default=2, help="2 en Windows; 8 en Linux")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("weights/training"))
    parser.add_argument("--name", default="detector")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.data.exists():
        raise SystemExit(f"No existe el descriptor del dataset: {args.data}")

    from ultralytics import YOLO

    device = resolve_device(args.device)  # type: ignore[arg-type]
    if device == "cpu":
        logger.warning(
            "Entrenando en CPU. Con este tamano de modelo el entrenamiento pasa de "
            "minutos a horas; conviene resolver el acceso a GPU antes de continuar."
        )

    modelo = YOLO(args.model)

    resultados = modelo.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        seed=args.seed,
        freeze=args.freeze or None,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        # Precision mixta: mitad de memoria de activaciones, y la GPU de desarrollo
        # es Ampere, con Tensor Cores que la aprovechan de verdad.
        amp=(device == "cuda"),
        # Augmentacion orientada a la brecha de dominio medida: el material recorre
        # luminancias de 19 a 163 y tiene polvo en suspension, mientras que los
        # datasets publicos de maquinaria son mayoritariamente diurnos y nitidos.
        hsv_v=0.6,  # variacion de brillo, bastante por encima del 0.4 por defecto
        hsv_s=0.5,  # saturacion: el polvo lava el color de la escena
        degrees=5.0,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Sin volteo vertical ni rotaciones fuertes: la maquinaria pesada tiene una
        # orientacion canonica respecto al suelo y romperla ensena un invariante falso.
        flipud=0.0,
        plots=True,
    )

    directorio = Path(resultados.save_dir)
    resumen = {
        "modelo_base": args.model,
        "dataset": str(args.data),
        "epocas": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "freeze": args.freeze,
        "semilla": args.seed,
        "runtime": describe_runtime(device),
        "pesos": str(directorio / "weights" / "best.pt"),
    }
    (directorio / "resumen_entrenamiento.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nPesos: {directorio / 'weights' / 'best.pt'}")
    print(f"Curvas y matriz de confusion: {directorio}")
    print("\nPara usarlo en el pipeline, apuntar 'detector.weights' en configs/ a esa ruta.")


if __name__ == "__main__":
    main()
