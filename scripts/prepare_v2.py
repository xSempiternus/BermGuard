"""Carpeta de entrenamiento de ``detector_v2``: los cuatro videos, etiquetas homogeneizadas.

Es el paso siguiente a la validación cruzada (``docs/entrenamiento/sesion_gpu.md``). La
variante y las épocas no se eligen acá: se leen de la decisión de LOVO
(``docs/experimentos/lovo/resumen.json``), salvo que se fuercen con ``--variant`` y
``--epochs``.

El armado reutiliza el de ``cross_validate.py``, para que v2 entrene exactamente con los
datos que se validaron:

* cada etiqueta pasa por ``homogeneizar``: sin eso Ultralytics descarta 36 de los 51
  frames de ``video_01``, que mezclan cajas y polígonos;
* si ganó ``copypaste``, se agrega un frame aumentado por frame, con recortes de los
  cuatro videos (ya no hay video excluido del que cuidarse);
* si ganó ``pretrain``, los datos son los de ``base`` y cambia el punto de partida.

``val`` apunta a la misma carpeta de train. Ultralytics exige una partición de validación,
pero con ``--no-val`` sólo valida al final y no elige nada con ella. Esa cifra se mide
sobre frames que el modelo vio y no es una evaluación: el rendimiento esperado de v2 es
el de LOVO.

Uso:
    python scripts/prepare_v2.py                 # según resumen.json
    python scripts/prepare_v2.py --variant base --epochs 60
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cross_validate import PATRON, aumentar, enlazar, recolectar  # noqa: E402

from bermguard.training.etiquetas import homogeneizar  # noqa: E402

EPOCAS = 60
PESOS_PREENTRENADOS = Path("weights/pretrain/externo/weights/best.pt")


def decision(resumen: Path) -> tuple[str, int]:
    """Variante ganadora y épocas, según el resumen de LOVO."""
    datos = json.loads(resumen.read_text(encoding="utf-8"))["decision"]
    if datos.get("pendiente_control"):
        raise SystemExit(
            "La decisión de LOVO es provisoria: falta el control base2x. "
            "Correr scripts/gpu_session.py hasta el final."
        )
    return datos["ganadora"], EPOCAS * 2 if datos.get("epocas_dobles") else EPOCAS


def preparar(origen: Path, destino: Path, variante: str, seed: int) -> dict[str, int]:
    """Escribe ``destino`` con los cuatro videos en train. Devuelve los conteos."""
    if destino.exists():
        shutil.rmtree(destino)
    imagenes, etiquetas = destino / "images" / "train", destino / "labels" / "train"
    etiquetas.mkdir(parents=True)
    pares = [par for lista in recolectar(origen).values() for par in lista]
    for imagen, etiqueta in pares:
        nombre = PATRON.search(imagen.stem).group(0)  # type: ignore[union-attr]
        enlazar(imagen, imagenes / f"{nombre}.jpg")
        (etiquetas / f"{nombre}.txt").write_text(
            homogeneizar(etiqueta.read_text(encoding="utf-8")), encoding="utf-8"
        )
    pegados = aumentar(pares, imagenes, etiquetas, seed) if variante == "copypaste" else 0
    (destino / "data.yaml").write_text(
        "# Generado por scripts/prepare_v2.py. val = train: sólo para que Ultralytics\n"
        "# arranque; con --no-val no se elige nada con ella y no es una evaluación.\n"
        "train: images/train\nval: images/train\n"
        "nc: 2\nnames: ['caex', 'bulldozer']\n",
        encoding="utf-8",
    )
    return {"frames": len(pares), "imagenes": len(list(imagenes.glob("*.jpg"))), "pegados": pegados}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--out", type=Path, default=Path("data/v2"))
    parser.add_argument("--resumen", type=Path, default=Path("docs/experimentos/lovo/resumen.json"))
    parser.add_argument("--variant", choices=["base", "copypaste", "pretrain"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.variant:
        variante, epocas = args.variant, args.epochs or EPOCAS
    else:
        variante, epocas = decision(args.resumen)
        epocas = args.epochs or epocas
    conteo = preparar(args.dataset, args.out, variante, args.seed)
    modelo = PESOS_PREENTRENADOS if variante == "pretrain" else Path("weights/yolo11s.pt")
    if not modelo.exists() and (Path("runs/detect") / modelo).exists():
        modelo = Path("runs/detect") / modelo  # Ultralytics 8.4 antepone runs/detect
    print(f"[v2] variante {variante}, {epocas} épocas, {conteo}")
    print(
        "\nEntrenar con:\n"
        f"  python scripts/train_detector.py --data {args.out.as_posix()}/data.yaml "
        f"--model {modelo.as_posix()} --epochs {epocas} --patience 0 --no-val --batch 4 "
        "--workers 0 --name detector_v2"
    )


if __name__ == "__main__":
    main()
