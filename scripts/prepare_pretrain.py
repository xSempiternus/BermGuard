"""Junta uno o más datasets públicos exportados en formato YOLO para preentrenar.

Cada ``--source`` es la carpeta de un export de Roboflow ("YOLOv8" o "YOLO11"):
un ``data.yaml`` con ``names`` y particiones ``train``/``valid``/``test`` con
``images`` y ``labels``. ``--map`` traduce los nombres de clase de origen a
``caex`` o ``bulldozer``; las clases sin traducción se descartan
(``bermguard/training/pretrain.py``).

Ninguna imagen de estos datasets viene de los videos del proyecto, así que el
modelo preentrenado sirve para los cuatro folds de ``cross_validate.py`` sin fuga.

Uso:
    python scripts/prepare_pretrain.py \\
        --source data/externos/dji_65 --source data/externos/bulldozer-lpoe2 \\
        --map dozer=bulldozer bull_dozer=bulldozer tippertruck=caex dump_truck=caex
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from bermguard.training.pretrain import (
    CLASES_PROYECTO,
    normalizar,
    parse_class_map,
    remap_labels,
)

PARTICIONES = {"train": "train", "valid": "val", "val": "val", "test": "val"}


def nombres_de(data_yaml: Path) -> list[str]:
    nombres = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(nombres, dict):
        return [nombres[k] for k in sorted(nombres)]
    return list(nombres)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--map", nargs="+", required=True, help="origen=caex|bulldozer")
    parser.add_argument("--out", type=Path, default=Path("data/pretrain"))
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Conservar imágenes sin clases traducidas, como ejemplos de fondo",
    )
    args = parser.parse_args()

    mapa = parse_class_map(args.map)
    if args.out.exists():
        shutil.rmtree(args.out)

    resumen: dict[str, dict[str, int]] = {}
    for fuente in args.source:
        nombres = nombres_de(fuente / "data.yaml")
        sin_traducir = sorted(n for n in nombres if normalizar(n) not in mapa)
        conteo = {"imagenes": 0, "vacias_descartadas": 0, "caex": 0, "bulldozer": 0}
        for carpeta, particion in PARTICIONES.items():
            for etiqueta in sorted((fuente / carpeta / "labels").glob("*.txt")):
                imagenes = list((fuente / carpeta / "images").glob(f"{etiqueta.stem}.*"))
                if not imagenes:
                    continue
                texto = remap_labels(etiqueta.read_text(encoding="utf-8"), nombres, mapa)
                if not texto and not args.keep_empty:
                    conteo["vacias_descartadas"] += 1
                    continue
                nombre = f"{fuente.name}__{etiqueta.stem}"
                destino_img = args.out / "images" / particion / f"{nombre}{imagenes[0].suffix}"
                destino_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(imagenes[0], destino_img)
                destino_lbl = args.out / "labels" / particion / f"{nombre}.txt"
                destino_lbl.parent.mkdir(parents=True, exist_ok=True)
                destino_lbl.write_text(texto, encoding="utf-8")
                conteo["imagenes"] += 1
                for linea in texto.splitlines():
                    conteo[CLASES_PROYECTO[int(linea.split()[0])]] += 1
        resumen[fuente.name] = conteo
        print(f"{fuente.name}: {conteo}; descartadas: {sin_traducir}")

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 2\nnames: {list(CLASES_PROYECTO)}\n",
        encoding="utf-8",
    )
    (args.out / "resumen.json").write_text(
        json.dumps({"mapa": args.map, "fuentes": resumen}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
