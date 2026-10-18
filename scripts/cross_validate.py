"""Validación cruzada dejando un video fuera (LOVO), con y sin copy-paste.

Con cuatro videos, una sola partición entrena con tres y valida con uno: es una
muestra de tamaño uno de "cómo rinde el detector en un video nuevo". LOVO entrena
cuatro veces, cada una dejando fuera un video distinto, y reporta la media y la
dispersión entre folds. La dispersión entre folds es la respuesta honesta a
"¿cuánto cambia el resultado si el video nuevo es otro?".

Cada fold se entrena con los mismos hiperparámetros, llamando a
``train_detector.py``, y se evalúa con ``evaluate_detector.py``, de modo que no hay
una segunda definición de ninguno de los dos.

La variante ``copypaste`` agrega, **sólo a la partición de entrenamiento del
fold**, un frame aumentado por cada frame de entrenamiento, con uno o dos
bulldozers recortados de los videos de entrenamiento de ese mismo fold. Un recorte
del video de validación sería fuga.

Uso:
    python scripts/cross_validate.py --variants base copypaste
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from bermguard.training.copy_paste import (
    CLASE_BULLDOZER,
    extract_cutout,
    format_polygons,
    parse_polygons,
    paste,
)

PATRON = re.compile(r"(video_\d+)_f(\d+)")
VIDEOS = ("video_01", "video_02", "video_03", "video_04")


def recolectar(origen: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Pares (imagen, etiqueta) de todo el dataset, agrupados por video."""
    por_video: dict[str, list[tuple[Path, Path]]] = {v: [] for v in VIDEOS}
    for particion in ("train", "val"):
        for imagen in sorted((origen / "images" / particion).glob("*.jpg")):
            etiqueta = origen / "labels" / particion / f"{imagen.stem}.txt"
            m = PATRON.search(imagen.stem)
            if m and etiqueta.exists():
                por_video[m.group(1)].append((imagen, etiqueta))
    return por_video


def enlazar(origen: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def aumentar(
    pares: list[tuple[Path, Path]], destino_img: Path, destino_lbl: Path, seed: int
) -> int:
    """Escribe un frame aumentado por cada frame de entrenamiento. Devuelve los pegados."""
    rng = np.random.default_rng(seed)
    recortes = []
    for imagen, etiqueta in pares:
        img = cv2.imread(str(imagen))
        alto, ancho = img.shape[:2]
        for p in parse_polygons(etiqueta.read_text(encoding="utf-8"), ancho, alto):
            if p.cls == CLASE_BULLDOZER:
                r = extract_cutout(img, p)
                if r is not None:
                    recortes.append(r)
    if not recortes:
        return 0

    pegados = 0
    for imagen, etiqueta in pares:
        img = cv2.imread(str(imagen))
        alto, ancho = img.shape[:2]
        poligonos = parse_polygons(etiqueta.read_text(encoding="utf-8"), ancho, alto)
        for _ in range(int(rng.integers(1, 3))):
            r = recortes[int(rng.integers(len(recortes)))]
            resultado = paste(img, poligonos, r, rng)
            if resultado is None:
                continue
            img, poligonos = resultado
            pegados += 1
        nombre = f"{PATRON.search(imagen.stem).group(0)}_cp"  # type: ignore[union-attr]
        cv2.imwrite(str(destino_img / f"{nombre}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (destino_lbl / f"{nombre}.txt").write_text(
            format_polygons(poligonos, ancho, alto), encoding="utf-8"
        )
    return pegados


def preparar_fold(
    por_video: dict[str, list[tuple[Path, Path]]], val: str, variante: str, raiz: Path, seed: int
) -> tuple[Path, dict[str, int]]:
    fold = raiz / variante / val
    if fold.exists():
        shutil.rmtree(fold)
    conteo = {"train": 0, "val": 0, "pegados": 0}
    entrenamiento: list[tuple[Path, Path]] = []
    for video, pares in por_video.items():
        particion = "val" if video == val else "train"
        for imagen, etiqueta in pares:
            nombre = PATRON.search(imagen.stem).group(0)  # type: ignore[union-attr]
            enlazar(imagen, fold / "images" / particion / f"{nombre}.jpg")
            enlazar(etiqueta, fold / "labels" / particion / f"{nombre}.txt")
            conteo[particion] += 1
            if particion == "train":
                entrenamiento.append((imagen, etiqueta))
    if variante == "copypaste":
        conteo["pegados"] = aumentar(
            entrenamiento, fold / "images" / "train", fold / "labels" / "train", seed
        )
    (fold / "data.yaml").write_text(
        f"path: {fold.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "nc: 2\nnames: ['caex', 'bulldozer']\n",
        encoding="utf-8",
    )
    return fold, conteo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--work", type=Path, default=Path("data/cv"))
    parser.add_argument("--variants", nargs="+", default=["base", "copypaste"])
    parser.add_argument("--folds", nargs="+", default=list(VIDEOS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="4 y no 8: con el escritorio abierto quedan ~2.6 GB de VRAM y con 8 Windows "
        "pagina la GPU a RAM (350 s por iteración). Igual para todas las variantes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 en Windows: relanzar los workers en cada época lleva la época de 22 s a 42 s.",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/experimentos/lovo"))
    parser.add_argument("--skip-train", action="store_true", help="Sólo reevaluar")
    args = parser.parse_args()

    por_video = recolectar(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    for variante in args.variants:
        for val in args.folds:
            proyecto = Path("weights/cv") / variante
            pesos = proyecto / val / "weights" / "best.pt"
            fold, conteo = preparar_fold(por_video, val, variante, args.work, args.seed)
            print(f"[{variante}/{val}] {conteo}", flush=True)
            if not args.skip_train:
                subprocess.run(
                    [
                        python,
                        "scripts/train_detector.py",
                        "--data",
                        str(fold / "data.yaml"),
                        "--epochs",
                        str(args.epochs),
                        "--seed",
                        str(args.seed),
                        "--batch",
                        str(args.batch),
                        "--workers",
                        str(args.workers),
                        "--project",
                        str(proyecto),
                        "--name",
                        val,
                    ],
                    check=True,
                )
            # Ultralytics 8.4 antepone runs/detect a un proyecto relativo.
            candidatos = [pesos, Path("runs/detect") / pesos]
            pesos = next((c for c in candidatos if c.exists()), pesos)
            salida = args.out / f"{variante}_{val}.json"
            subprocess.run(
                [
                    python,
                    "scripts/evaluate_detector.py",
                    "--weights",
                    str(pesos),
                    "--images",
                    str(fold / "images" / "val"),
                    "--labels",
                    str(fold / "labels" / "val"),
                    "--out",
                    str(salida),
                ],
                check=True,
            )
            reporte = json.loads(salida.read_text(encoding="utf-8"))
            reporte["particion"] = conteo
            salida.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
