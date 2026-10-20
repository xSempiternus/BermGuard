"""Validación cruzada dejando un video fuera (LOVO): base, copy-paste y preentrenado.

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

La variante ``pretrain`` usa la misma partición que ``base`` y cambia sólo el punto
de partida: en vez de los pesos COCO, un modelo afinado antes con datasets públicos
de maquinaria (``scripts/prepare_pretrain.py``, ADR 0004). Ese preentrenamiento se
hace una sola vez y sirve para los cuatro folds, porque ninguna de esas imágenes
viene de los videos del proyecto.

En cada fold la partición ``val`` es el video excluido, que es también el de
evaluación. Por eso los folds se entrenan con épocas fijas, sin paciencia y sin
validar en cada época, y se evalúa ``last.pt``: si ``best.pt`` o el corte temprano se
eligieran mirando ese video, la métrica saldría optimista y favorecería a las
variantes con curvas más ruidosas.

Un fold terminado deja ``resumen_entrenamiento.json`` junto a sus pesos. Si la
corrida se corta (Ctrl+C para liberar la GPU), volver a lanzar el mismo comando
salta los folds terminados y reentrena desde cero el que quedó a medias.

Uso:
    python scripts/cross_validate.py --variants base copypaste pretrain
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
from bermguard.training.etiquetas import homogeneizar

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
            # Se escribe en vez de enlazar: un archivo que mezcla cajas y polígonos
            # haría que Ultralytics descarte el frame (bermguard/training/etiquetas.py).
            destino = fold / "labels" / particion / f"{nombre}.txt"
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(homogeneizar(etiqueta.read_text(encoding="utf-8")), encoding="utf-8")
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


def resolver_pesos(pesos: Path) -> Path:
    # Ultralytics 8.4 antepone runs/detect a un proyecto relativo.
    candidatos = [pesos, Path("runs/detect") / pesos]
    return next((c for c in candidatos if c.exists()), pesos)


def entrenar(
    python: str,
    data: Path,
    proyecto: Path,
    nombre: str,
    args: argparse.Namespace,
    modelo: str,
    epocas: int,
    extra: tuple[str, ...] = (),
) -> None:
    subprocess.run(
        [
            python,
            "scripts/train_detector.py",
            "--data",
            str(data),
            "--model",
            modelo,
            "--epochs",
            str(epocas),
            "--seed",
            str(args.seed),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--project",
            str(proyecto),
            "--name",
            nombre,
            "--device",
            args.device,
            *extra,
        ],
        check=True,
    )


def terminado(proyecto: Path, nombre: str) -> bool:
    """Si el entrenamiento llegó al final: el resumen se escribe sólo al terminar."""
    return resolver_pesos(proyecto / nombre / "resumen_entrenamiento.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--work", type=Path, default=Path("data/cv"))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "copypaste"],
        choices=["base", "copypaste", "pretrain"],
    )
    parser.add_argument("--folds", nargs="+", default=list(VIDEOS))
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Fijas, sin paciencia. detector_v1 seguía mejorando en la época 40 de 42",
    )
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
    parser.add_argument(
        "--pretrain-data",
        type=Path,
        default=Path("data/pretrain/data.yaml"),
        help="Salida de prepare_pretrain.py; sólo para la variante pretrain",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("docs/experimentos/lovo"))
    parser.add_argument("--skip-train", action="store_true", help="Sólo reevaluar")
    parser.add_argument(
        "--base-model",
        default="weights/yolo11s.pt",
        help="Pesos COCO de partida; la copia local evita depender de la red",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=Path("weights"),
        help="Raíz de pesos. Cambiarla en pruebas, para no dejar folds 'terminados' falsos",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="cpu sólo para pruebas de humo: entrenar en CPU tarda horas por fold",
    )
    args = parser.parse_args()

    por_video = recolectar(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    # El preentrenamiento sí usa best.pt y paciencia: su validación son imágenes
    # públicas, no los videos del proyecto.
    pesos_previos = resolver_pesos(
        args.weights_root / "pretrain" / "externo" / "weights" / "best.pt"
    )
    if "pretrain" in args.variants and not terminado(args.weights_root / "pretrain", "externo"):
        if args.skip_train or not args.pretrain_data.exists():
            raise SystemExit(
                f"Falta {pesos_previos} y no se puede entrenar: corré antes prepare_pretrain.py"
            )
        print(f"[pretrain] preentrenando con {args.pretrain_data}", flush=True)
        entrenar(
            python,
            args.pretrain_data,
            args.weights_root / "pretrain",
            "externo",
            args,
            args.base_model,
            args.pretrain_epochs,
        )
        pesos_previos = resolver_pesos(pesos_previos)

    for variante in args.variants:
        modelo = str(pesos_previos) if variante == "pretrain" else args.base_model
        for val in args.folds:
            proyecto = args.weights_root / "cv" / variante
            pesos = proyecto / val / "weights" / "last.pt"
            fold, conteo = preparar_fold(por_video, val, variante, args.work, args.seed)
            print(f"[{variante}/{val}] {conteo}", flush=True)
            if args.skip_train or terminado(proyecto, val):
                print(f"[{variante}/{val}] ya entrenado, sólo se evalúa", flush=True)
            else:
                entrenar(
                    python,
                    fold / "data.yaml",
                    proyecto,
                    val,
                    args,
                    modelo,
                    args.epochs,
                    extra=("--patience", "0", "--no-val"),
                )
            pesos = resolver_pesos(pesos)
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
                    "--device",
                    args.device,
                ],
                check=True,
            )
            reporte = json.loads(salida.read_text(encoding="utf-8"))
            reporte["particion"] = conteo
            salida.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
