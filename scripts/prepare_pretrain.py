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
        --source data/externos/mines --source data/externos/bulldozer \\
        --map dozer=bulldozer bull_dozer=bulldozer dumper=caex dumb_truck=caex \\
        --require bulldozer --exclude data/pretrain_excluir.txt --repeat mines=5

``--exclude`` salta imágenes revisadas a mano (duplicados, etiquetas erróneas,
juguetes). ``--repeat mines=5`` copia cada imagen de Mines cinco veces en train
(sufijos ``_r1``...``_r4``) para que la fuente más parecida al rajo no quede
ahogada por las ~800 de bulldozer-lpoe2; val nunca se repite. Las instancias
degeneradas (polígonos de ancho o alto < 0,5 %) se quitan siempre.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from bermguard.training.pretrain import (
    CLASES_PROYECTO,
    contiene,
    descartar_degeneradas,
    leer_exclusiones,
    nombres_repetidos,
    normalizar,
    parse_class_map,
    parse_repeat,
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
    parser.add_argument(
        "--require",
        choices=CLASES_PROYECTO,
        help=(
            "Conservar sólo imágenes con al menos una instancia de esta clase. Con "
            "'bulldozer' se descartan las imágenes sin dozer; en las que quedan, los "
            "camiones siguen etiquetados como caex, porque sin etiqueta enseñarían que un "
            "camión es fondo"
        ),
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        help=(
            "Lista de imágenes a saltar (data/pretrain_excluir.txt): un nombre de archivo de "
            "origen por línea, sin el prefijo de la fuente, con '# motivo' opcional"
        ),
    )
    parser.add_argument(
        "--repeat",
        nargs="+",
        default=[],
        metavar="FUENTE=N",
        help=(
            "Sobremuestrear una fuente en train: cada imagen entra N veces (copias con sufijo "
            "_rN). Nunca se repite en val"
        ),
    )
    args = parser.parse_args()

    mapa = parse_class_map(args.map)
    repeticiones = parse_repeat(args.repeat)
    excluir = leer_exclusiones(args.exclude.read_text(encoding="utf-8")) if args.exclude else set()
    desconocidas = set(repeticiones) - {f.name for f in args.source}
    if desconocidas:
        parser.error(f"--repeat nombra fuentes que no están en --source: {sorted(desconocidas)}")
    if args.out.exists():
        shutil.rmtree(args.out)

    resumen: dict[str, dict[str, int]] = {}
    for fuente in args.source:
        nombres = nombres_de(fuente / "data.yaml")
        sin_traducir = sorted(n for n in nombres if normalizar(n) not in mapa)
        veces = repeticiones.get(fuente.name, 1)
        conteo = {
            "imagenes": 0,
            "excluidas": 0,
            "vacias_descartadas": 0,
            "sin_clase_requerida": 0,
            "instancias_degeneradas": 0,
            "caex": 0,
            "bulldozer": 0,
            "train": 0,
            "val": 0,
            "repeticion_train": veces,
            "train_con_repeticion": 0,
        }
        for carpeta, particion in PARTICIONES.items():
            for etiqueta in sorted((fuente / carpeta / "labels").glob("*.txt")):
                imagenes = list((fuente / carpeta / "images").glob(f"{etiqueta.stem}.*"))
                if not imagenes:
                    continue
                if imagenes[0].name in excluir or etiqueta.stem in excluir:
                    conteo["excluidas"] += 1
                    continue
                texto = remap_labels(etiqueta.read_text(encoding="utf-8"), nombres, mapa)
                texto, degeneradas = descartar_degeneradas(texto)
                conteo["instancias_degeneradas"] += degeneradas
                if not texto and not args.keep_empty:
                    conteo["vacias_descartadas"] += 1
                    continue
                if args.require and not contiene(texto, args.require):
                    conteo["sin_clase_requerida"] += 1
                    continue
                base = f"{fuente.name}__{etiqueta.stem}"
                copias = nombres_repetidos(base, veces if particion == "train" else 1)
                for nombre in copias:
                    destino_img = args.out / "images" / particion / f"{nombre}{imagenes[0].suffix}"
                    destino_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(imagenes[0], destino_img)
                    destino_lbl = args.out / "labels" / particion / f"{nombre}.txt"
                    destino_lbl.parent.mkdir(parents=True, exist_ok=True)
                    destino_lbl.write_text(texto, encoding="utf-8")
                conteo["imagenes"] += 1
                conteo[particion] += 1
                if particion == "train":
                    conteo["train_con_repeticion"] += len(copias)
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
        json.dumps(
            {
                "mapa": args.map,
                "requiere": args.require,
                "excluir": args.exclude.as_posix() if args.exclude else None,
                "repetir": repeticiones,
                "fuentes": resumen,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
