"""Divide el dataset anotado en entrenamiento y validación, agrupando por video.

Herramienta de desarrollo. Existe porque la partición aleatoria que ofrecen por
defecto las herramientas de anotación **mide lo que no corresponde** en este
contexto.

Los frames de un mismo video están fuertemente correlacionados: el frame 60 y el
66 comparten escena, iluminación, encuadre y las mismas máquinas en casi la misma
posición. Si uno cae en entrenamiento y el otro en validación, la métrica de
validación premia haber memorizado un frame casi idéntico y no haber aprendido a
generalizar. El número sube y no significa nada.

La partición correcta agrupa por **video**: uno de los clips se reserva íntegro
para validación, de modo que la métrica responde la pregunta que importa —cómo se
comporta el modelo sobre material que nunca vio.

Acepta la salida de una exportación en formato YOLO, incluidos los nombres que
Roboflow genera al reescribirlos (``video_02_f00120_jpg.rf.<hash>.jpg``).

Uso:
    python scripts/split_dataset.py data/export --out data/dataset --val video_03
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

PATRON_VIDEO = re.compile(r"(video_\d+)")
"""Extrae el video de origen del nombre de archivo.

Las herramientas de anotación suelen reescribir los nombres añadiendo sufijos y
hashes. Lo que se conserva en todos los casos observados es el prefijo original,
así que se busca ese patrón en lugar de asumir una estructura exacta.
"""

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png"}


def origen(nombre: str) -> str | None:
    coincidencia = PATRON_VIDEO.search(nombre)
    return coincidencia.group(1) if coincidencia else None


def recolectar(raiz: Path) -> dict[str, list[Path]]:
    """Agrupa por video todas las imágenes bajo ``raiz``, recorriendo subdirectorios.

    Recorre en profundidad porque una exportación puede venir ya dividida en
    ``train/``, ``valid/`` y ``test/``. Esa división se descarta: es justamente la
    que este script viene a reemplazar.
    """
    grupos: dict[str, list[Path]] = defaultdict(list)
    sin_clasificar: list[Path] = []

    for ruta in raiz.rglob("*"):
        if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES_IMAGEN:
            continue
        video = origen(ruta.name)
        if video is None:
            sin_clasificar.append(ruta)
        else:
            grupos[video].append(ruta)

    if sin_clasificar:
        logger.warning(
            "%d imagenes sin video identificable en el nombre; quedan fuera. Ejemplo: %s",
            len(sin_clasificar),
            sin_clasificar[0].name,
        )
    return grupos


def etiqueta_de(imagen: Path) -> Path | None:
    """Ubica el ``.txt`` correspondiente a una imagen.

    Busca primero junto a la imagen y luego en un directorio ``labels`` hermano,
    que son los dos diseños que producen las exportaciones habituales.
    """
    candidatos = [
        imagen.with_suffix(".txt"),
        imagen.parent.parent / "labels" / f"{imagen.stem}.txt",
    ]
    return next((c for c in candidatos if c.exists()), None)


def copiar(imagenes: list[Path], destino: Path, particion: str) -> tuple[int, int]:
    dir_img = destino / "images" / particion
    dir_lbl = destino / "labels" / particion
    dir_img.mkdir(parents=True, exist_ok=True)
    dir_lbl.mkdir(parents=True, exist_ok=True)

    copiadas = 0
    sin_etiqueta = 0
    for imagen in imagenes:
        etiqueta = etiqueta_de(imagen)
        if etiqueta is None:
            # Un frame sin caja es informacion valida: le ensena al modelo como se
            # ve una escena vacia y reduce falsos positivos. Pero sin archivo de
            # etiqueta Ultralytics no distingue "vacio" de "sin anotar", asi que se
            # crea vacio explicitamente.
            sin_etiqueta += 1
            (dir_lbl / f"{imagen.stem}.txt").write_text("", encoding="utf-8")
        else:
            shutil.copy2(etiqueta, dir_lbl / etiqueta.name)
        shutil.copy2(imagen, dir_img / imagen.name)
        copiadas += 1
    return copiadas, sin_etiqueta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="Directorio de la exportacion anotada")
    parser.add_argument("--out", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--val",
        default="video_03",
        help="Video reservado integro para validacion",
    )
    parser.add_argument("--clases", nargs="+", default=["caex", "bulldozer"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    grupos = recolectar(args.export)
    if not grupos:
        raise SystemExit(f"No se encontraron imagenes bajo {args.export}")

    print("Imagenes por video de origen:")
    for video in sorted(grupos):
        print(f"  {video}: {len(grupos[video])}")

    if args.val not in grupos:
        raise SystemExit(
            f"El video de validacion '{args.val}' no aparece en la exportacion. "
            f"Disponibles: {sorted(grupos)}"
        )

    if args.out.exists():
        shutil.rmtree(args.out)

    val = grupos[args.val]
    train = [img for video, imgs in grupos.items() if video != args.val for img in imgs]

    n_train, vacias_train = copiar(train, args.out, "train")
    n_val, vacias_val = copiar(val, args.out, "val")

    (args.out / "data.yaml").write_text(
        "# Generado por scripts/split_dataset.py\n"
        f"# Particion agrupada por video. Validacion: {args.val} completo.\n"
        "# La particion NO es aleatoria: frames vecinos del mismo clip estan\n"
        "# correlacionados y mezclarlos entre particiones infla la metrica.\n"
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(args.clases)}\n"
        f"names: {args.clases}\n",
        encoding="utf-8",
    )

    print(f"\ntrain: {n_train} imagenes ({vacias_train} sin cajas)")
    print(f"val:   {n_val} imagenes ({vacias_val} sin cajas)  <- {args.val}")
    print(f"\ndata.yaml en {args.out / 'data.yaml'}")

    proporcion = n_val / (n_train + n_val)
    if not 0.15 <= proporcion <= 0.35:
        logger.warning(
            "La validacion es el %.0f%% del total. Fuera del rango habitual "
            "15-35%%; considerar otro video con --val.",
            proporcion * 100,
        )


if __name__ == "__main__":
    main()
