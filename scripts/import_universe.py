"""Arma un export YOLO a partir de las anotaciones públicas de un proyecto de Universe.

Hay proyectos públicos de Roboflow Universe sin ninguna versión generada (por ejemplo
``mines-project/mines-jo13g``), así que no tienen zip de descarga. Sus imágenes y
anotaciones igual son públicas: la página de cada imagen las pide a
``/query/source``. Ese JSON se junta desde el navegador (ver ADR 0004), y este script
lo convierte al mismo formato que un export "YOLOv8", para que ``prepare_pretrain.py``
lo trate como a cualquier otra fuente.

Entrada: ``{"imagenes": [{"id", "name", "split", "url", "anotacion": {"width",
"height", "boxes": [...]}}]}``. Los polígonos pasan como polígonos y las cajas como
``cx cy w h``, todo normalizado.

Uso:
    python scripts/import_universe.py data/externos/mines/anotaciones_universe.json \\
        --out data/externos/mines
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

PARTICIONES = ("train", "valid", "test")


def linea_yolo(caja: dict, indice: int, ancho: float, alto: float) -> str:
    """Una caja o polígono de Roboflow, en píxeles, a una línea YOLO normalizada."""
    puntos = caja.get("points")
    if caja.get("type") == "polygon" and puntos and len(puntos) >= 3:
        coords = [f"{x / ancho:.6f} {y / alto:.6f}" for x, y in puntos]
        return " ".join([str(indice), *coords])
    cx, cy = float(caja["x"]) / ancho, float(caja["y"]) / alto
    w, h = float(caja["width"]) / ancho, float(caja["height"]) / alto
    return f"{indice} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def descargar(url: str, intentos: int = 3) -> bytes:
    for intento in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=60) as respuesta:
                return respuesta.read()
        except OSError:
            if intento == intentos - 1:
                raise
            time.sleep(5)
    raise AssertionError("inalcanzable")


def guardar_orientada(contenido: bytes, destino: Path, lado_max: int) -> tuple[int, int]:
    """Aplica la rotación EXIF a los píxeles y achica la imagen.

    Roboflow anota sobre la imagen ya rotada, así que la etiqueta sólo calza si el
    lector también rota. Guardarla rotada y sin EXIF quita esa dependencia; las
    coordenadas normalizadas no cambian al achicar.
    """
    imagen = ImageOps.exif_transpose(Image.open(io.BytesIO(contenido))).convert("RGB")
    imagen.thumbnail((lado_max, lado_max))
    imagen.save(destino, quality=92)
    return imagen.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anotaciones", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1920, help="Lado mayor al guardar")
    args = parser.parse_args()

    datos = json.loads(args.anotaciones.read_text(encoding="utf-8"))
    imagenes = [im for im in datos["imagenes"] if im.get("anotacion")]
    sin_anotacion = len(datos["imagenes"]) - len(imagenes)
    nombres = sorted({c["label"] for im in imagenes for c in im["anotacion"]["boxes"]})

    for im in imagenes:
        particion = im["split"] if im["split"] in PARTICIONES else "train"
        carpeta = args.out / particion
        (carpeta / "images").mkdir(parents=True, exist_ok=True)
        (carpeta / "labels").mkdir(parents=True, exist_ok=True)
        stem = Path(im["name"]).stem
        anotacion = im["anotacion"]
        ancho, alto = float(anotacion["width"]), float(anotacion["height"])
        lineas = [linea_yolo(c, nombres.index(c["label"]), ancho, alto) for c in anotacion["boxes"]]
        (carpeta / "labels" / f"{stem}.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
        imagen = carpeta / "images" / f"{stem}.jpg"
        if not imagen.exists():
            w, h = guardar_orientada(descargar(im["url"]), imagen, args.max_side)
            if (w > h) != (ancho > alto):
                raise SystemExit(f"{im['name']}: orientación {w}x{h} no calza con {ancho}x{alto}")

    (args.out / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
        f"nc: {len(nombres)}\nnames: {nombres}\n\n"
        f"# fuente: {datos.get('fuente', '')} ({datos.get('licencia', '')})\n",
        encoding="utf-8",
    )
    print(f"{len(imagenes)} imágenes, clases {nombres}, {sin_anotacion} sin anotación")


if __name__ == "__main__":
    main()
