"""Caracteriza el material de entrada antes de diseñar sobre él.

Herramienta de desarrollo. Mide lo que los supuestos habituales dan por sentado:
que todos los videos comparten resolución y tasa de frames, que cada archivo es
una toma continua, y que la condición lumínica es una propiedad del archivo.

Ninguna de las tres se cumple en este material, y las tres tienen consecuencias
de diseño. Los números que produce este script sustentan
``docs/analisis_material.md`` y los ADR 0002 y 0003.

Uso:
    python scripts/probe_videos.py data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bermguard.io.video_reader import VideoReader, discover_videos
from bermguard.vision.lighting import classify_luma, mean_luma

CUT_LUMA_DELTA = 20.0
"""Salto de luminancia media entre frames consecutivos que se considera un corte.

Calibrado sobre este material: las transiciones graduales de ``video_02`` a
``video_04`` se mueven varias unidades por frame, mientras que los cortes duros de
``video_01`` saltan más de 20. Un detector basado sólo en cambio estructural no
los separa igual de bien, porque un corte entre dos tomas del mismo encuadre
cambia poco la estructura y mucho la exposición.
"""

# La clasificación lumínica vive en bermguard.vision.lighting porque el pipeline
# también la necesita (ADR 0003). Duplicarla acá haría que este reporte y el
# sistema pudieran divergir en silencio.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directorio con los videos")
    args = parser.parse_args()

    for video in discover_videos(args.input):
        lumas: list[float] = []
        with VideoReader(video) as reader:
            info = reader.info
            for _index, _timestamp, frame in reader.frames():
                lumas.append(mean_luma(frame))

        serie = np.asarray(lumas, dtype=np.float64)
        deltas = np.abs(np.diff(serie))
        cortes = np.flatnonzero(deltas > CUT_LUMA_DELTA)

        print(f"\n=== {video.name} ===")
        print(
            f"  {info.width}x{info.height} @ {info.fps:g} fps, "
            f"{len(serie)} frames decodificados ({info.frame_count} declarados)"
        )
        print(f"  luminancia: min={serie.min():.0f} max={serie.max():.0f} media={serie.mean():.0f}")

        if cortes.size:
            print(f"  cortes duros: {cortes.size}")
            for i in cortes:
                print(
                    f"    t={i / info.fps:.2f}s (frame {i})  "
                    f"luma {serie[i]:.0f} -> {serie[i + 1]:.0f}"
                )
        else:
            print("  cortes duros: ninguno (transiciones graduales)")

        condiciones = [classify_luma(v).value for v in serie]
        reparto = {c: condiciones.count(c) / len(condiciones) for c in set(condiciones)}
        reparto_txt = "  ".join(
            f"{c}={p:.0%}" for c, p in sorted(reparto.items(), key=lambda kv: -kv[1])
        )
        print(f"  condicion luminica por frame: {reparto_txt}")


if __name__ == "__main__":
    main()
