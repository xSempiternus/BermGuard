"""Benchmark de exactitud del pretil sobre escenas sintéticas con la cresta conocida.

El material no trae el perfil real del pretil, así que sobre él sólo se mide
estabilidad. Este benchmark mide **exactitud** en condiciones controladas: recorre
la cantidad de huellas de neumático (el distractor que el análisis del material
identificó como el problema principal) y el contraste de la cresta, y para cada
combinación reporta la fracción de columnas a 5 px o menos de la cresta verdadera.

Mide la formulación, no el sistema en el material real: una escena sintética
reproduce los hechos medidos que hacen difícil el problema, no la escena entera.

Uso:
    python scripts/benchmark_berm_synthetic.py --out docs/experimentos/pretil_sintetico.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bermguard.benchmark.synthetic import SceneSpec, crest_error_px, make_scene
from bermguard.vision.berm.argmax_baseline import ArgmaxBermSegmenter
from bermguard.vision.berm.classical import ClassicalBermSegmenter

METODOS = {"camino_optimo": ClassicalBermSegmenter, "argmax": ArgmaxBermSegmenter}


def evaluar(spec: SceneSpec, semillas: int) -> dict[str, dict[str, float]]:
    salida: dict[str, dict[str, float]] = {}
    for nombre, clase in METODOS.items():
        dentro, mae = [], []
        for semilla in range(semillas):
            escena = make_scene(spec, seed=semilla)
            perfil = clase().segment(escena.image)
            if perfil is None:
                dentro.append(0.0)
                continue
            error = crest_error_px(perfil.crest_y_px, escena.crest_y_px)
            dentro.append(error["within_5"])
            mae.append(error["mae"])
        arreglo = np.asarray(dentro)
        # Error estándar entre semillas: cada semilla es una escena independiente.
        salida[nombre] = {
            "within_5": round(float(arreglo.mean()), 4),
            "within_5_se": round(float(arreglo.std(ddof=1) / np.sqrt(arreglo.size)), 4),
            "mae_px": round(float(np.mean(mae)), 2) if mae else float("nan"),
        }
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    filas = []
    for contraste in (8.0, 12.0, 18.0):
        for huellas in (0, 2, 4, 6, 8, 10, 12):
            resultado = evaluar(SceneSpec(crest_contrast=contraste, tracks=huellas), args.seeds)
            filas.append({"contraste": contraste, "huellas": huellas, **resultado})
            print(
                f"contraste {contraste:4.0f}  huellas {huellas:2d}  "
                + "  ".join(f"{k} {v['within_5']:.2f}" for k, v in resultado.items())
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"semillas": args.seeds, "metrica": "fraccion de columnas a <=5 px", "filas": filas},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
