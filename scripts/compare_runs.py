"""Compara dos corridas completas del pipeline, video por video y por condición de luz.

Sirve para medir el efecto de un cambio con todo lo demás fijo: la misma máquina, los
mismos videos y el mismo detector. Lee los ``metadata.json`` y ``berm_profile.csv``
que escribe cada corrida.

Uso:
    python scripts/compare_runs.py --before out_base --after out_new \\
        --out docs/experimentos/pretil_antes_despues.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def leer(raiz: Path) -> dict[str, dict[str, object]]:
    corridas: dict[str, dict[str, object]] = {}
    for meta in sorted(raiz.glob("*/method_*/metadata.json")):
        clave = f"{meta.parent.parent.name}/{meta.parent.name}"
        datos = json.loads(meta.read_text(encoding="utf-8"))
        filas = list(csv.DictReader((meta.parent / "berm_profile.csv").open(encoding="utf-8")))
        por_luz: dict[str, list[float]] = {}
        for fila in filas:
            altura = fila.get("height_m_median") or ""
            if altura:
                por_luz.setdefault(fila["lighting"], []).append(float(altura))
        corridas[clave] = {
            "jitter_mediana_px": datos.get("berm_crest_jitter_px"),
            "jitter_columna_px": datos.get("berm_crest_jitter_col_px"),
            "cobertura": datos.get("berm_coverage_mean"),
            "sin_pretil": datos.get("berm_dropout_rate"),
            "alertas": datos.get("proximity_alerts"),
            "altura_mediana_por_luz": {
                k: round(float(np.median(v)), 3) for k, v in sorted(por_luz.items())
            },
            "frames_con_altura": sum(len(v) for v in por_luz.values()),
        }
    return corridas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    antes, despues = leer(args.before), leer(args.after)
    reporte = {"antes": antes, "despues": despues}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    columnas = ("jitter_mediana_px", "jitter_columna_px", "cobertura", "sin_pretil")
    print(f"{'corrida':20s} " + " ".join(f"{c:>18s}" for c in columnas))
    for clave in sorted(set(antes) | set(despues)):
        a, d = antes.get(clave, {}), despues.get(clave, {})
        celdas = [f"{_fmt(a.get(c))} > {_fmt(d.get(c))}" for c in columnas]
        print(f"{clave:20s} " + " ".join(f"{x:>18s}" for x in celdas))


def _fmt(valor: object) -> str:
    return "-" if valor is None else f"{float(valor):.2f}"  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
