"""Convierte los JSON de LOVO en la decisión de variante, el umbral y el bloque publicable.

Lee ``docs/experimentos/lovo/{variante}_{video}.json`` (salida de
``scripts/cross_validate.py``) y escribe ``resumen.json`` y ``resumen.md`` en el mismo
directorio. La lógica, la regla de decisión y el criterio de umbral están en
``bermguard/benchmark/lovo.py``.

Uso:
    python scripts/summarize_lovo.py
    python scripts/summarize_lovo.py --criterio recall_min --minimo 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bermguard.benchmark.lovo import (
    CriterioUmbral,
    ReglaDecision,
    a_markdown,
    cargar,
    limpiar_nan,
    resumir_lovo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("docs/experimentos/lovo"))
    parser.add_argument("--clase", default="bulldozer", help="Clase cuyo umbral se elige")
    parser.add_argument(
        "--criterio", default="precision_min", choices=["precision_min", "recall_min", "max_f1"]
    )
    parser.add_argument("--minimo", type=float, default=0.5, help="Piso del criterio")
    parser.add_argument("--empate", type=float, default=0.01)
    parser.add_argument("--efecto-minimo", type=float, default=0.02)
    parser.add_argument("--folds-minimos", type=int, default=3)
    parser.add_argument("--caida-media-caex", type=float, default=0.02)
    parser.add_argument("--caida-fold-caex", type=float, default=0.05)
    args = parser.parse_args()

    datos = cargar(args.dir)
    if "base" not in datos:
        raise SystemExit(f"No hay JSON de la variante base en {args.dir}: corré cross_validate.py")
    for variante, folds in datos.items():
        print(f"{variante}: {', '.join(sorted(folds))}", file=sys.stderr)

    resumen = resumir_lovo(
        datos,
        CriterioUmbral(tipo=args.criterio, minimo=args.minimo),
        ReglaDecision(
            empate=args.empate,
            efecto_minimo=args.efecto_minimo,
            folds_minimos=args.folds_minimos,
            caida_media_caex=args.caida_media_caex,
            caida_fold_caex=args.caida_fold_caex,
        ),
        clase_umbral=args.clase,
    )
    comando = "python scripts/summarize_lovo.py " + " ".join(sys.argv[1:])
    resumen["comando"] = comando.strip()
    (args.dir / "resumen.json").write_text(
        json.dumps(limpiar_nan(resumen), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = a_markdown(resumen, comando.strip())
    (args.dir / "resumen.md").write_text(markdown, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    print(markdown)


if __name__ == "__main__":
    main()
