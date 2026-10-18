"""Aplica la auditoría de etiquetas de ``data/auditoria_etiquetas.json``.

La auditoría encontró bulldozers etiquetados como ``caex``: pre-anotaciones del
modelo COCO, que entrega toda la maquinaria como ``truck``, que quedaron en formato
caja sin pasar por la corrección. Cada cambio registra la línea original, y el
script sólo la modifica si sigue siendo exactamente esa: es idempotente y no pisa
una corrección hecha a mano después.

Uso:
    python scripts/apply_label_audit.py            # aplica
    python scripts/apply_label_audit.py --check    # sólo informa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAIZ = Path("data/dataset/labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/auditoria_etiquetas.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    auditoria = json.loads(args.audit.read_text(encoding="utf-8"))
    aplicados = ya = distintos = 0
    por_archivo: dict[Path, list[str]] = {}
    for cambio in auditoria["cambios"]:
        ruta = RAIZ / cambio["archivo"]
        lineas = por_archivo.setdefault(ruta, ruta.read_text(encoding="utf-8").splitlines())
        actual = lineas[cambio["linea"]]
        valores = cambio["original"].split()
        corregida = " ".join([str(cambio["a"]), *valores[1:]])
        if actual == corregida:
            ya += 1
        elif actual == cambio["original"]:
            lineas[cambio["linea"]] = corregida
            aplicados += 1
        else:
            distintos += 1
            print(f"! {ruta}:{cambio['linea']} cambió desde la auditoría; no se toca")

    if not args.check:
        for ruta, lineas in por_archivo.items():
            ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    verbo = "Por aplicar" if args.check else "Aplicados"
    print(f"{verbo}: {aplicados}. Ya aplicados: {ya}. Modificados a mano: {distintos}.")
    print(f"Pendientes de revisión manual: {len(auditoria['pendientes'])}")


if __name__ == "__main__":
    main()
