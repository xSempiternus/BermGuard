"""Escribe ``portfolio.json``, las métricas que la landing del portafolio lee de este repo.

El esquema lo define el repo de la landing (``docs/SYNC.md`` allá): ``metricas.tiles``
como lista de ``{v, l}`` y ``metricas.resultados`` como ``{clase, mAP, recall}`` con
valores entre 0 y 1. Los campos extra se agregan bajo claves nuevas, que la
validación de la landing ignora.

Todo sale de ``docs/experimentos``: el número que ve un visitante es el mismo que
está en el reporte, sin copiarlo a mano.

Uso:
    python scripts/build_portfolio.py
    python scripts/build_portfolio.py --detector docs/experimentos/detector_v2_val.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

EXP = Path("docs/experimentos")


def _json(ruta: Path) -> dict:  # type: ignore[type-arg]
    return json.loads(ruta.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        type=Path,
        default=EXP / "detector_v1_val.json",
        help="Evaluación del detector que corre en configs/, en el video no visto",
    )
    args = parser.parse_args()
    final = _json(args.detector)
    lovo = sorted((EXP / "lovo").glob("*.json"))
    sintetico = _json(EXP / "pretil_sintetico.json")
    auditoria = _json(Path("data/auditoria_etiquetas.json"))

    def media_lovo(variante: str, clase: str) -> float:
        valores = [
            _json(r)["por_clase"][clase]["ap50"] for r in lovo if r.name.startswith(variante + "_")
        ]
        return float(np.mean(valores)) if valores else float("nan")

    # La mayor ventaja del camino óptimo en el barrido con la cresta visible
    # (contraste 18), con la cantidad de huellas a la que ocurre.
    def ventaja_de(f: dict) -> float:  # type: ignore[type-arg]
        return float(f["camino_optimo"]["within_5"] - f["argmax"]["within_5"]) * 100

    duro = max((f for f in sintetico["filas"] if f["contraste"] == 18.0), key=ventaja_de)
    ventaja = ventaja_de(duro)

    # Aceleración de la etapa del pretil a media resolución, Método 1, por video.
    escala = _json(EXP / "pretil_media_resolucion.json")
    razones = [
        escala["antes"][k]["etapa_pretil_ms"] / escala["despues"][k]["etapa_pretil_ms"]
        for k in escala["antes"]
        if k.endswith("method_1")
    ]
    caex, bull = final["por_clase"]["caex"], final["por_clase"]["bulldozer"]
    metricas = {
        "tiles": [
            {"v": f"{caex['recall']:.2f}", "l": "recall · CAEX en el video no visto"},
            {"v": str(len(auditoria["cambios"])), "l": "etiquetas corregidas en la auditoría"},
            {
                "v": f"{min(razones):.1f}–{max(razones):.1f}×",
                "l": "más rápido el pretil a media resolución",
            },
            {
                "v": f"+{ventaja:.0f} pts",
                "l": f"exactitud del pretil vs argmax, {duro['huellas']} huellas (sintético)",
            },
        ],
        "resultados": [
            {"clase": "CAEX", "mAP": round(caex["ap50"], 3), "recall": round(caex["recall"], 3)},
            {
                "clase": "Bulldozer",
                "mAP": round(bull["ap50"], 3),
                "recall": round(bull["recall"], 3),
            },
        ],
        "ic95": {
            "CAEX": caex["ic95"]["ap50"],
            "Bulldozer": bull["ic95"]["ap50"],
        },
        "umbral_operacion": final["umbral_operacion"],
    }
    # Sólo se publica la validación cruzada completa: un promedio de folds a medias
    # sería otra cifra que no se puede defender.
    completos = {
        v: {c: round(media_lovo(v, c), 3) for c in ("caex", "bulldozer")}
        for v in ("base", "copypaste")
        if len([r for r in lovo if r.name.startswith(v + "_")]) == 4
    }
    if completos:
        metricas["lovo"] = completos
    salida = {"schema": 1, "fecha": dt.date.today().isoformat(), "metricas": metricas}
    Path("portfolio.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
