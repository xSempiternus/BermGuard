"""Evalúa el detector entrenado y reporta el desglose por clase.

Herramienta de desarrollo. Existe porque las métricas que emite el entrenamiento no
bastan para juzgar el modelo:

* El **mAP global** promedia clases con comportamientos muy distintos. Con un
  desbalance de 7:1 queda dominado por la clase mayoritaria y oculta el estado de la
  minoritaria, que es justamente la razón de ser del entrenamiento.
* La **matriz de confusión** se calcula a un umbral de confianza fijo, mientras el
  mAP integra sobre todos los umbrales. Las dos lecturas pueden parecer
  contradictorias sin serlo: una clase puede tener mAP no nulo y aun así no aparecer
  nunca al umbral de operación. Compararlas es lo que revela ese caso, y es
  exactamente lo que ocurre con ``bulldozer`` en este proyecto.

``workers=0`` no es un detalle: en Windows los procesos worker de Ultralytics
re-importan el módulo principal, y si el script llega por stdin eso falla con
``OSError: Invalid argument: '<stdin>'``.

Uso:
    python scripts/eval_detector.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

PESOS = "weights/detector_v1.pt"
DATASET = "data/dataset/data.yaml"
SALIDA = Path("docs/entrenamiento/metricas_por_clase.json")


def main() -> None:
    modelo = YOLO(PESOS)
    resultado = modelo.val(
        data=DATASET,
        split="val",
        device="cuda",
        workers=0,
        verbose=False,
        plots=False,
    )
    caja = resultado.box

    # ap_class_index sólo contiene las clases con al menos una predicción, de modo
    # que sus índices no coinciden con los del modelo. Mapear a ciegas atribuiría
    # métricas a la clase equivocada.
    presentes = list(caja.ap_class_index)
    por_clase: dict[str, object] = {}
    for indice, nombre in modelo.names.items():
        if indice not in presentes:
            por_clase[nombre] = "sin predicciones"
            continue
        i = presentes.index(indice)
        por_clase[nombre] = {
            "mAP50": round(float(caja.ap50[i]), 4),
            "mAP50-95": round(float(caja.ap[i]), 4),
            "precision": round(float(caja.p[i]), 4),
            "recall": round(float(caja.r[i]), 4),
        }

    reporte = {
        "pesos": PESOS,
        "dataset": DATASET,
        "global": {
            "mAP50": round(float(caja.map50), 4),
            "mAP50-95": round(float(caja.map), 4),
            "precision": round(float(caja.mp), 4),
            "recall": round(float(caja.mr), 4),
        },
        "por_clase": por_clase,
    }

    texto = json.dumps(reporte, indent=2, ensure_ascii=False)
    print(texto)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(texto + "\n", encoding="utf-8")
    print(f"\nEscrito en {SALIDA}")


if __name__ == "__main__":
    main()
