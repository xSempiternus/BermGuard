"""Evaluación del detector con incertidumbre, en el punto de operación real.

Existe porque las métricas de Ultralytics no alcanzan para este proyecto:

* **La precisión y el recall que reporta son los del umbral que maximiza el F1**,
  no los del umbral con el que opera el pipeline (0.25 en ``configs/``). Un
  supervisor ve las alertas del umbral de operación, así que es ahí donde hay que
  medir.
* **No trae intervalos.** Con 11 bulldozers en validación, un punto de mAP no dice
  nada sin saber cuánto se movería con otra muestra.

El AP se calcula como en COCO: 101 puntos de recall, precisión interpolada, una
detección es correcta si su IoU con una caja anotada de la misma clase es al menos
0.5 y cada caja anotada se empareja una sola vez.

El bootstrap remuestrea **bloques de frames consecutivos** y no frames sueltos,
porque dos frames vecinos son casi la misma imagen. Remuestrearlos por separado
trataría como independientes observaciones que no lo son y daría intervalos
demasiado angostos.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UMBRAL_IOU = 0.5


@dataclass(frozen=True, slots=True)
class Box:
    """Caja en píxeles, con clase y confianza (1.0 para las anotadas)."""

    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float = 1.0


@dataclass(slots=True)
class ImageEval:
    """Anotaciones y predicciones de una imagen."""

    name: str
    truth: list[Box]
    preds: list[Box]
    group: str = ""
    """Bloque de frames consecutivos al que pertenece, para el bootstrap."""
    tags: dict[str, str] = field(default_factory=dict)
    """Atributos para desglosar, como la condición de luz."""


def read_yolo_labels(path: Path, width: int, height: int) -> list[Box]:
    """Lee un archivo de etiquetas YOLO, en formato caja o polígono.

    Las anotaciones del proyecto son polígonos (``cls x1 y1 x2 y2 ...``). La caja es
    la envolvente del polígono, que es lo mismo que hace Ultralytics al entrenar un
    detector de cajas con etiquetas de segmentación.
    """
    cajas: list[Box] = []
    if not path.exists():
        return cajas
    for linea in path.read_text(encoding="utf-8").splitlines():
        valores = linea.split()
        if len(valores) < 5:
            continue
        cls = int(valores[0])
        coords = np.asarray(valores[1:], dtype=np.float64)
        if coords.size == 4:
            cx, cy, w, h = coords
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        cajas.append(Box(x1 * width, y1 * height, x2 * width, y2 * height, cls))
    return cajas


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def _emparejar(imagen: ImageEval, cls: int) -> tuple[list[tuple[float, bool]], int]:
    """Marca cada predicción de la clase como acierto o falso positivo.

    Returns:
        ``(confianza, es_acierto)`` por predicción y el número de cajas anotadas.
    """
    verdad = [b for b in imagen.truth if b.cls == cls]
    preds = sorted((b for b in imagen.preds if b.cls == cls), key=lambda b: -b.conf)
    usadas = [False] * len(verdad)
    marcas: list[tuple[float, bool]] = []
    for pred in preds:
        mejor, indice = 0.0, -1
        for j, caja in enumerate(verdad):
            if usadas[j]:
                continue
            valor = iou(pred, caja)
            if valor > mejor:
                mejor, indice = valor, j
        acierto = mejor >= UMBRAL_IOU
        if acierto:
            usadas[indice] = True
        marcas.append((pred.conf, acierto))
    return marcas, len(verdad)


def class_metrics(images: Sequence[ImageEval], cls: int, operating_conf: float) -> dict[str, float]:
    """AP@0.5 de una clase y precisión, recall y F1 en el umbral de operación."""
    marcas: list[tuple[float, bool]] = []
    total = 0
    for imagen in images:
        m, n = _emparejar(imagen, cls)
        marcas.extend(m)
        total += n
    if total == 0:
        return {
            "ap50": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "n_truth": 0.0,
        }

    marcas.sort(key=lambda t: -t[0])
    confs = np.asarray([c for c, _ in marcas])
    aciertos = np.asarray([a for _, a in marcas], dtype=float)
    tp = np.cumsum(aciertos)
    fp = np.cumsum(1.0 - aciertos)
    recall = tp / total
    precision = tp / np.maximum(tp + fp, 1e-9)

    # Precision interpolada (envolvente monotona) muestreada en 101 recalls, como COCO.
    envolvente = np.maximum.accumulate(precision[::-1])[::-1] if precision.size else precision
    puntos = np.linspace(0, 1, 101)
    ap = 0.0
    for r in puntos:
        cumplen = envolvente[recall >= r]
        ap += float(cumplen.max()) if cumplen.size else 0.0
    ap /= puntos.size

    en_operacion = confs >= operating_conf
    tp_op = float(aciertos[en_operacion].sum())
    fp_op = float(en_operacion.sum()) - tp_op
    p_op = tp_op / (tp_op + fp_op) if (tp_op + fp_op) > 0 else float("nan")
    r_op = tp_op / total
    f1 = 2 * p_op * r_op / (p_op + r_op) if p_op == p_op and (p_op + r_op) > 0 else 0.0
    return {"ap50": ap, "precision": p_op, "recall": r_op, "f1": f1, "n_truth": float(total)}


def threshold_sweep(
    images: Sequence[ImageEval], cls: int, thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """Precisión, recall y F1 de una clase para cada umbral de confianza."""
    filas = []
    for umbral in thresholds:
        m = class_metrics(images, cls, umbral)
        filas.append(
            {
                "conf": float(umbral),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )
    return filas


def bootstrap_ci(
    images: Sequence[ImageEval],
    cls: int,
    operating_conf: float,
    replicas: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Intervalo de confianza por bootstrap de bloques para cada métrica.

    Remuestrea grupos completos (``ImageEval.group``) con reposición. Las réplicas
    sin ninguna caja anotada de la clase se descartan, porque en ellas la métrica no
    está definida.
    """
    grupos: dict[str, list[ImageEval]] = {}
    for imagen in images:
        grupos.setdefault(imagen.group or imagen.name, []).append(imagen)
    claves = sorted(grupos)
    rng = np.random.default_rng(seed)
    muestras: dict[str, list[float]] = {"ap50": [], "precision": [], "recall": [], "f1": []}
    for _ in range(replicas):
        elegidos = rng.choice(len(claves), size=len(claves), replace=True)
        muestra = [img for i in elegidos for img in grupos[claves[i]]]
        m = class_metrics(muestra, cls, operating_conf)
        if m["n_truth"] == 0:
            continue
        for k in muestras:
            if m[k] == m[k]:  # descarta NaN
                muestras[k].append(m[k])
    cola = (1 - level) / 2 * 100
    return {
        k: (float(np.percentile(v, cola)), float(np.percentile(v, 100 - cola)))
        if v
        else (float("nan"), float("nan"))
        for k, v in muestras.items()
    }
