"""Construcción de las implementaciones concretas de cada método.

Es el único lugar del código que menciona clases concretas. El orquestador conoce
sólo los `Protocol`, de modo que incorporar un método nuevo consiste en escribir
sus implementaciones y registrarlas acá.

Esa separación tiene un efecto verificable en este proyecto: sustituir el detector
preentrenado por el especializado del ADR 0002 cambia una ruta en ``configs/`` y
nada más.
"""

from __future__ import annotations

import logging

from bermguard.core.config import PipelineConfig
from bermguard.core.exceptions import ConfigError
from bermguard.pipeline.orchestrator import Orchestrator
from bermguard.vision.berm.classical import ClassicalBermSegmenter
from bermguard.vision.detectors.yolo_detector import YoloDetector
from bermguard.vision.tracking import IouTracker

logger = logging.getLogger(__name__)

METODOS_DISPONIBLES: tuple[int, ...] = (1, 2)


def build_orchestrator(config: PipelineConfig, device: str) -> Orchestrator:
    """Arma el pipeline correspondiente al método declarado en ``config``.

    Raises:
        ConfigError: Si el método no está registrado.
    """
    if config.method not in METODOS_DISPONIBLES:
        raise ConfigError(
            f"Metodo {config.method} no registrado. Disponibles: {METODOS_DISPONIBLES}"
        )

    detector = YoloDetector(
        weights=config.detector.weights,
        device=device,
        confidence=config.detector.confidence,
        iou=config.detector.iou,
        image_size=config.detector.image_size,
        half_precision=config.detector.half_precision,
    )

    # El tracker es comun a todos los metodos: la identidad persistente no depende
    # de como se segmente el terreno.
    tracker = IouTracker(
        high_confidence=config.tracker.high_confidence,
        min_hits=config.tracker.min_hits,
        max_age=config.tracker.max_age,
        iou_high=config.tracker.iou_high,
        iou_low=config.tracker.iou_low,
        velocity_smoothing=config.tracker.velocity_smoothing,
    )

    # El metodo 1 segmenta el pretil con el prior geometrico explicito; el metodo 2
    # usara una representacion aprendida, que es el eje del benchmark.
    berm = ClassicalBermSegmenter() if config.method == 1 else None

    # Altura y proximidad aun no estan implementadas. Se pasan como
    # ausentes en lugar de con implementaciones vacias: el orquestador distingue
    # "no calculado" de "calculado y sin resultado", y esa diferencia queda
    # registrada en los avisos del metadata.json.
    return Orchestrator(
        config=config,
        detector=detector,
        device=device,
        preprocessor=None,
        tracker=tracker,
        berm_segmenter=berm,
        height_estimator=None,
        proximity=None,
    )
