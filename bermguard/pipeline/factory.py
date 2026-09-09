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
from bermguard.vision.detectors.yolo_detector import YoloDetector

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

    # Las etapas de terreno, tracking y proximidad aun no estan implementadas.
    # Se pasan como ausentes en lugar de con implementaciones vacias: el
    # orquestador distingue "no calculado" de "calculado y sin resultado", y esa
    # diferencia queda registrada en los avisos del metadata.json.
    return Orchestrator(
        config=config,
        detector=detector,
        device=device,
        preprocessor=None,
        tracker=None,
        berm_segmenter=None,
        height_estimator=None,
        proximity=None,
    )
