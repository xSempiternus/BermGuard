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

from bermguard.analytics.height import GroundPlaneHeightEstimator
from bermguard.analytics.proximity import ProximityAnalyzer
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
        duplicate_containment=config.tracker.duplicate_containment,
        velocity_smoothing=config.tracker.velocity_smoothing,
    )

    # El metodo 1 segmenta el pretil con el prior geometrico explicito; el metodo 2
    # usara una representacion aprendida, que es el eje del benchmark.
    berm = ClassicalBermSegmenter() if config.method == 1 else None

    proximity = ProximityAnalyzer(
        caution_m=config.proximity.caution_m,
        critical_m=config.proximity.critical_m,
        frames_to_escalate=config.proximity.frames_to_escalate,
        frames_to_deescalate=config.proximity.frames_to_deescalate,
        horizontal_fov_deg=config.proximity.horizontal_fov_deg,
    )

    # El estimador de altura y el de proximidad comparten el campo de vision
    # asumido, porque ambos derivan del mismo modelo de plano de suelo. Tenerlo en
    # un solo lugar de la configuracion evita que discrepen.
    altura = GroundPlaneHeightEstimator(horizontal_fov_deg=config.proximity.horizontal_fov_deg)

    # El preprocesador se pasa ausente y no como una implementacion vacia: el
    # detector opera sobre frames sin acondicionar por decision explicita (ADR 0005).
    return Orchestrator(
        config=config,
        detector=detector,
        device=device,
        preprocessor=None,
        tracker=tracker,
        berm_segmenter=berm,
        height_estimator=altura,
        proximity=proximity,
    )
