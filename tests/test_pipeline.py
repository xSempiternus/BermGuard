"""Test de integración: el orquestador completo sobre un video sintético.

El detector es un doble que devuelve dos CAEX fijos; el resto de las etapas son
las reales. Comprueba que el pipeline escribe todos los artefactos que promete y
que las métricas del metadata salen con valores coherentes.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from bermguard.analytics.height import GroundPlaneHeightEstimator
from bermguard.analytics.proximity import ProximityAnalyzer
from bermguard.benchmark.synthetic import SceneSpec, make_scene
from bermguard.core.config import config_for_method
from bermguard.core.types import BBox, Detection, VehicleClass
from bermguard.pipeline.orchestrator import Orchestrator
from bermguard.vision.berm.classical import ClassicalBermSegmenter
from bermguard.vision.tracking import IouTracker

FRAMES = 24


class DetectorFijo:
    """Dos CAEX quietos, con la cresta sintética dentro de su banda de búsqueda."""

    name = "doble:fijo"

    def detect(self, frame):
        return [
            Detection(-1, VehicleClass.CAEX, BBox(100, 120, 198, 300), 0.9),
            Detection(-1, VehicleClass.CAEX, BBox(220, 120, 318, 300), 0.9),
        ]


@pytest.fixture
def video(tmp_path: Path) -> Path:
    destino = tmp_path / "sintetico.mp4"
    writer = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (640, 360))
    for i in range(FRAMES):
        writer.write(make_scene(SceneSpec(tracks=0), seed=i, phase=0.01 * i).image)
    writer.release()
    return destino


def test_pipeline_escribe_todos_los_artefactos(video: Path, tmp_path: Path):
    config = config_for_method(1, Path("configs"))
    orquestador = Orchestrator(
        config=config,
        detector=DetectorFijo(),
        device="cpu",
        tracker=IouTracker(),
        berm_segmenter=ClassicalBermSegmenter(),
        height_estimator=GroundPlaneHeightEstimator(),
        proximity=ProximityAnalyzer(),
    )
    salida = tmp_path / "out"
    metadata = orquestador.process_video(video, salida)

    esperados = {
        "sintetico_osd.mp4",
        "metadata.json",
        "berm_profile.csv",
        "proximity_events.csv",
        "berm_height.png",
    }
    assert esperados <= {p.name for p in salida.iterdir()}

    escrito = json.loads((salida / "metadata.json").read_text(encoding="utf-8"))
    assert escrito["frames_processed"] == FRAMES == metadata.frames_processed
    assert escrito["berm_coverage_mean"] > 0.5
    # La escena se mueve 0.01 rad por frame: columna a columna la cresta casi no salta.
    assert escrito["berm_crest_jitter_col_px"] < 1.0
    assert "berm" in escrito["timing"]["stage_ms"]
