"""Tests de la proximidad con histéresis, de los cortes de toma y de la luz.

La geometría es la de ``test_metrologia``: cámara a 20 m y contactos en la fila
300, así que cada píxel de separación lateral equivale a 0.1 m.
"""

from __future__ import annotations

import numpy as np

from bermguard.analytics.proximity import ProximityAnalyzer
from bermguard.core.types import BBox, Detection, LightingCondition, RiskLevel, VehicleClass
from bermguard.vision.lighting import LightingThresholds, classify_luma
from bermguard.vision.shots import ShotDetector

HORIZONTE = 100.0
ANCHO_FRAME = 1280


def par(separacion_m: float) -> list[Detection]:
    """Dos CAEX en la misma fila, separados ``separacion_m`` entre contactos."""
    dx = separacion_m * 10.0
    return [
        Detection(1, VehicleClass.CAEX, BBox(100, 220, 198, 300), 0.9),
        Detection(2, VehicleClass.CAEX, BBox(100 + dx, 220, 198 + dx, 300), 0.9),
    ]


def analizador() -> ProximityAnalyzer:
    return ProximityAnalyzer(
        caution_m=20.0, critical_m=10.0, frames_to_escalate=3, frames_to_deescalate=10
    )


def secuencia(proximidad: ProximityAnalyzer, separaciones: list[float]) -> list[RiskLevel]:
    return [
        proximidad.evaluate(par(s), i, HORIZONTE, ANCHO_FRAME)[1]
        for i, s in enumerate(separaciones)
    ]


class TestHisteresis:
    def test_no_sube_de_nivel_antes_de_3_frames(self):
        niveles = secuencia(analizador(), [5.0, 5.0, 5.0])
        assert niveles == [RiskLevel.SAFE, RiskLevel.SAFE, RiskLevel.CRITICAL]

    def test_un_frame_aislado_no_genera_alerta(self):
        niveles = secuencia(analizador(), [30.0, 5.0, 30.0, 5.0, 30.0])
        assert set(niveles) == {RiskLevel.SAFE}

    def test_baja_recien_despues_de_10_frames(self):
        niveles = secuencia(analizador(), [5.0] * 3 + [30.0] * 10)
        assert niveles[2] is RiskLevel.CRITICAL
        assert niveles[-2] is RiskLevel.CRITICAL
        assert niveles[-1] is RiskLevel.SAFE

    def test_precaucion_entre_umbrales(self):
        niveles = secuencia(analizador(), [15.0] * 3)
        assert niveles[-1] is RiskLevel.CAUTION

    def test_reset_olvida_el_nivel(self):
        proximidad = analizador()
        secuencia(proximidad, [5.0] * 3)
        proximidad.reset()
        assert proximidad.evaluate(par(30.0), 0, HORIZONTE, ANCHO_FRAME)[1] is RiskLevel.SAFE

    def test_un_equipo_solo_es_seguro(self):
        solo = par(5.0)[:1]
        assert analizador().evaluate(solo, 0, HORIZONTE, ANCHO_FRAME) == {1: RiskLevel.SAFE}


def frame(luma: int) -> np.ndarray:
    return np.full((36, 64, 3), luma, dtype=np.uint8)


class TestTomas:
    def test_el_primer_frame_abre_toma(self):
        assert ShotDetector().update(frame(100)) is True

    def test_un_salto_de_luminancia_es_corte(self):
        tomas = ShotDetector(luma_delta=20.0, min_length=2)
        cortes = [tomas.update(frame(v)) for v in [100, 100, 100, 160, 160]]
        assert cortes == [True, False, False, True, False]
        assert tomas.shot_index == 1

    def test_un_destello_breve_no_abre_otra_toma(self):
        tomas = ShotDetector(luma_delta=20.0, min_length=6)
        cortes = [tomas.update(frame(v)) for v in [100, 100, 180, 100, 100]]
        assert cortes == [True, False, False, False, False]


class TestLuz:
    def test_umbrales(self):
        umbrales = LightingThresholds(night_max=70.0, day_min=110.0)
        assert classify_luma(19.0, umbrales) is LightingCondition.NIGHT
        assert classify_luma(90.0, umbrales) is LightingCondition.DUSK
        assert classify_luma(163.0, umbrales) is LightingCondition.DAY
