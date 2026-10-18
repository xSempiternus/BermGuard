"""Tests de la cadena métrica: plano de suelo, altura del pretil y distancias.

Las cuentas se hacen a mano con el modelo pinhole. Con el horizonte en la fila
100 y un CAEX cuyo contacto está en la fila 300, un ancho de caja de 98 px implica
una cámara a ``9.8 · 200 / 98 = 20 m``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bermguard.analytics.geometry import (
    CAEX_ANCHO_NOMINAL_M,
    fit_ground_plane,
    pairwise_distances,
)
from bermguard.analytics.height import (
    INCERTIDUMBRE_MINIMA,
    GroundPlaneHeightEstimator,
    median_height,
)
from bermguard.core.exceptions import InsufficientDataError
from bermguard.core.types import BBox, BermPixels, Detection, VehicleClass

HORIZONTE = 100.0
ANCHO_FRAME = 1280


def equipo(
    x: float,
    y_suelo: float = 300.0,
    ancho: float = 98.0,
    track: int = 1,
    clase: VehicleClass = VehicleClass.CAEX,
) -> Detection:
    return Detection(
        track_id=track,
        vehicle_class=clase,
        bbox=BBox(x, y_suelo - 80, x + ancho, y_suelo),
        confidence=0.9,
    )


def pretil(cresta: float, base: float, columnas: int = ANCHO_FRAME) -> BermPixels:
    return BermPixels(
        crest_y_px=np.full(columnas, cresta, dtype=np.float32),
        base_y_px=np.full(columnas, base, dtype=np.float32),
        coverage=1.0,
        confidence=1.0,
    )


class TestPlanoDeSuelo:
    def test_altura_de_camara_desde_el_ancho_nominal(self):
        modelo = fit_ground_plane([equipo(500)], ANCHO_FRAME, HORIZONTE, 60.0)
        assert modelo.camera_height_m == pytest.approx(20.0)
        assert modelo.anchors == 1

    def test_la_mediana_ignora_un_ancla_desviada(self):
        buenos = [equipo(100, track=1), equipo(400, track=2)]
        malo = equipo(700, ancho=40.0, track=3)  # caja angosta: implica una cámara a 49 m
        modelo = fit_ground_plane([*buenos, malo], ANCHO_FRAME, HORIZONTE, 60.0)
        assert modelo.camera_height_m == pytest.approx(20.0)

    def test_sin_caex_no_hay_escala(self):
        with pytest.raises(InsufficientDataError):
            fit_ground_plane(
                [equipo(500, clase=VehicleClass.BULLDOZER)], ANCHO_FRAME, HORIZONTE, 60.0
            )


class TestAltura:
    def estimar(self, cresta: float, base: float, fov: float = 60.0):
        estimador = GroundPlaneHeightEstimator(horizontal_fov_deg=fov)
        estimador.observe_frame(HORIZONTE, ANCHO_FRAME)
        return estimador.estimate(pretil(cresta, base), [equipo(500)], frame_index=0)

    def test_formula_de_altura(self):
        # H = (y_base − y_cresta) · h / (y_base − y_h) = 10 · 20 / 200 = 1 m
        perfil = self.estimar(cresta=290, base=300)
        assert perfil is not None
        assert median_height(perfil) == pytest.approx(1.0)

    def test_la_focal_se_cancela(self):
        angosto = self.estimar(cresta=290, base=300, fov=40.0)
        amplio = self.estimar(cresta=290, base=300, fov=90.0)
        assert angosto is not None and amplio is not None
        assert median_height(angosto) == pytest.approx(median_height(amplio))

    def test_un_ancla_declara_la_incertidumbre_minima(self):
        perfil = self.estimar(cresta=290, base=300)
        assert perfil is not None
        assert float(perfil.height_m_hi[0]) == pytest.approx(1.0 + INCERTIDUMBRE_MINIMA)
        assert float(perfil.height_m_lo[0]) == pytest.approx(1.0 - INCERTIDUMBRE_MINIMA)

    def test_sin_horizonte_no_mide(self):
        estimador = GroundPlaneHeightEstimator()
        assert estimador.estimate(pretil(290, 300), [equipo(500)], 0) is None

    def test_cresta_bajo_la_base_no_es_medicion(self):
        perfil = self.estimar(cresta=310, base=300)
        assert perfil is not None
        assert np.isnan(perfil.height_m).all()
        assert median_height(perfil) is None


class TestDistancias:
    def test_distancia_lateral_independiente_de_la_focal(self):
        # Mismo contacto en la fila 300: separación lateral = Δx · h / (y − y_h).
        a, b = equipo(100, track=1), equipo(300, track=2)
        for fov in (40.0, 60.0, 90.0):
            modelo = fit_ground_plane([a, b], ANCHO_FRAME, HORIZONTE, fov)
            distancia = pairwise_distances([a, b], modelo)[(1, 2)]
            assert distancia == pytest.approx(200 * 20.0 / 200)

    def test_detecciones_sin_track_no_entran(self):
        a, b = equipo(100, track=1), equipo(300, track=-1)
        modelo = fit_ground_plane([a, b], ANCHO_FRAME, HORIZONTE, 60.0)
        assert pairwise_distances([a, b], modelo) == {}

    def test_ancho_nominal_documentado(self):
        assert pytest.approx(9.8) == CAEX_ANCHO_NOMINAL_M
