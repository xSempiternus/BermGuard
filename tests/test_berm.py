"""Tests de los segmentadores del pretil, sobre escenas sintéticas.

La escena sintética trae la cresta verdadera, así que acá sí se mide exactitud y
no sólo estabilidad. Ver ``bermguard/benchmark/synthetic.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bermguard.benchmark.synthetic import SceneSpec, crest_error_px, make_scene
from bermguard.core.types import BBox, Detection, VehicleClass
from bermguard.vision.berm.argmax_baseline import ArgmaxBermSegmenter
from bermguard.vision.berm.band import search_band
from bermguard.vision.berm.classical import ClassicalBermSegmenter

SEMILLAS = range(5)


def caex(x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(
        track_id=1, vehicle_class=VehicleClass.CAEX, bbox=BBox(x1, y1, x2, y2), confidence=0.9
    )


def exactitud_media(segmentador_cls: type, spec: SceneSpec) -> float:
    """Fracción media de columnas a 5 px o menos de la cresta, sobre varias semillas."""
    valores = []
    for semilla in SEMILLAS:
        escena = make_scene(spec, seed=semilla)
        perfil = segmentador_cls().segment(escena.image)
        assert perfil is not None
        valores.append(crest_error_px(perfil.crest_y_px, escena.crest_y_px)["within_5"])
    return float(np.mean(valores))


class TestExactitud:
    @pytest.mark.parametrize("segmentador", [ClassicalBermSegmenter, ArgmaxBermSegmenter])
    def test_sin_distractores_ambos_encuentran_la_cresta(self, segmentador):
        escena = make_scene(SceneSpec(tracks=0), seed=0)
        perfil = segmentador().segment(escena.image)
        assert perfil is not None
        error = crest_error_px(perfil.crest_y_px, escena.crest_y_px)
        assert error["mae"] < 2.0
        assert error["coverage"] > 0.95

    def test_con_huellas_el_camino_optimo_supera_al_argmax(self):
        # Es la hipótesis del ADR 0006: imponer la continuidad durante la búsqueda
        # resiste los distractores cortos que el argmax por columna toma.
        spec = SceneSpec(tracks=6)
        assert (
            exactitud_media(ClassicalBermSegmenter, spec)
            > exactitud_media(ArgmaxBermSegmenter, spec) + 0.05
        )


class TestContrato:
    def test_frame_sin_estructura_no_inventa_pretil(self):
        plano = np.full((360, 640, 3), 90, dtype=np.uint8)
        assert ClassicalBermSegmenter().segment(plano) is None

    @pytest.mark.parametrize("segmentador", [ClassicalBermSegmenter, ArgmaxBermSegmenter])
    def test_banda_de_puro_ruido_no_da_pretil(self, segmentador):
        # La cresta está en la fila ~170 y la caja acota la banda a 200-314: dentro
        # sólo hay ruido. Antes de usar la mediana de la respuesta positiva como
        # suelo de ruido, esto salía con 83 % de cobertura.
        escena = make_scene(SceneSpec(tracks=0), seed=0)
        perfil = segmentador().segment(escena.image, [caex(100, 200, 198, 300)])
        assert perfil is None or perfil.coverage < 0.1

    def test_columnas_tapadas_por_maquinaria_quedan_sin_medicion(self):
        escena = make_scene(SceneSpec(tracks=0), seed=0)
        perfil = ClassicalBermSegmenter().segment(escena.image, [caex(250, 100, 400, 300)])
        assert perfil is not None
        # El interior de la caja, lejos del borde dilatado, no tiene respuesta.
        assert np.isnan(perfil.crest_y_px[290:360]).all()
        assert np.isfinite(perfil.crest_y_px[:200]).mean() > 0.9

    def test_reset_descarta_la_cresta_del_frame_anterior(self):
        abajo = make_scene(SceneSpec(tracks=0, crest_row=200.0), seed=0)
        arriba = make_scene(SceneSpec(tracks=0, crest_row=150.0), seed=1)

        sin_reset = ClassicalBermSegmenter()
        sin_reset.segment(abajo.image)
        arrastrado = sin_reset.segment(arriba.image)

        con_reset = ClassicalBermSegmenter()
        con_reset.segment(abajo.image)
        con_reset.reset()
        limpio = con_reset.segment(arriba.image)

        assert arrastrado is not None and limpio is not None
        error_arrastrado = crest_error_px(arrastrado.crest_y_px, arriba.crest_y_px)["mae"]
        error_limpio = crest_error_px(limpio.crest_y_px, arriba.crest_y_px)["mae"]
        # Sin reset, el suavizado temporal mezcla la escena anterior con la nueva.
        assert error_arrastrado > 10.0
        assert error_limpio < 2.0


class TestBanda:
    def test_sin_maquinaria_corta_el_primer_plano(self):
        gris = make_scene(SceneSpec(), seed=0).image[:, :, 0]
        y_min, y_max = search_band(gris, [])
        assert y_min == 0
        assert y_max == int(360 * 0.80)

    def test_la_maquinaria_acota_ambos_limites(self):
        gris = make_scene(SceneSpec(), seed=0).image[:, :, 0]
        y_min, y_max = search_band(gris, [caex(100, 200, 200, 300)], band_margin_frac=0.0)
        # Arriba: una altura de caja sobre el contacto. Abajo: el contacto.
        assert (y_min, y_max) == (200, 300)

    def test_el_cielo_sin_textura_queda_fuera(self):
        gris = make_scene(SceneSpec(), seed=0).image[:, :, 0].copy()
        gris[:60, :] = 200
        y_min, _ = search_band(gris, [])
        assert y_min == 60
