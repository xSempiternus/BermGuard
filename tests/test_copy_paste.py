"""Tests del copy-paste augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from bermguard.training.copy_paste import (
    CLASE_BULLDOZER,
    Polygon,
    extract_cutout,
    format_polygons,
    parse_polygons,
    paste,
)


def test_lee_cajas_y_poligonos_mezclados():
    texto = "0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"
    caja, poligono = parse_polygons(texto, 100, 100)
    assert caja.bbox == pytest.approx((40, 30, 60, 70))
    assert poligono.cls == CLASE_BULLDOZER
    assert poligono.bbox == pytest.approx((10, 10, 30, 30))


def test_ida_y_vuelta_del_formato():
    original = [Polygon(1, np.array([[10.0, 20.0], [30.0, 20.0], [30.0, 50.0]]))]
    leido = parse_polygons(format_polygons(original, 100, 100), 100, 100)
    assert leido[0].points == pytest.approx(original[0].points, abs=1e-4)


def rectangulo(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def escena() -> tuple[np.ndarray, list[Polygon]]:
    imagen = np.full((360, 640, 3), 120, dtype=np.uint8)
    imagen[200:260, 100:180] = 30  # un "bulldozer" oscuro
    bulldozer = Polygon(1, rectangulo(100, 200, 180, 260))
    caex = Polygon(0, rectangulo(400, 180, 520, 270))
    return imagen, [bulldozer, caex]


def test_el_pegado_no_tapa_lo_anotado_y_agrega_su_etiqueta():
    imagen, poligonos = escena()
    recorte = extract_cutout(imagen, poligonos[0])
    assert recorte is not None
    resultado = paste(imagen, poligonos, recorte, np.random.default_rng(0))
    assert resultado is not None
    nueva, etiquetas = resultado
    assert len(etiquetas) == 3
    x1, y1, x2, y2 = etiquetas[-1].bbox
    for previo in poligonos:
        px1, py1, px2, py2 = previo.bbox
        inter = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
        assert inter / ((x2 - x1) * (y2 - y1)) <= 0.1 + 1e-6
    assert not np.array_equal(nueva, imagen)


def test_el_brillo_se_ajusta_al_destino():
    imagen, poligonos = escena()
    recorte = extract_cutout(imagen, poligonos[0])
    oscura = (imagen * 0.2).astype(np.uint8)
    resultado = paste(oscura, poligonos, recorte, np.random.default_rng(1))
    assert resultado is not None
    _, etiquetas = resultado
    x1, y1, x2, y2 = (int(v) for v in etiquetas[-1].bbox)
    # Sin ajuste, el recorte (30) quedaría más brillante que su entorno nocturno (24).
    assert resultado[0][y1 + 5 : y2 - 5, x1 + 5 : x2 - 5].mean() <= 24 + 1
