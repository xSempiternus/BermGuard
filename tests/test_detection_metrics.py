"""Tests del evaluador de detección: AP, punto de operación y bootstrap."""

from __future__ import annotations

import pytest

from bermguard.benchmark.detection import (
    Box,
    ImageEval,
    bootstrap_ci,
    class_metrics,
    read_yolo_labels,
)


def caja(x: float, conf: float = 1.0, cls: int = 0) -> Box:
    return Box(x, 0, x + 10, 10, cls=cls, conf=conf)


def test_detector_perfecto():
    imagenes = [ImageEval(f"i{k}", [caja(0)], [caja(0, 0.9)]) for k in range(4)]
    m = class_metrics(imagenes, 0, 0.25)
    assert m["ap50"] == pytest.approx(1.0)
    assert (m["precision"], m["recall"]) == (1.0, 1.0)


def test_el_punto_de_operacion_excluye_lo_que_esta_bajo_el_umbral():
    # Un acierto con 0.9 y uno con 0.1: al umbral 0.25 sólo cuenta el primero.
    imagenes = [
        ImageEval("a", [caja(0)], [caja(0, 0.9)]),
        ImageEval("b", [caja(0)], [caja(0, 0.1)]),
    ]
    m = class_metrics(imagenes, 0, 0.25)
    assert m["recall"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(1.0)
    # El AP sí integra todos los umbrales.
    assert m["ap50"] == pytest.approx(1.0)


def test_duplicado_cuenta_como_falso_positivo():
    imagenes = [ImageEval("a", [caja(0)], [caja(0, 0.9), caja(1, 0.8)])]
    m = class_metrics(imagenes, 0, 0.25)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)


def test_falso_positivo_con_mas_confianza_baja_el_ap():
    imagenes = [ImageEval("a", [caja(0)], [caja(50, 0.95), caja(0, 0.5)])]
    assert class_metrics(imagenes, 0, 0.25)["ap50"] == pytest.approx(0.5, abs=0.01)


def test_el_intervalo_contiene_la_estimacion_puntual():
    imagenes = [
        ImageEval(f"i{k}", [caja(0)], [caja(0, 0.9)] if k % 3 else [caja(40, 0.9)], group=f"g{k}")
        for k in range(30)
    ]
    puntual = class_metrics(imagenes, 0, 0.25)["recall"]
    bajo, alto = bootstrap_ci(imagenes, 0, 0.25, replicas=300)["recall"]
    assert bajo <= puntual <= alto
    assert alto - bajo > 0


def test_lee_poligonos_como_envolvente(tmp_path):
    archivo = tmp_path / "a.txt"
    archivo.write_text("1 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n", encoding="utf-8")
    (leida,) = read_yolo_labels(archivo, 100, 100)
    assert (leida.cls, leida.x1, leida.y1, leida.x2, leida.y2) == pytest.approx((1, 10, 20, 30, 60))
