"""Tests de la homogeneización de etiquetas cajas/polígonos."""

from __future__ import annotations

from bermguard.training.etiquetas import homogeneizar

CAJA = "1 0.5 0.5 0.2 0.4"
POLIGONO = "0 0.1 0.1 0.3 0.1 0.3 0.3"


def test_archivo_mixto_queda_solo_con_poligonos():
    filas = [f.split() for f in homogeneizar(f"{CAJA}\n{POLIGONO}\n").splitlines()]
    assert all(len(f) > 5 for f in filas)
    assert filas[0] == ["1", "0.400000", "0.300000", "0.600000", "0.300000"] + [
        "0.600000",
        "0.700000",
        "0.400000",
        "0.700000",
    ]
    assert filas[1] == POLIGONO.split()


def test_archivo_homogeneo_no_cambia():
    assert homogeneizar(f"{CAJA}\n{CAJA}\n") == f"{CAJA}\n{CAJA}\n"
    assert homogeneizar(f"{POLIGONO}\n") == f"{POLIGONO}\n"
    assert homogeneizar("") == ""
