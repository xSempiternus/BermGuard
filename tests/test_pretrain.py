"""Tests del remapeo de datasets públicos a las clases del proyecto."""

from __future__ import annotations

import pytest

from bermguard.training.pretrain import parse_class_map, remap_labels

NOMBRES = ["Bull_dozer", "excavator", "tippertruck"]


def test_variantes_de_nombre_son_la_misma_clase():
    mapa = parse_class_map(["bull dozer=bulldozer", "Tipper-Truck=caex"])
    assert mapa == {"bulldozer": 1, "tippertruck": 0}


def test_destino_desconocido_falla():
    with pytest.raises(ValueError):
        parse_class_map(["dozer=excavadora"])


def test_reasigna_indices_y_descarta_clases_sin_traduccion():
    mapa = parse_class_map(["bull_dozer=bulldozer", "tippertruck=caex"])
    texto = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n2 0.1 0.1 0.3 0.1 0.3 0.3\n"
    assert remap_labels(texto, NOMBRES, mapa) == "1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.3 0.1 0.3 0.3\n"


def test_imagen_sin_clases_traducidas_queda_vacia():
    mapa = parse_class_map(["tippertruck=caex"])
    assert remap_labels("1 0.5 0.5 0.2 0.2\n9 0.1 0.1 0.1 0.1\n", NOMBRES, mapa) == ""
